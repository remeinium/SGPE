"""
WWHO(SGPE) GPE Trainer
"""

import argparse
import gc
import heapq
import json
import logging
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from router import CodeSwitchSegmenter, Script
from export import export_hf_tokenizer

# ─── Logging ──────

try:
    import psutil as _psutil
    def _ram_mb() -> str:
        p = _psutil.Process()
        rss = p.memory_info().rss / 1024**2
        avail = _psutil.virtual_memory().available / 1024**2
        return f"RSS={rss:.0f}MB avail={avail:.0f}MB"
except ImportError:
    def _ram_mb() -> str:
        try:
            with open("/proc/meminfo") as f:
                info = {l.split(":")[0]: int(l.split()[1])
                        for l in f if ":" in l}
            avail = info.get("MemAvailable", 0) // 1024
            return f"avail={avail}MB"
        except Exception:
            return "ram=N/A"

_logger: logging.Logger | None = None

def _log(msg: str):
    full = f"[{time.strftime('%H:%M:%S')}] [{_ram_mb()}] {msg}"
    print(full, flush=True)
    if _logger:
        _logger.info(full)

def _setup_logging(output_dir: str):
    global _logger
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(message)s",
    )
    _logger = logging.getLogger("wwho_trainer")
    _log(f"Log started: {log_path}")


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

from router import CodeSwitchSegmenter, Script, _get_char_script

# ─── Multiprocessing ──────
_worker_segmenter: CodeSwitchSegmenter | None = None
_worker_sinhala_trie = None
_worker_devanagari_trie = None
_worker_script_mode: str = "mixed"


def _init_worker(script_mode: str):
    global _worker_segmenter, _worker_sinhala_trie, _worker_devanagari_trie, _worker_script_mode
    from linguis_trie import build_sinhala_linguis_trie, build_devanagari_linguis_trie
    _worker_segmenter     = CodeSwitchSegmenter()
    _worker_sinhala_trie   = build_sinhala_linguis_trie()
    _worker_script_mode   = script_mode
    if script_mode in ("devanagari", "mixed"):
        _worker_devanagari_trie = build_devanagari_linguis_trie()


def _pretokenize_line(text: str) -> list[str]:
    tokens: list[str] = []

    for seg in _worker_segmenter.segment(text):
        if seg.script == Script.LATIN:
            tokens.append(seg.text)
        elif seg.script == Script.SINHALA:
            syllables = _worker_sinhala_trie.tokenize(
                seg.text, leading_space=seg.has_leading_space
            )
            tokens.extend(syllables)
        elif seg.script == Script.DEVANAGARI:
            if _worker_script_mode == "sinhala":
                # Devanagari excluded 
                tokens.append(seg.text)
            else:
                syllables = _worker_devanagari_trie.tokenize(
                    seg.text, leading_space=seg.has_leading_space
                )
                tokens.extend(syllables)
        else:
            # Unknown script — pass through as boundary token
            tokens.append(seg.text)

    return tokens


def _is_boundary_token(token: str) -> bool:
    for ch in token:
        if _get_char_script(ch) is not None:
            return False
    return True

def segment_into_words(syllables: list[str]) -> list[list[str]]:
    words: list[list[str]] = []
    current: list[str] = []

    for tok in syllables:
        if _is_boundary_token(tok):
            if current:
                words.append(current)
                current = []
            words.append([tok])
        else:
            if tok[0] in (' ', '\t', '\n', '\r') and current:
                words.append(current)
                current = []
            current.append(tok)

    if current:
        words.append(current)
    return words


# ─── Symbol Table  ──────

class SymbolTable:
    def __init__(self):
        self._str_to_id: dict[str, int] = {}
        self._id_to_str: list[str] = []

    def get_or_add(self, token: str) -> int:
        if token in self._str_to_id:
            return self._str_to_id[token]
        new_id = len(self._id_to_str)
        self._str_to_id[token] = new_id
        self._id_to_str.append(token)
        return new_id

    def add_merged(self, a_id: int, b_id: int) -> int:
        merged_str = self._id_to_str[a_id] + self._id_to_str[b_id]
        return self.get_or_add(merged_str)

    def to_str(self, token_id: int) -> str:
        return self._id_to_str[token_id]

    def to_id(self, token: str) -> int | None:
        return self._str_to_id.get(token)

    def __len__(self) -> int:
        return len(self._id_to_str)


# ─── GPETrainer ──────

class GPETrainer:

    def __init__(
        self,
        vocab_size: int = 200_000,
        min_freq: int = 2,
        num_workers: int | None = None,
        checkpoint_every: int = 10_000,
        prune_freq: int = 50,
        script_mode: str = "mixed",
    ):
        self.target_vocab_size = vocab_size
        self.min_freq = min_freq
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.checkpoint_every = checkpoint_every
        self.prune_freq = prune_freq
        self.script_mode = script_mode
        self.merges: list[tuple[int, int]] = []
        self.symbols = SymbolTable()

    def stream_and_count(
        self, train_file: str, output_dir: str = "output"
    ) -> tuple[Counter, set[str]]:
        # ── 1. Count lines ──────
        print("  counting lines...", end=" ", flush=True)
        with open(train_file, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
        print(f"{num_lines:,}")

        CHUNK_SIZE = 5_000_000  # max sentences per Pool lifetime
        BATCH      = 4_096      # sentences per pool.map() call

        partial_dir = os.path.join(output_dir, "_partial_counters")
        os.makedirs(partial_dir, exist_ok=True)

        total_lines = 0
        chunk_idx   = 0
        partial_paths: list[str] = []

        PARTIAL_PRUNE = 2
        def _save_partial(counter: Counter, idx: int, n_sent: int):
            if PARTIAL_PRUNE > 1:
                to_save = Counter(
                    {k: v for k, v in counter.items() if v >= PARTIAL_PRUNE}
                )
            else:
                to_save = counter
            pkl_path = os.path.join(partial_dir, f"partial_{idx:04d}.pkl")
            with open(pkl_path, "wb") as pf:
                pickle.dump(to_save, pf, protocol=pickle.HIGHEST_PROTOCOL)
            partial_paths.append(pkl_path)
            pkl_mb = os.path.getsize(pkl_path) / 1024**2
            pbar.write(
                f"  chunk {idx+1} done: {n_sent:,} sent "
                f"-> {len(to_save):,} word types (pruned from {len(counter):,}) "
                f"-> {pkl_path} ({pkl_mb:.0f} MB)"
            )
            _log(f"CHUNK {idx+1} saved: {n_sent:,} sent, "
                 f"{len(to_save):,} word types, {pkl_mb:.0f} MB")
            del to_save
            counter.clear()
            gc.collect()
            _log(f"CHUNK {idx+1} post-gc")

        chunk_counter: Counter = Counter()
        chunk_sent  = 0
        batch_buf:  list[str] = []

        pool = Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.script_mode,),
        )

        with open(train_file, "r", encoding="utf-8") as f:
            pbar = tqdm(f, total=num_lines, unit=" sent",
                        desc=f"  pre-tokenizing [chunk 1]")

            for raw_line in pbar:
                try:
                    obj  = json.loads(raw_line)
                    text = obj.get("text", "").strip()
                except json.JSONDecodeError:
                    text = raw_line.strip()
                if not text:
                    continue

                batch_buf.append(text)
                total_lines += 1
                chunk_sent  += 1

                if len(batch_buf) >= BATCH:
                    self._process_batch(pool, batch_buf, chunk_counter)
                    batch_buf = []
                if chunk_sent >= CHUNK_SIZE:
                    if batch_buf:
                        self._process_batch(pool, batch_buf, chunk_counter)
                        batch_buf = []
                    pool.close()
                    pool.join()
                    pool = None
                    gc.collect()

                    _save_partial(chunk_counter, chunk_idx, chunk_sent)
                    chunk_idx  += 1
                    chunk_sent  = 0
                    pbar.set_description(
                        f"  pre-tokenizing [chunk {chunk_idx + 1}]"
                    )
                    gc.collect()

                    pool = Pool(
                        processes=self.num_workers,
                        initializer=_init_worker,
                        initargs=(self.script_mode,),
                    )

            if batch_buf:
                self._process_batch(pool, batch_buf, chunk_counter)
            pool.close()
            pool.join()
            gc.collect()

            if chunk_counter:
                _save_partial(chunk_counter, chunk_idx, chunk_sent)
                chunk_idx += 1

            pbar.close()

        print(f"  {total_lines:,} sentences -> {chunk_idx} chunks processed")

        # ── 3. Sequential merge with intermediate pruning ──────
        _log(f"MERGE START: {len(partial_paths)} partial counters, min_freq={self.min_freq}")
        N = len(partial_paths)
        word_counter: Counter = Counter()
        for i, pkl_path in enumerate(partial_paths):
            _log(f"MERGE [{i+1}/{N}] loading {pkl_path}")
            with open(pkl_path, "rb") as pf:
                partial: Counter = pickle.load(pf)
            _log(f"MERGE [{i+1}/{N}] loaded {len(partial):,} types, updating master...")
            word_counter.update(partial)
            del partial
            gc.collect()
            _log(f"MERGE [{i+1}/{N}] after update+gc: {len(word_counter):,} types")

            remaining = N - i - 1
            safe_prune = max(1, self.min_freq - remaining)
            before = len(word_counter)
            
            if safe_prune > 1:
                word_counter = Counter(
                    {k: v for k, v in word_counter.items() if v >= safe_prune}
                )
            
            if i > 0 and i % 5 == 0:
                hard_threshold = max(2, self.min_freq // 2) 
                word_counter = Counter(
                    {k: v for k, v in word_counter.items() if v >= hard_threshold}
                )
                _log(f"MERGE [{i+1}/{N}] HARD PRUNE TRIGGERED (threshold={hard_threshold})")

            gc.collect()
            pruned_n = before - len(word_counter)
            
            if pruned_n > 0:
                msg = (f"    [{i+1}/{N}] merged -> {len(word_counter):,} types "
                       f"(pruned {pruned_n:,})")
                print(msg, flush=True)
                _log(f"MERGE [{i+1}/{N}] post-prune: {len(word_counter):,} types "
                     f"(removed {pruned_n:,})")
            else:
                print(f"    [{i+1}/{N}] merged -> {len(word_counter):,} types", flush=True)
                _log(f"MERGE [{i+1}/{N}] no prune needed, {len(word_counter):,} types")
                
            os.remove(pkl_path)
            _log(f"MERGE [{i+1}/{N}] deleted {pkl_path}")

        try:
            os.rmdir(partial_dir)
        except OSError:
            pass

        n_types     = len(word_counter)
        n_instances = sum(word_counter.values())
        print(f"\n  Final: {total_lines:,} sent -> {n_types:,} word types "
              f"({n_instances:,} instances)")
        return word_counter, set()  

    def _process_batch(
        self,
        pool: Pool,
        batch: list[str],
        word_counter: Counter,
    ):
        syllable_streams = pool.map(_pretokenize_line, batch, chunksize=128)

        for stream in syllable_streams:
            words = segment_into_words(stream)
            for w in words:
                if not w:
                    continue
                if not _is_boundary_token(w[0]):
                    word_counter[tuple(w)] += 1

    @staticmethod
    def compute_syllable_freqs(word_counter: Counter) -> Counter:
        syl_freq: Counter[str] = Counter()
        for word_tuple, word_freq in word_counter.items():
            for syl in word_tuple:
                syl_freq[syl] += word_freq
        return syl_freq

    def build_word_types(
        self,
        word_counter: Counter,
        boundary_tokens: set[str], 
        syl_freq: Counter | None = None,
    ) -> tuple[list[list[int]], list[int]]:
        UNK_SENTINEL = -1
        pruned_set: set[str] = set()

        if syl_freq is not None and self.prune_freq > 0:
            for syl, freq in syl_freq.items():
                if freq < self.prune_freq:
                    pruned_set.add(syl)

        word_types: list[list[int]] = []
        word_freqs: list[int] = []
        pruned_word_count = 0

        for word_tuple, freq in word_counter.items():
            ids = []
            for tok in word_tuple:
                if tok in pruned_set:
                    ids.append(UNK_SENTINEL)
                else:
                    ids.append(self.symbols.get_or_add(tok))
            word_types.append(ids)
            word_freqs.append(freq)
            if UNK_SENTINEL in ids:
                pruned_word_count += 1

        if pruned_set:
            print(f"  pruned {len(pruned_set):,} rare syllables (freq < {self.prune_freq})")
            print(f"  {pruned_word_count:,} word types contain [UNK] syllables")

        return word_types, word_freqs

    @staticmethod
    def build_token_index(word_types: list[list[int]]) -> dict[int, set[int]]:
        index: dict[int, set[int]] = defaultdict(set)
        for wt_idx, wt in enumerate(word_types):
            for tid in wt:
                if tid >= 0:
                    index[tid].add(wt_idx)
        return dict(index)

    def count_all_pairs(
        self,
        word_types: list[list[int]],
        word_freqs: list[int],
    ) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for wt_idx, wt in enumerate(word_types):
            f = word_freqs[wt_idx]
            for i in range(len(wt) - 1):
                a, b = wt[i], wt[i + 1]
                if a < 0 or b < 0:
                    continue
                counts[(a, b)] += f
        return dict(counts)

    @staticmethod
    def _build_heap(pair_counts: dict) -> list:
        heap = [(-freq, pair) for pair, freq in pair_counts.items() if freq > 0]
        heapq.heapify(heap)
        return heap

    @staticmethod
    def _heap_push(heap, pair, freq):
        if freq > 0:
            heapq.heappush(heap, (-freq, pair))

    def _pop_best(self, heap, pair_counts):
        while heap:
            neg_freq, pair = heapq.heappop(heap)
            actual = pair_counts.get(pair, 0)
            if actual <= 0:
                continue
            if actual != -neg_freq:
                self._heap_push(heap, pair, actual)
                continue
            return pair, actual
        return None, 0

    def merge_and_update(
        self,
        word_types: list[list[int]],
        word_freqs: list[int],
        pair: tuple[int, int],
        pair_counts: dict[tuple[int, int], int],
        token_index: dict[int, set[int]],
        merged_id: int,
        heap: list,
    ) -> int:
        a, b = pair
        total_applied = 0
        candidates = list(token_index.get(a, set()) & token_index.get(b, set()))
        pair_counts.pop(pair, None)
        dirty_pairs: dict[tuple[int, int], int] = {}

        for wt_idx in candidates:
            wt = word_types[wt_idx]
            freq = word_freqs[wt_idx]
            if len(wt) < 2:
                continue
            new_wt: list[int] = []
            i = 0
            changed = False

            while i < len(wt):
                if i + 1 < len(wt) and wt[i] == a and wt[i + 1] == b:
                    if new_wt and new_wt[-1] >= 0:
                        lp = (new_wt[-1], a)
                        pair_counts[lp] = pair_counts.get(lp, 0) - freq
                        dirty_pairs[lp] = pair_counts[lp]
                    if i + 2 < len(wt) and wt[i + 2] >= 0:
                        rp = (b, wt[i + 2])
                        pair_counts[rp] = pair_counts.get(rp, 0) - freq
                        dirty_pairs[rp] = pair_counts[rp]
                    new_wt.append(merged_id)
                    total_applied += freq
                    changed = True
                    if len(new_wt) >= 2 and new_wt[-2] >= 0:
                        lp = (new_wt[-2], merged_id)
                        pair_counts[lp] = pair_counts.get(lp, 0) + freq
                        dirty_pairs[lp] = pair_counts[lp]
                    if i + 2 < len(wt) and wt[i + 2] >= 0:
                        rp = (merged_id, wt[i + 2])
                        pair_counts[rp] = pair_counts.get(rp, 0) + freq
                        dirty_pairs[rp] = pair_counts[rp]
                    i += 2
                else:
                    new_wt.append(wt[i])
                    i += 1

            if changed:
                word_types[wt_idx] = new_wt
                if merged_id not in token_index:
                    token_index[merged_id] = set()
                token_index[merged_id].add(wt_idx)
                remaining = set(new_wt)
                if a not in remaining and wt_idx in token_index.get(a, set()):
                    token_index[a].discard(wt_idx)
                if b not in remaining and wt_idx in token_index.get(b, set()):
                    token_index[b].discard(wt_idx)

        for tok_id in (a, b):
            if tok_id in token_index and not token_index[tok_id]:
                del token_index[tok_id]

        for p, cnt in dirty_pairs.items():
            if cnt <= 0:
                pair_counts.pop(p, None)
            else:
                self._heap_push(heap, p, cnt)

        return total_applied

    def save_checkpoint(self, step: int, output_dir: str, elapsed: float):
        merge_strs = [
            [self.symbols.to_str(a), self.symbols.to_str(b)]
            for a, b in self.merges
        ]
        ckpt = {
            "step": step,
            "script_mode": self.script_mode,
            "merges": merge_strs,
            "elapsed_seconds": round(elapsed, 1),
        }
        path = os.path.join(output_dir, f"checkpoint_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return path, size_mb

    def load_checkpoint(self, ckpt_path: str):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        print(f"  loaded checkpoint: step {ckpt['step']}, "
              f"{len(ckpt['merges'])} merges, "
              f"{ckpt['elapsed_seconds']:.1f}s elapsed")
        return ckpt

    def replay_merges(self, merge_strs, word_types, word_freqs, token_index, pair_counts):
        print(f"  replaying {len(merge_strs)} merges...", flush=True)
        t0 = time.time()
        dummy_heap: list = []
        for a_str, b_str in tqdm(merge_strs, desc="  replaying", unit=" merge"):
            a_id = self.symbols.to_id(a_str)
            b_id = self.symbols.to_id(b_str)
            if a_id is None or b_id is None:
                continue
            merged_id = self.symbols.add_merged(a_id, b_id)
            self.merges.append((a_id, b_id))
            self.merge_and_update(
                word_types, word_freqs, (a_id, b_id), pair_counts,
                token_index, merged_id, dummy_heap,
            )
        print(f"  replayed {len(self.merges)} merges in {time.time()-t0:.1f}s")

    def train(self, train_file: str, output_dir: str = "output",
              resume_path: str | None = None):
        os.makedirs(output_dir, exist_ok=True)

        print(f"WWHO (SGPE) GPE Trainer — script_mode={self.script_mode}, "
              f"workers={self.num_workers}")
        print(f"Training file: {train_file}\n")

        print("[1/5] Streaming pre-tokenization (CodeSwitchRouter)...")
        t_start = time.time()
        word_counter, boundary_tokens = self.stream_and_count(train_file, output_dir)

        print("\n[2/5] Building ID corpus...")
        syl_freq = None
        if self.prune_freq > 0:
            syl_freq = self.compute_syllable_freqs(word_counter)
            total_syls = len(syl_freq)
            surviving = sum(1 for f in syl_freq.values() if f >= self.prune_freq)
            print(f"  syllable pruning: {total_syls:,} unique syllables, "
                  f"{surviving:,} survive (freq >= {self.prune_freq})")

        word_types, word_freqs = self.build_word_types(
            word_counter, boundary_tokens, syl_freq=syl_freq,
        )
        del word_counter, syl_freq

        base_vocab = len(self.symbols)
        total_instances = sum(word_freqs)
        print(f"  base vocab (syllables + boundaries): {base_vocab:,}")
        print(f"  word types: {len(word_types):,} ({total_instances:,} instances)")

        print("\n[3/5] Building index and counting pairs...")
        token_index = self.build_token_index(word_types)
        pair_counts = self.count_all_pairs(word_types, word_freqs)
        print(f"  {len(pair_counts):,} unique pairs")

        start_step = 0
        elapsed_prior = 0.0
        if resume_path:
            print(f"\n  Resuming from {resume_path}...")
            ckpt = self.load_checkpoint(resume_path)
            self.replay_merges(
                ckpt["merges"], word_types, word_freqs, token_index, pair_counts,
            )
            start_step = ckpt["step"]
            elapsed_prior = ckpt["elapsed_seconds"]
            pair_counts = self.count_all_pairs(word_types, word_freqs)
            print(f"  rebuilt pair counts: {len(pair_counts):,} unique pairs")

        total_vocab_needed = self.target_vocab_size - len(SPECIAL_TOKENS)
        num_merges = max(0, total_vocab_needed - base_vocab)
        remaining = num_merges - start_step
        print(f"\n  merge budget: {num_merges:,} "
              f"(starting at {start_step}, {remaining:,} remaining, min_freq={self.min_freq})")

        print(f"\n[4/5] Merge loop...")
        heap = self._build_heap(pair_counts)
        t0 = time.time()
        pbar = tqdm(range(start_step + 1, num_merges + 1),
                    desc="  merging", unit=" merge")

        for step in pbar:
            best_pair, freq = self._pop_best(heap, pair_counts)
            if best_pair is None or freq < self.min_freq:
                pbar.write(f"  stopping at step {step}: "
                           f"{'no pairs' if best_pair is None else f'freq={freq} < {self.min_freq}'}")
                break

            a_id, b_id = best_pair
            merged_id = self.symbols.add_merged(a_id, b_id)
            self.merges.append(best_pair)

            n_applied = self.merge_and_update(
                word_types, word_freqs, best_pair, pair_counts,
                token_index, merged_id, heap,
            )

            if step <= 10 or step % 1000 == 0:
                a_s = self.symbols.to_str(a_id)
                b_s = self.symbols.to_str(b_id)
                m_s = self.symbols.to_str(merged_id)
                elapsed = time.time() - t0 + elapsed_prior
                pbar.write(f"  [{step:>7}/{num_merges}] "
                           f"'{a_s}' + '{b_s}' -> '{m_s}' "
                           f"(freq={freq:,}, applied={n_applied:,}) [{elapsed:.1f}s]")

            if self.checkpoint_every > 0 and step % self.checkpoint_every == 0:
                elapsed = time.time() - t0 + elapsed_prior
                path, sz = self.save_checkpoint(step, output_dir, elapsed)
                pbar.write(f"  >> checkpoint: {path} ({sz:.2f} MB)")

            pbar.set_postfix(freq=freq, vocab=len(self.symbols))

        pbar.close()
        merge_elapsed = time.time() - t0
        total_elapsed = merge_elapsed + elapsed_prior
        print(f"  done: {len(self.merges)} merges in {merge_elapsed:.1f}s "
              f"(total {total_elapsed:.1f}s)")

        print("\n[5/5] Building vocabulary and exporting...")
        self._save_output(word_types, word_freqs, boundary_tokens, output_dir)

        wall = time.time() - t_start
        print(f"\nTotal wall time: {wall:.1f}s ({wall/60:.1f} min)")

    def _save_output(self, word_types, word_freqs, boundary_tokens, output_dir):
        final_freq: Counter[int] = Counter()
        for wt_idx, wt in enumerate(word_types):
            f = word_freqs[wt_idx]
            for tid in wt:
                if tid >= 0:
                    final_freq[tid] += f

        vocab: dict[str, int] = {}
        for i, st in enumerate(SPECIAL_TOKENS):
            vocab[st] = i
        next_id = len(SPECIAL_TOKENS)

        for tid, _ in final_freq.most_common():
            tok_str = self.symbols.to_str(tid)
            if tok_str not in vocab:
                vocab[tok_str] = next_id
                next_id += 1

        for sid in range(len(self.symbols)):
            s = self.symbols.to_str(sid)
            if s not in vocab:
                vocab[s] = next_id
                next_id += 1

        print(f"  vocab size: {len(vocab):,}")
        print(f"  merge rules: {len(self.merges):,}")

        merge_strs = [
            [self.symbols.to_str(a), self.symbols.to_str(b)]
            for a, b in self.merges
        ]

        output = {
            "version": "wwho_sgpe",
            "script_mode": self.script_mode,
            "vocab_size": len(vocab),
            "special_tokens": SPECIAL_TOKENS,
            "num_merges": len(self.merges),
            "prune_freq": self.prune_freq,
            "leading_space": True,
            "merges": merge_strs,
            "vocab": vocab,
        }

        path = os.path.join(output_dir, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  saved: {path} ({size_mb:.2f} MB)")

        self.save_checkpoint(len(self.merges), output_dir, 0)

        hf_path = os.path.join(output_dir, "tokenizer.json")
        export_hf_tokenizer(vocab, merge_strs, SPECIAL_TOKENS, hf_path,
                               script_mode=self.script_mode)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — WWHO")
        print(f"  Script mode: {self.script_mode}")
        print(f"  Vocab size:  {len(vocab):,}")
        print(f"  Merge rules: {len(self.merges):,}")
        print(f"  Word types:  {len(word_types):,}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="WWHO (SGPE) GPE Trainer")
    parser.add_argument("--train_file", type=str, default="dataset/mixed_train.jsonl")
    parser.add_argument("--vocab_size", type=int, default=128_000,
                        help="Target SGPE vocab size (default 128K)")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--prune_freq", type=int, default=50,
                        help="Drop syllables below this corpus frequency to [UNK]")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=10_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--script_mode", type=str, default="mixed",
                        choices=["sinhala", "devanagari", "mixed"],
                        help="Which Indic script(s) to merge in BPE "
                             "(English/code always stays as boundary tokens)")
    args = parser.parse_args()
    _setup_logging(args.output_dir)
    _log(f"Starting WWHO (SGPE) trainer: train_file={args.train_file} "
         f"vocab_size={args.vocab_size} script_mode={args.script_mode} "
         f"prune_freq={args.prune_freq} min_freq={args.min_freq}")

    trainer = GPETrainer(
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        prune_freq=args.prune_freq,
        script_mode=args.script_mode,
    )
    trainer.train(args.train_file, args.output_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()
