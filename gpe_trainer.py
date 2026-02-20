"""
SGPE Layer 2 — GPE Trainer

Usage:
    python gpe_trainer.py --train_file data/train.jsonl
    python gpe_trainer.py --train_file data/train.jsonl --vocab_size 100000
    python gpe_trainer.py --train_file data/train.jsonl --resume output/checkpoint_15000.json
"""

import argparse
import heapq
import json
import os
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from linguis_trie import LinguisTrie
from export import export_hf_tokenizer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

_SINHALA_RE = re.compile(r'[\u0D80-\u0DFF\u200D]')

_worker_tokenizer: LinguisTrie | None = None


def _init_worker():
    global _worker_tokenizer
    _worker_tokenizer = LinguisTrie()


def _pretokenize_line(text: str) -> list[str]:
    return _worker_tokenizer.tokenize(text, leading_space=True)


def _is_boundary_token(token: str) -> bool:
    """True if the token is non-Sinhala (space, punctuation, digit, etc)."""
    if not token:
        return True
    return not bool(_SINHALA_RE.search(token))


def segment_into_words(syllables: list[str]) -> list[list[str]]:
    """
    Word boundary logic: a leading-space Sinhala token starts a new word.
    Example: ['ම', 'ම', ' ය', 'න', 'වා'] -> [['ම', 'ම'], [' ය', 'න', 'වා']]
    """
    words: list[list[str]] = []
    current: list[str] = []

    for tok in syllables:
        if _is_boundary_token(tok):
            # pure non-Sinhala token
            if current:
                words.append(current)
                current = []
            words.append([tok])
        else:
            # Sinhala token
            if tok[0] in (' ', '\t', '\n', '\r') and current:
                words.append(current)
                current = []
            current.append(tok)

    if current:
        words.append(current)
    return words


class SymbolTable:
    """Bidirectional string <-> int mapping."""

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


class GPETrainer:

    def __init__(
        self,
        vocab_size: int = 100_000,
        min_freq: int = 2,
        num_workers: int | None = None,
        checkpoint_every: int = 5000,
        prune_freq: int = 100,
    ):
        self.target_vocab_size = vocab_size
        self.min_freq = min_freq
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.checkpoint_every = checkpoint_every
        self.prune_freq = prune_freq
        self.merges: list[tuple[int, int]] = []
        self.symbols = SymbolTable()

    def stream_and_count(self, train_file: str) -> tuple[Counter, set[str]]:
        """Stream sentences through Layer 1"""
        word_counter: Counter[tuple[str, ...]] = Counter()
        boundary_tokens: set[str] = set()
        total_lines = 0

        print("  counting lines...", end=" ", flush=True)
        with open(train_file, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
        print(f"{num_lines:,}")

        BATCH = 8192
        batch: list[str] = []

        with Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            with open(train_file, "r", encoding="utf-8") as f:
                pbar = tqdm(f, total=num_lines, unit=" sent", desc="  pre-tokenizing")
                for raw_line in pbar:
                    obj = json.loads(raw_line)
                    text = obj["text"].strip()
                    if not text:
                        continue
                    batch.append(text)
                    total_lines += 1

                    if len(batch) >= BATCH:
                        self._process_batch(pool, batch, word_counter,
                                            boundary_tokens)
                        batch = []

                if batch:
                    self._process_batch(pool, batch, word_counter,
                                        boundary_tokens)
                    batch = []

                pbar.close()

        n_types = len(word_counter)
        n_instances = sum(word_counter.values())
        print(f"  {total_lines:,} sentences -> {n_types:,} unique word types "
              f"({n_instances:,} total instances)")
        print(f"  {len(boundary_tokens):,} unique boundary tokens")
        return word_counter, boundary_tokens

    def _process_batch(
        self,
        pool: Pool,
        batch: list[str],
        word_counter: Counter,
        boundary_tokens: set[str],
    ):
        """Pre-tokenize a batch and fold results into the word counter."""
        syllable_streams = pool.map(_pretokenize_line, batch, chunksize=256)

        for stream in syllable_streams:
            words = segment_into_words(stream)
            for w in words:
                if _is_boundary_token(w[0]):
                    boundary_tokens.add(w[0])
                else:
                    word_counter[tuple(w)] += 1

    @staticmethod
    def compute_syllable_freqs(word_counter: Counter) -> Counter:
        """Syllable frequencies, weighted by word occurrence count."""
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
        """
        Convert word types to integer ID lists.
        Rare syllables (below prune_freq) become UNK sentinel.
        """
        UNK_SENTINEL = -1

        pruned_set: set[str] = set()
        if syl_freq is not None and self.prune_freq > 0:
            for syl, freq in syl_freq.items():
                if freq < self.prune_freq:
                    pruned_set.add(syl)

        for bt in sorted(boundary_tokens):
            self.symbols.get_or_add(bt)

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
        """token_id -> set of word_type indices that contain it."""
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
        """Count pair frequencies, weighted by word occurrence."""
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
    def _build_heap(pair_counts: dict) -> list[tuple[int, tuple[int, int]]]:
        """Max-heap via negation (heapq is a min-heap)."""
        heap = [(-freq, pair) for pair, freq in pair_counts.items() if freq > 0]
        heapq.heapify(heap)
        return heap

    @staticmethod
    def _heap_push(heap, pair, freq):
        if freq > 0:
            heapq.heappush(heap, (-freq, pair))

    def _pop_best(self, heap, pair_counts):
        """Pop the best pair, skipping stale heap entries."""
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

        candidates_a = token_index.get(a, set())
        candidates_b = token_index.get(b, set())
        candidates = list(candidates_a & candidates_b)

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
                    # decrement left neighbor pair
                    if new_wt and new_wt[-1] >= 0:
                        left_pair = (new_wt[-1], a)
                        pair_counts[left_pair] = pair_counts.get(left_pair, 0) - freq
                        dirty_pairs[left_pair] = pair_counts[left_pair]

                    # decrement right neighbor pair
                    if i + 2 < len(wt) and wt[i + 2] >= 0:
                        right_pair = (b, wt[i + 2])
                        pair_counts[right_pair] = pair_counts.get(right_pair, 0) - freq
                        dirty_pairs[right_pair] = pair_counts[right_pair]

                    new_wt.append(merged_id)
                    total_applied += freq
                    changed = True

                    # increment new left neighbor pair
                    if len(new_wt) >= 2 and new_wt[-2] >= 0:
                        lp = (new_wt[-2], merged_id)
                        pair_counts[lp] = pair_counts.get(lp, 0) + freq
                        dirty_pairs[lp] = pair_counts[lp]

                    # increment new right neighbor pair
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

    def replay_merges(
        self,
        merge_strs: list[list[str]],
        word_types: list[list[int]],
        word_freqs: list[int],
        token_index: dict[int, set[int]],
        pair_counts: dict[tuple[int, int], int],
    ):
        """Replay saved merges. Fast because heap isn't maintained."""
        print(f"  replaying {len(merge_strs)} merges...", flush=True)
        t0 = time.time()

        dummy_heap: list = []
        for i, (a_str, b_str) in enumerate(tqdm(merge_strs, desc="  replaying",
                                                  unit=" merge")):
            a_id = self.symbols.to_id(a_str)
            b_id = self.symbols.to_id(b_str)
            if a_id is None or b_id is None:
                continue
            merged_id = self.symbols.add_merged(a_id, b_id)
            self.merges.append((a_id, b_id))

            pair = (a_id, b_id)
            self.merge_and_update(
                word_types, word_freqs, pair, pair_counts,
                token_index, merged_id, dummy_heap,
            )

        elapsed = time.time() - t0
        print(f"  replayed {len(self.merges)} merges in {elapsed:.1f}s")

    def train(self, train_file: str, output_dir: str = "output",
              resume_path: str | None = None):
        os.makedirs(output_dir, exist_ok=True)

        # stream and count word types
        print("[1/5] Streaming pre-tokenization and word counting (leading_space=True)...")
        t_start = time.time()
        word_counter, boundary_tokens = self.stream_and_count(train_file)

        # build ID corpus
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

        # handle resume
        start_step = 0
        elapsed_prior = 0.0
        if resume_path:
            print(f"\n  Resuming from {resume_path}...")
            ckpt = self.load_checkpoint(resume_path)
            self.replay_merges(
                ckpt["merges"], word_types, word_freqs,
                token_index, pair_counts,
            )
            start_step = ckpt["step"]
            elapsed_prior = ckpt["elapsed_seconds"]
            pair_counts = self.count_all_pairs(word_types, word_freqs)
            print(f"  rebuilt pair counts: {len(pair_counts):,} unique pairs")

        # merge budget
        total_vocab_needed = self.target_vocab_size - len(SPECIAL_TOKENS)
        if base_vocab >= total_vocab_needed:
            num_merges = 0
            print("  base vocab already >= target. nothing to merge.")
        else:
            num_merges = total_vocab_needed - base_vocab
        remaining = num_merges - start_step
        print(f"  merge budget: {num_merges:,} (starting at {start_step}, "
              f"{remaining:,} remaining, min_freq={self.min_freq})")

        # merge loop
        print(f"\n[4/5] Merge loop...")
        heap = self._build_heap(pair_counts)

        t0 = time.time()
        pbar = tqdm(range(start_step + 1, num_merges + 1),
                    desc="  merging", unit=" merge")

        for step in pbar:
            best_pair, freq = self._pop_best(heap, pair_counts)
            if best_pair is None or freq < self.min_freq:
                if best_pair is None:
                    pbar.write(f"  no more pairs at step {step}. stopping.")
                else:
                    pbar.write(f"  freq={freq} < min_freq={self.min_freq} "
                               f"at step {step}. stopping.")
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
                pbar.write(f"  [{step:>6}/{num_merges}] "
                           f"'{a_s}' + '{b_s}' -> '{m_s}' "
                           f"(freq={freq:,}, applied={n_applied:,}) "
                           f"[{elapsed:.1f}s]")

            if self.checkpoint_every > 0 and step % self.checkpoint_every == 0:
                elapsed = time.time() - t0 + elapsed_prior
                path, sz = self.save_checkpoint(step, output_dir, elapsed)
                pbar.write(f"  >> checkpoint saved: {path} ({sz:.2f} MB)")

            pbar.set_postfix(freq=freq, vocab=len(self.symbols))

        pbar.close()
        merge_elapsed = time.time() - t0
        total_elapsed = merge_elapsed + elapsed_prior
        print(f"  merge loop done: {len(self.merges)} merges in {merge_elapsed:.1f}s "
              f"(total {total_elapsed:.1f}s)")

        # build and save vocab
        print("\n[5/5] Building vocabulary...")
        self._save_output(word_types, word_freqs, boundary_tokens, output_dir)

        wall = time.time() - t_start
        print(f"\ntotal wall time: {wall:.1f}s ({wall/60:.1f} min)")

    def _save_output(self, word_types, word_freqs, boundary_tokens, output_dir):
        """Build the final vocab and save."""
        # count token frequencies in the final BPE corpus
        final_freq: Counter[int] = Counter()
        for wt_idx, wt in enumerate(word_types):
            f = word_freqs[wt_idx]
            for tid in wt:
                if tid >= 0:
                    final_freq[tid] += f

        for bt in boundary_tokens:
            bt_id = self.symbols.to_id(bt)
            if bt_id is not None and bt_id not in final_freq:
                final_freq[bt_id] = 1

        # layout: [special tokens] [BPE tokens by frequency]
        vocab: dict[str, int] = {}
        for i, st in enumerate(SPECIAL_TOKENS):
            vocab[st] = i

        next_id = len(SPECIAL_TOKENS)

        for tid, _ in final_freq.most_common():
            tok_str = self.symbols.to_str(tid)
            if tok_str not in vocab:
                vocab[tok_str] = next_id
                next_id += 1

        # any leftover symbols that didn't appear in the final corpus
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
            "version": "sgpe_v1.0.0",
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
        export_hf_tokenizer(vocab, merge_strs, SPECIAL_TOKENS, hf_path)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"  Vocab size:  {len(vocab):,}")
        print(f"  Merge rules: {len(self.merges):,}")
        print(f"  Word types:  {len(word_types):,}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SGPE v1.0.0 GPE Trainer")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--vocab_size", type=int, default=100_000)
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Stop when best pair freq drops below this")
    parser.add_argument("--prune_freq", type=int, default=100,
                        help="Drop syllables with corpus freq below this to [UNK]")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint JSON to resume from")
    args = parser.parse_args()

    trainer = GPETrainer(
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        prune_freq=args.prune_freq,
    )
    trainer.train(args.train_file, args.output_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()
