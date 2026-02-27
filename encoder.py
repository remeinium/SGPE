"""
==========================================
WWHO Encoder  (Unified Meta-Vocabulary)
==========================================
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from linguis_trie import LinguisTrie, build_sinhala_linguis_trie
from gpe_trainer import segment_into_words, _is_boundary_token

class SGPEEncoder:

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab: dict[str, int]             = data["vocab"]
        self.merges: list[tuple[str, str]]     = [tuple(m) for m in data["merges"]]
        self.special_tokens: list[str]         = data["special_tokens"]
        self.tokenizer                         = build_sinhala_linguis_trie()
        self.unk_id                            = self.vocab.get("[UNK]", 1)
        self.leading_space: bool               = data.get("leading_space", False)

        self._merge_priority: dict[tuple[str, str], int] = {
            (a, b): rank for rank, (a, b) in enumerate(self.merges)
        }

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def _apply_merges_to_word(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 1:
            return tokens

        while True:
            best_rank = len(self.merges)
            best_idx  = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx  = i
            if best_idx == -1:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return tokens

    def tokenize(self, text: str) -> list[str]:
        syllables = self.layer1_tokenize(text)
        words     = segment_into_words(syllables)
        result: list[str] = []
        for word_tokens in words:
            if len(word_tokens) == 1 and _is_boundary_token(word_tokens[0]):
                result.append(word_tokens[0])
                continue
            cleaned = [t if t in self.vocab else "[UNK]" for t in word_tokens]
            result.extend(self._apply_merges_to_word(cleaned))
        return result

    def layer1_tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text, leading_space=self.leading_space)

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_token.get(i, "") for i in ids)


# ============================================================================
# MetaVocab — unified ID space
# ============================================================================

class MetaVocab:
    def __init__(self, sgpe_vocab: dict[str, int], tiktoken_size: int):
        self.tiktoken_size: int             = tiktoken_size
        self._sgpe_raw: dict[str, int]      = sgpe_vocab  
        self._sgpe_offset: dict[str, int]   = {
            tok: idx + tiktoken_size for tok, idx in sgpe_vocab.items()
        }
        self._sgpe_reverse: dict[int, str]  = {
            v: k for k, v in self._sgpe_offset.items()
        }

    @property
    def total_size(self) -> int:
        return self.tiktoken_size + len(self._sgpe_raw)

    def encode_sgpe_token(self, token: str, unk_id_raw: int) -> int:
        return self._sgpe_offset.get(token, unk_id_raw + self.tiktoken_size)

    def decode_id(self, uid: int) -> Optional[str]:
        if uid < self.tiktoken_size:
            return None
        return self._sgpe_reverse.get(uid)

    def is_tiktoken_id(self, uid: int) -> bool:
        return uid < self.tiktoken_size

    def sgpe_unk_id(self, raw_unk: int) -> int:
        return raw_unk + self.tiktoken_size


# ============================================================================
# WWHOMetaEncoder 
# ============================================================================

class WWHOMetaEncoder:

    def __init__(self, vocab_path: str, tiktoken_model: str = "o200k_base"):
        # Load SGPE vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sgpe_vocab: dict[str, int]         = data["vocab"]
        self._merges: list[tuple[str, str]] = [tuple(m) for m in data["merges"]]
        self._special_tokens: list[str]     = data["special_tokens"]
        self._leading_space: bool           = data.get("leading_space", False)
        self._raw_unk_id: int               = sgpe_vocab.get("[UNK]", 1)

        if " " not in sgpe_vocab:
            next_id = max(sgpe_vocab.values()) + 1
            sgpe_vocab[" "] = next_id

        try:
            from router import _INDIC_PUNCT_CHARS
            for ch in _INDIC_PUNCT_CHARS:
                if ch not in sgpe_vocab:
                    next_id = max(sgpe_vocab.values()) + 1
                    sgpe_vocab[ch] = next_id
        except ImportError:
            pass

        self._merge_priority: dict[tuple[str, str], int] = {
            (a, b): rank for rank, (a, b) in enumerate(self._merges)
        }

        # tiktoken
        try:
            import tiktoken as _tiktoken
            self._tik = _tiktoken.get_encoding(tiktoken_model)
        except Exception as e:
            raise RuntimeError(
                f"tiktoken ({tiktoken_model!r}) unavailable: {e}. "
            )

        # Unified vocab
        self._meta = MetaVocab(sgpe_vocab, self._tik.n_vocab)
        self._space_id: int = self._meta._sgpe_offset[" "]

        # Router
        from router import CodeSwitchSegmenter, Script
        self._segmenter = CodeSwitchSegmenter()
        self._Script    = Script

        # Indic LinguisTries
        from linguis_trie import build_sinhala_linguis_trie, build_devanagari_linguis_trie
        self._sinhala_dfa    = build_sinhala_linguis_trie()
        self._devanagari_dfa = build_devanagari_linguis_trie()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._meta.total_size

    @property
    def tiktoken_size(self) -> int:
        return self._meta.tiktoken_size

    @property
    def vocab(self) -> dict[str, int]:
        return self._meta._sgpe_raw

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for seg in self._segmenter.segment(text):
            if seg.script == self._Script.LATIN:
                ids.extend(self._tik.encode(seg.text))
            else:
                dfa = (
                    self._sinhala_dfa
                    if seg.script == self._Script.SINHALA
                    else self._devanagari_dfa
                )
                syllables = dfa.tokenize(seg.text, leading_space=seg.has_leading_space)
                words     = segment_into_words(syllables)
                for word_toks in words:
                    if len(word_toks) == 1 and _is_boundary_token(word_toks[0]):
                        ids.extend(self._tik.encode(word_toks[0]))
                        continue
                    merged = self._apply_merges(word_toks)
                    for tok in merged:
                        ids.append(self._meta.encode_sgpe_token(tok, self._raw_unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        tik_buf: list[int] = []
        result_parts: list[str] = []

        def _flush_tik():
            if tik_buf:
                result_parts.append(self._tik.decode(tik_buf))
                tik_buf.clear()

        for uid in ids:
            if self._meta.is_tiktoken_id(uid):
                tik_buf.append(uid)
            else:
                _flush_tik()
                tok = self._meta.decode_id(uid)
                result_parts.append(tok if tok is not None else "")

        _flush_tik()
        return "".join(result_parts)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in self._segmenter.segment(text):
            if seg.script == self._Script.LATIN:
                ids = self._tik.encode(seg.text)
                tokens.extend(self._tik.decode([i]) for i in ids)
            else:
                dfa = (
                    self._sinhala_dfa
                    if seg.script == self._Script.SINHALA
                    else self._devanagari_dfa
                )
                syllables = dfa.tokenize(seg.text, leading_space=seg.has_leading_space)
                words     = segment_into_words(syllables)
                for word_toks in words:
                    if len(word_toks) == 1 and _is_boundary_token(word_toks[0]):
                        ids = self._tik.encode(word_toks[0])
                        tokens.extend(self._tik.decode([i]) for i in ids)
                        continue
                    tokens.extend(self._apply_merges(word_toks))
        return tokens

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 1:
            return tokens
        sgpe = self._meta._sgpe_raw
        cleaned = [t if t in sgpe else "[UNK]" for t in tokens]
        while True:
            best_rank = len(self._merges)
            best_idx  = -1
            for i in range(len(cleaned) - 1):
                pair = (cleaned[i], cleaned[i + 1])
                rank = self._merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx  = i
            if best_idx == -1:
                break
            merged = cleaned[best_idx] + cleaned[best_idx + 1]
            cleaned = cleaned[:best_idx] + [merged] + cleaned[best_idx + 2:]
        return cleaned


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WWHO Encoder (Unified Meta-Vocabulary)")
    parser.add_argument("--vocab",  type=str, default="output/vocab.json",
                        help="Path to WWHO vocab.json")
    parser.add_argument("--text",   type=str, required=True,
                        help="Text to encode (supports mixed Latin + Indic)")
    parser.add_argument("--mode",   type=str, default="meta",
                        choices=["sgpe", "meta"],
                        help="'sgpe' = pure SGPE encoder; 'meta' = unified meta-encoder")
    parser.add_argument("--tiktoken_model", type=str, default="o200k_base")
    args = parser.parse_args()

    if args.mode == "sgpe":
        enc    = SGPEEncoder(args.vocab)
        tokens = enc.tokenize(args.text)
        ids    = enc.encode(args.text)
        print(f"[SGPEEncoder]")
        print(f"  tokens : {tokens}")
        print(f"  ids    : {ids}")
        print(f"  count  : {len(tokens)}")
    else:
        enc    = WWHOMetaEncoder(args.vocab, tiktoken_model=args.tiktoken_model)
        tokens = enc.tokenize(args.text)
        ids    = enc.encode(args.text)
        decoded = enc.decode(ids)
        print(f"[WWHOMetaEncoder]")
        print(f"  vocab_size    : {enc.vocab_size:,}  "
              f"(tiktoken={enc.tiktoken_size:,} + SGPE={enc.vocab_size - enc.tiktoken_size:,})")
        print(f"  tokens : {tokens}")
        print(f"  ids    : {ids}")
        print(f"  count  : {len(tokens)}")
        print(f"  decoded: {decoded!r}")
        print(f"  lossless: {decoded == args.text}")


if __name__ == "__main__":
    main()
