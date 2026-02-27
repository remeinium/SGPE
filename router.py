"""
==========================================
Code-Switching Router
==========================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import tiktoken

from linguis_trie import build_sinhala_linguis_trie, build_devanagari_linguis_trie, LinguisTrie


# ---------------------------------------------------------------------------
# Script-block detection
# ---------------------------------------------------------------------------

class Script(Enum):
    LATIN  = auto()   # ASCII, Latin, digits, punctuation, code, emoji, etc.
    SINHALA     = auto()
    DEVANAGARI  = auto()

_sinhala_dfa    = build_sinhala_linguis_trie()
_devanagari_dfa = build_devanagari_linguis_trie()

_INDIC_PUNCT_CHARS = "\u0964\u0965"

def _get_char_script(ch: str) -> Optional[Script]:
    if '\u0D80' <= ch <= '\u0DFF':
        return Script.SINHALA
    if '\u0900' <= ch <= '\u097F':
        return Script.DEVANAGARI
    if ch in _INDIC_PUNCT_CHARS:
        return Script.SINHALA  # Dandas handled identically by both schemas
    return None

def _is_indic_joiner(ch: str) -> bool:
    # True if ZWJ or ZWNJ
    return ch in ('\u200C', '\u200D')


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class TextSegment:
    text: str
    script: Script
    has_leading_space: bool = False   # True if a boundary space was absorbed


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

class CodeSwitchSegmenter:
    def segment(self, text: str) -> list[TextSegment]:
        if not text:
            return []

        segments: list[TextSegment] = []
        n = len(text)
        pos = 0

        while pos < n:
            ch = text[pos]
            ch_script = _get_char_script(ch)

            is_indic_start = (ch_script is not None)

            if not is_indic_start:
                # ─── 1. Accumulate Latin block ───
                start = pos
                while pos < n:
                    ch2 = text[pos]
                    if _get_char_script(ch2) is not None:
                        break  # Found distinct Indic start
                    pos += 1
                
                latin_chunk = text[start:pos]
                
                has_ls = False
                if pos < n and latin_chunk.endswith(" "):
                    latin_chunk = latin_chunk[:-1]
                    has_ls = True
                
                if latin_chunk:
                    segments.append(TextSegment(text=latin_chunk, script=Script.LATIN))

                if has_ls and pos < n:
                    indic_start = pos
                    current_script = _get_char_script(text[pos]) or Script.SINHALA
                    
                    while pos < n:
                        c = text[pos]
                        c_script = _get_char_script(c)
                        if _is_indic_joiner(c):
                            pos += 1
                        elif c_script is not None:
                            if c_script != current_script and c not in _INDIC_PUNCT_CHARS:
                                break
                            pos += 1
                        else:
                            break
                            
                    segments.append(TextSegment(
                        text=text[indic_start:pos],
                        script=current_script,
                        has_leading_space=True
                    ))
            else:
                # ─── 2. Accumulate Indic block (no prior Latin with space) ───
                indic_start = pos
                current_script = ch_script
                
                while pos < n:
                    c = text[pos]
                    c_script = _get_char_script(c)
                    if _is_indic_joiner(c):
                        pos += 1
                    elif c_script is not None:
                        if c_script != current_script and c not in _INDIC_PUNCT_CHARS:
                            break
                        pos += 1
                    else:
                        break
                        
                segments.append(TextSegment(
                    text=text[indic_start:pos],
                    script=current_script,
                    has_leading_space=False
                ))

        return segments



# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class CodeSwitchRouter:
    def __init__(
        self,
        tiktoken_model: str = "o200k_base",
        sinhala_schema: Optional[str] = None,
        devanagari_schema: Optional[str] = None,
    ):
        # Indic DFAs
        self._sinhala_dfa:    LinguisTrie = build_sinhala_linguis_trie()
        self._devanagari_dfa: LinguisTrie = build_devanagari_linguis_trie()

        self._enc = tiktoken.get_encoding(tiktoken_model)

        self._segmenter = CodeSwitchSegmenter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize_to_strings(self, text: str) -> list[str]:
        result: list[str] = []
        for seg in self._segmenter.segment(text):
            result.extend(self._route_segment_strings(seg))
        return result

    def tokenize_to_ids(self, text: str) -> list[int]:
        raise NotImplementedError(
            "Use WWHOMetaEncoder.encode() for unified IDs. "
            "tokenize_to_ids() on the raw router is intentionally not implemented "
            "to prevent accidental ID space collision."
        )

        return self._enc.encode(text)

    def tiktoken_decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def tiktoken_vocab_size(self) -> int:
        return self._enc.n_vocab

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route_segment_strings(self, seg: TextSegment) -> list[str]:
        if seg.script == Script.LATIN:
            ids = self._enc.encode(seg.text)
            return [self._enc.decode([i]) for i in ids]

        # Indic — route to appropriate DFA
        dfa = (
            self._sinhala_dfa
            if seg.script == Script.SINHALA
            else self._devanagari_dfa
        )
        return dfa.tokenize(seg.text, leading_space=seg.has_leading_space)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = CodeSwitchRouter()

    test_cases = [
        # Pure Sinhala
        "ශ්‍රී ලංකාව",
        # Pure English
        "Hello, world!",
        # Mixed — English then Sinhala
        "The capital is කොළඹ.",
        # Mixed — Sinhala then English
        "ලංකාව is beautiful.",
        # Mixed — Devanagari
        "Hello नमस्ते world",
        # Code-switching with numbers
        "2026 AI සහ machine learning",
        # Boundary space edge-case
        "GPT-4 ශ්‍රී ලංකා",
        # Dense Sinhala
        "ආචාර්යවරයාගේ වෛද්‍ය විද්‍යා පර්යේෂණය සාර්ථකයි.",
        # Dense Devanagari
        "विद्यालय में पढ़ाई होती है।",
        # Multi-script sentence
        "AI (Artificial Intelligence) සහ देवनागरी text.",
    ]

    print("=" * 70)
    print("CodeSwitchRouter — self-test")
    print("=" * 70)

    seg = CodeSwitchSegmenter()
    for text in test_cases:
        tokens = router.tokenize_to_strings(text)
        blocks = seg.segment(text)
        print(f"\n  Input  : {text!r}")
        print(f"  Blocks : {[(b.text, b.script.name, b.has_leading_space) for b in blocks]}")
        print(f"  Tokens : {tokens}")
        print(f"  Count  : {len(tokens)}")
