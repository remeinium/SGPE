"""
==========================================
Code-Switching Router
==========================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

@dataclass
class TextSegment:
    text: str
    language: str                    
    has_leading_space: bool = False   


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

class CodeSwitchSegmenter:
    def __init__(self, language_blocks: dict[str, list[tuple[int, int]]] = None):
        """
        language_blocks: maps language name (e.g. 'sinhala') to a list of (start_cp, end_cp) inclusive
        """
        self._ranges: list[tuple[int, int, str]] = []
        if language_blocks:
            for lang, blocks in language_blocks.items():
                for start, end in blocks:
                    self._ranges.append((start, end, lang))

    def _get_char_language(self, ch: str) -> Optional[str]:
        if ch in ('\u200C', '\u200D'):
            return "__joiner__"
        cp = ord(ch)
        for start, end, lang in self._ranges:
            if start <= cp <= end:
                return lang
        return None

    def segment(self, text: str) -> list[TextSegment]:
        if not text:
            return []

        segments: list[TextSegment] = []
        n = len(text)
        pos = 0

        while pos < n:
            ch = text[pos]
            ch_lang = self._get_char_language(ch)

            is_indic_start = (ch_lang is not None)

            if not is_indic_start:
                # ─── 1. Accumulate Latin block ───
                start = pos
                while pos < n:
                    ch2 = text[pos]
                    lang2 = self._get_char_language(ch2)
                    if lang2 is not None and lang2 != "__joiner__":
                        break  
                    pos += 1
                
                latino_only = text[start:pos]
                
                has_ls = False
                if pos < n and latino_only.endswith(" "):
                    latino_only = latino_only[:-1]
                    has_ls = True
                
                if latino_only:
                    segments.append(TextSegment(text=latino_only, language="latin"))

                if has_ls and pos < n:
                    indic_start = pos
                    current_lang = self._get_char_language(text[pos])
                    if current_lang == "__joiner__" or current_lang is None:
                        current_lang = "__unknown__"
                    
                    while pos < n:
                        c = text[pos]
                        c_lang = self._get_char_language(c)
                        if c_lang == "__joiner__":
                            pos += 1
                        elif c_lang is not None:
                            if current_lang == "__unknown__":
                                current_lang = c_lang
                            elif c_lang != current_lang:
                                break
                            pos += 1
                        else:
                            break
                            
                    segments.append(TextSegment(
                        text=text[indic_start:pos],
                        language=current_lang,
                        has_leading_space=True
                    ))
            else:
                indic_start = pos
                current_lang = ch_lang
                
                while pos < n:
                    c = text[pos]
                    c_lang = self._get_char_language(c)
                    if c_lang == "__joiner__":
                        pos += 1
                    elif c_lang is not None:
                        if c_lang != current_lang:
                            break
                        pos += 1
                    else:
                        break
                        
                segments.append(TextSegment(
                    text=text[indic_start:pos],
                    language=current_lang,
                    has_leading_space=False
                ))

        return segments

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    language_blocks = {
        "sinhala": [(0x0d80, 0x0dff)],
        "devanagari": [(0x0900, 0x097f)]
    }
    seg = CodeSwitchSegmenter(language_blocks)
    
    for text in test_cases:
        blocks = seg.segment(text)
        print(f"\n  Input  : {text!r}")
        print(f"  Blocks : {[(b.text, b.language, b.has_leading_space) for b in blocks]}")
