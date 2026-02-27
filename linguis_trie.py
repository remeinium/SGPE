"""
==========================================
Table-Driven DFA Tokenizer
==========================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Schema loading and validation
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when a language schema JSON is malformed or incomplete."""

@dataclass
class LanguageSchema:
    language: str
    grammar_notation: str
    char_classes: dict[str, set[int]]         # class-label → set of codepoints
    transitions: dict[str, dict[str, Optional[str]]]  # state → (class → next_state | None)
    start_state: str
    accept_states: set[str]
    emit_states: set[str]

    def get_regex(self) -> str:
        parts = []
        for cps in self.char_classes.values():
            for cp in cps:
                parts.append(chr(cp))
        
        if not parts:
            return ""
            
        safe_parts = []
        for p in parts:
            if p in ('-', ']', '\\', '^'):
                safe_parts.append('\\' + p)
            else:
                safe_parts.append(p)
                
        char_set = "".join(set(safe_parts))
        return f"[{char_set}]+"


class SchemaLoader:
    def load(self, path: str) -> LanguageSchema:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        language = raw.get("language", "unknown")
        grammar  = raw.get("grammar_notation", "")

        if "char_classes" not in raw:
            raise SchemaError(f"[{path}] Missing 'char_classes' key.")
        if "dfa" not in raw:
            raise SchemaError(f"[{path}] Missing 'dfa' key.")

        char_classes: dict[str, set[int]] = {}
        for label, definition in raw["char_classes"].items():
            if label.startswith("_"):
                continue
            cps: set[int] = set()
            for rng in definition.get("ranges", []):
                lo, hi = int(rng[0], 16), int(rng[1], 16)
                cps.update(range(lo, hi + 1))
            for cp_hex in definition.get("codepoints", []):
                cps.add(int(cp_hex, 16))
            char_classes[label] = cps

        dfa_raw = raw["dfa"]
        start_state   = dfa_raw.get("start", "START")
        accept_states = set(dfa_raw.get("accept_states", []))
        emit_states   = set(dfa_raw.get("emit_states", []))
        transitions   = dfa_raw.get("transitions", {})

        return LanguageSchema(
            language=language,
            grammar_notation=grammar,
            char_classes=char_classes,
            transitions=transitions,
            start_state=start_state,
            accept_states=accept_states,
            emit_states=emit_states,
        )


# ---------------------------------------------------------------------------
# Codepoint classifier
# ---------------------------------------------------------------------------

class CharClassifier:
    def __init__(self, schema: LanguageSchema):
        self._table: dict[int, str] = {}
        for label, cps in schema.char_classes.items():
            for cp in cps:
                if cp in self._table:
                    continue
                self._table[cp] = label

    def classify(self, ch: str) -> str:
        return self._table.get(ord(ch), "O")


# ---------------------------------------------------------------------------
# DFA Tokenizer
# ---------------------------------------------------------------------------

class LinguisTrie:
    def __init__(self, schema: LanguageSchema):
        self._schema      = schema
        self._classifier  = CharClassifier(schema)
        self._transitions = schema.transitions
        self._start       = schema.start_state
        self._accept      = schema.accept_states
        self._emit        = schema.emit_states

    def tokenize(self, text: str, leading_space: bool = False) -> list[str]:
        tokens: list[str] = []
        n     = len(text)
        pos   = 0
        
        pending_space = " " if leading_space and text and text[0] not in (" ", "\t", "\n", "\r") else ""

        while pos < n:
            ch = text[pos]

            # ─── Whitespace handling (leading-space mode) ────────────
            if leading_space and ch in (" ", "\t", "\n", "\r"):
                ws_buffer = ""
                while pos < n and text[pos] in (" ", "\t", "\n", "\r"):
                    ws_buffer += text[pos]
                    pos += 1

                if ws_buffer.endswith(" "):
                    for ws_char in ws_buffer[:-1]:
                        tokens.append(ws_char)
                    pending_space = " "
                else:
                    for ws_char in ws_buffer:
                        tokens.append(ws_char)
                    pending_space = ""
                continue

            # ─── DFA syllable recognition ────────────────────
            cls       = self._classifier.classify(ch)
            init_next = self._transitions.get(self._start, {}).get(cls)

            if init_next is None:
                if pending_space:
                    tokens.append(pending_space + ch)
                    pending_space = ""
                else:
                    tokens.append(ch)
                pos += 1
                continue

            if init_next in self._emit:
                tokens.append(pending_space + ch)
                pending_space = ""
                pos += 1
                continue

            span_start = pos
            state      = init_next
            pos       += 1
            last_accept_pos = pos if state in self._accept else -1

            while pos < n:
                ch2  = text[pos]
                cls2 = self._classifier.classify(ch2)
                next_state = self._transitions.get(state, {}).get(cls2)

                if next_state is None:
                    break

                state = next_state
                pos  += 1

                if state in self._accept:
                    last_accept_pos = pos
                elif state in self._emit:
                    last_accept_pos = pos
                    break

            if last_accept_pos > span_start:
                emit_end = last_accept_pos
            else:
                emit_end = pos

            tokens.append(pending_space + text[span_start:emit_end])
            pending_space = ""
            pos = emit_end

        if pending_space:
            tokens.append(pending_space)

        return tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return self._schema.language

    @property
    def regex(self) -> str:
        return self._schema.get_regex()

    @property
    def grammar(self) -> str:
        return self._schema.grammar_notation


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")

_schema_loader = SchemaLoader()
_dfa_cache: dict[str, LinguisTrie] = {}


def build_linguis_trie(schema_path: str) -> LinguisTrie:
    if schema_path not in _dfa_cache:
        schema = _schema_loader.load(schema_path)
        _dfa_cache[schema_path] = LinguisTrie(schema)
    return _dfa_cache[schema_path]


def build_sinhala_linguis_trie() -> LinguisTrie:
    return build_linguis_trie(os.path.join(_SCHEMA_DIR, "sinhala.json"))


def build_devanagari_linguis_trie() -> LinguisTrie:
    return build_linguis_trie(os.path.join(_SCHEMA_DIR, "devanagari.json"))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("DFA Tokenizer — self-test")
    print("=" * 65)

    # --- Sinhala ---
    sinhala_dfa = build_sinhala_linguis_trie()
    print(f"\n[Sinhala DFA]  grammar: {sinhala_dfa.grammar}\n")

    sinhala_tests = [
        "ශ්‍රී ලංකා ද්වීපයේ ස්වෛරීභාවය සහ ත්‍රිවිධ හමුදාව.",
        "භාෂාවේ ප්‍රෞඪත්වය විදහාපායි",
        "ආචාර්යවරයාගේ වෛද්‍ය විද්‍යා පර්යේෂණය සාර්ථකයි.",
        "චන්ද්‍රයාගේ ආලෝකය පෘථිවියට ක්ෂණිකව ලැබේ.",
        "මම ක්‍ෂණිකව ගඟට පැන්නා",
        "සඤ්ඤක ක්ෂමතාවය ක්‍රමය සහ ඥානය",
        "ද්වී ත්වේ ලං කඃ",
        "2026 වසරේ AI තාක්ෂණය 60% දියුණුයි!",
    ]

    for text in sinhala_tests:
        toks = sinhala_dfa.tokenize(text, leading_space=True)
        print(f"  Input : {text}")
        print(f"  Syllables: {toks}")
        print(f"  Count : {len(toks)}")
        print("-" * 65)

    # --- Devanagari ---
    deva_dfa = build_devanagari_linguis_trie()
    print(f"\n[Devanagari DFA]  grammar: {deva_dfa.grammar}\n")

    deva_tests = [
        "नमस्ते",
        "भारत",
        "हिन्दी",
        "संस्कृत",
        "क़िला",
        "ज़िंदगी",
        "प्रेम",
        "द्वारा",
        "श्रीमान्",
        "हिन्दुस्तान",
        "नमस्कार दुनिया",
        "मैं ठीक हूँ",
        "विद्यालय में पढ़ाई होती है।",
    ]

    for text in deva_tests:
        toks = deva_dfa.tokenize(text, leading_space=True)
        print(f"  Input : {text}")
        print(f"  Syllables: {toks}")
        print(f"  Count : {len(toks)}")
        print("-" * 65)

    print("\nAll self-tests complete.")
    sys.exit(0)
