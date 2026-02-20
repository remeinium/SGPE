"""
==========================================
SGPE Layer 1 — LinguisTrie Pre-tokenizer
==========================================
"""



HAL = '\u0DCA'   # ්  virama / al-lakuna
ZWJ = '\u200D'   # zero-width joiner

# --- Independent vowels (svara) ---
VOWELS: set[str] = {
    '\u0D85',  # අ
    '\u0D86',  # ආ
    '\u0D87',  # ඇ
    '\u0D88',  # ඈ
    '\u0D89',  # ඉ
    '\u0D8A',  # ඊ
    '\u0D8B',  # උ
    '\u0D8C',  # ඌ
    '\u0D8D',  # ඍ
    '\u0D8E',  # ඎ
    '\u0D8F',  # ඏ
    '\u0D90',  # ඐ
    '\u0D91',  # එ
    '\u0D92',  # ඒ
    '\u0D93',  # ඓ
    '\u0D94',  # ඔ
    '\u0D95',  # ඕ
    '\u0D96',  # ඖ
}

# --- Consonants (vyanjana) ---
CONSONANTS: set[str] = {chr(c) for c in range(0x0D9A, 0x0DC7)}

# --- Dependent vowel signs (pili) ---
VOWEL_SIGNS: set[str] = {
    '\u0DCF',  # ා
    '\u0DD0',  # ැ
    '\u0DD1',  # ෑ
    '\u0DD2',  # ි
    '\u0DD3',  # ී
    '\u0DD4',  # ු
    '\u0DD5',  # ෕ (rare/archaic)
    '\u0DD6',  # ූ
    '\u0DD7',  # ෗ (rare/archaic)
    '\u0DD8',  # ෘ
    '\u0DD9',  # ෙ
    '\u0DDA',  # ේ
    '\u0DDB',  # ෛ
    '\u0DDC',  # ො
    '\u0DDD',  # ෝ
    '\u0DDE',  # ෞ
    '\u0DDF',  # ෟ
    '\u0DF2',  # ෲ
    '\u0DF3',  # ෳ
}

# --- Post-consonant modifiers (anusvara, visarga) ---
POST_MODIFIERS: set[str] = {
    '\u0D82',  # ං  anusvara
    '\u0D83',  # ඃ  visarga
}



def _is_consonant(ch: str) -> bool:
    return ch in CONSONANTS

def _is_vowel(ch: str) -> bool:
    return ch in VOWELS

def _is_vowel_sign(ch: str) -> bool:
    return ch in VOWEL_SIGNS

def _is_post_modifier(ch: str) -> bool:
    return ch in POST_MODIFIERS

def _is_hal(ch: str) -> bool:
    return ch == HAL

def _is_zwj(ch: str) -> bool:
    return ch == ZWJ

def _is_sinhala(ch: str) -> bool:
    """Any character in the Sinhala Unicode block or ZWJ."""
    cp = ord(ch)
    return (0x0D80 <= cp <= 0x0DFF) or cp == 0x200D



class LinguisTrie:
    def tokenize(self, text: str, leading_space: bool = False) -> list[str]:
        """
        Tokenize Sinhala text into atomic syllable tokens.
        Example: "මම යනවා" → [" මම", " ය", "න", "වා"]
        """
        tokens: list[str] = []
        n = len(text)
        pos = 0
        pending_space = ""

        while pos < n:
            ch = text[pos]

            # ─── Whitespace handling (leading-space mode) ─────────
            if leading_space and ch in (' ', '\t', '\n', '\r'):
                ws_buffer = ""
                while pos < n and text[pos] in (' ', '\t', '\n', '\r'):
                    ws_buffer += text[pos]
                    pos += 1
                
                if ws_buffer.endswith(' '):
                    for ws_char in ws_buffer[:-1]:
                         tokens.append(ws_char)
                    pending_space = " "
                else:
                    for ws_char in ws_buffer:
                        tokens.append(ws_char)
                    pending_space = ""
                continue

            # ─── Consonant-initiated syllable ─────────────────────
            if _is_consonant(ch):
                start = pos
                pos += 1

                # Absorb consonant cluster: (HAL [ZWJ] Consonant)*
                #   Handles: C්C (implicit), C්‍C (ZWJ), and stacks
                while pos < n and _is_hal(text[pos]):
                    if pos + 1 < n and _is_zwj(text[pos + 1]):
                        # HAL + ZWJ: must be followed by consonant
                        if pos + 2 < n and _is_consonant(text[pos + 2]):
                            pos += 3  
                            continue
                        else:
                            # Stray HAL+ZWJ at end — absorb HAL+ZWJ
                            pos += 2
                            break

                    elif pos + 1 < n and _is_consonant(text[pos + 1]):
                        # HAL + C (implicit conjunct, no ZWJ)
                        pos += 2  
                        continue

                    else:
                        break

                # ── Post-cluster modifiers ──

                if pos < n and _is_vowel_sign(text[pos]):
                    pos += 1   # pili
                elif pos < n and _is_hal(text[pos]):
                    pos += 1   # virama

                if pos < n and _is_post_modifier(text[pos]):
                    pos += 1   # anusvara/visarga

                tokens.append(pending_space + text[start:pos])
                pending_space = ""
                continue

            # ─── Independent vowel ────────────────────────────────
            if _is_vowel(ch):
                start = pos
                pos += 1

                # Vowel + post-modifier (e.g. අං)
                if pos < n and _is_post_modifier(text[pos]):
                    pos += 1

                tokens.append(pending_space + text[start:pos])
                pending_space = ""
                continue

            # ─── Orphan post-modifier ──
            if _is_post_modifier(ch) or _is_hal(ch) or _is_vowel_sign(ch):
                tokens.append(pending_space + ch)
                pending_space = ""
                pos += 1
                continue

            # ─── Non-Sinhala passthrough (punctuation, digits, etc.) ──
            if pending_space:
                tokens.append(pending_space + ch)
                pending_space = ""
            else:
                tokens.append(ch)
            pos += 1

        if pending_space:
            tokens.append(pending_space)

        return tokens



def build_linguistrie() -> LinguisTrie:
    """Build and return the LinguisTrie."""
    return LinguisTrie()



if __name__ == '__main__':
    trie = build_linguistrie()

    test_sentences = [
        # Core tests from the plan
        "ශ්‍රී ලංකා ද්වීපයේ ස්වෛරීභාවය සහ ත්‍රිවිධ හමුදාව.",
        "භාෂාවේ ප්‍රෞඪත්වය විදහාපායි",
        "ආචාර්යවරයාගේ වෛද්‍ය විද්‍යා පර්යේෂණය සාර්ථකයි.",
        "චන්ද්‍රයාගේ ආලෝකය පෘථිවියට ක්ෂණිකව ලැබේ.",
        "මම ක්‍ෂණිකව ගඟට පැන්නා",
        "සඤ්ඤක ක්ෂමතාවය ක්‍රමය සහ ඥානය",
        "ද්වී ත්වේ ලං කඃ",
        "න්ද්‍රී ක්ෂි ඤ්ඤ",
        "2026 වසරේ AI තාක්ෂණය 60% දියුණුයි!",
    ]

    for text in test_sentences:
        tokens = trie.tokenize(text)
        print(f"Input:  {text}")
        print(f"Tokens: {tokens}")
        print(f"Count:  {len(tokens)}")
        print("-" * 60)
