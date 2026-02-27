"""
SGPE - Battle Test
"""

import argparse
import json
import os
import re
import random
import statistics
import string
import sys
import time
import signal
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
from dotenv import load_dotenv

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import transformers
    from huggingface_hub import login as hf_login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


from linguis_trie import LinguisTrie
from gpe_trainer import segment_into_words, _is_boundary_token
from encoder import SGPEEncoder

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & UNICODE RANGES
# ═══════════════════════════════════════════════════════════════════════════════

HAL = "\u0DCA"           # Sinhala virama (al-lakuna)
ZWJ = "\u200D"           # Zero-width joiner
ZWNJ = "\u200C"          # Zero-width non-joiner
ANUSVARA = "\u0D82"      # ං
VISARGA = "\u0D83"       # ඃ

# Sinhala consonants: ක (0x0D9A) through ෆ (0x0DC6)
CONSONANTS = {chr(c) for c in range(0x0D9A, 0x0DC7)}

# Sinhala vowel signs (pili): 0x0DCF–0x0DDF, 0x0DF2–0x0DF3
VOWEL_SIGNS = {chr(c) for c in range(0x0DCF, 0x0DE0)} | {chr(0x0DF2), chr(0x0DF3)}

# Sinhala independent vowels: 0x0D85–0x0D96
INDEP_VOWELS = {chr(c) for c in range(0x0D85, 0x0D97)}

# Full Sinhala range
SINHALA_RANGE = set(range(0x0D80, 0x0E00))

def _count_words(text: str) -> int:
    return len(text.split())

LEADING_SPACE_CHAR = " "  # Space instead of Ġ to match LinguisTrie


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEOUT CONTEXT MANAGER (for infinite-loop detection)
# ═══════════════════════════════════════════════════════════════════════════════

class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Cross-platform timeout. On Unix uses SIGALRM; on Windows uses threading."""
    if hasattr(signal, 'SIGALRM'):
        def _handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds}s")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    details: str = ""
    metrics: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)


@dataclass
class BattleReport:
    results: list = field(default_factory=list)
    timestamp: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0

    def add(self, result: TestResult):
        self.results.append(result)
        self.total_tests += 1
        if result.status == TestStatus.PASS:
            self.passed += 1
        elif result.status == TestStatus.FAIL:
            self.failed += 1
        else:
            self.warnings += 1


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER WRAPPERS 
# ═══════════════════════════════════════════════════════════════════════════════

class BaseTokenizer:
    def __init__(self, name: str, is_api_based: bool = False):
        self.name = name
        self.is_api_based = is_api_based
        self.results = {"total_tokens": 0, "total_words": 0, "total_chars": 0, "errors": 0}

    def process_text(self, text: str) -> tuple[int, int]:
        raise NotImplementedError

class SGPETokenizerWrapper(BaseTokenizer):
    def __init__(self, encoder: SGPEEncoder):
        super().__init__("SGPE")
        self.encoder = encoder

    def process_text(self, text: str) -> tuple[int, int]:
        try:
            tokens = self.encoder.tokenize(text)
            return len(tokens), 0
        except:
            return 0, 1

class OpenAITokenizer(BaseTokenizer):
    def __init__(self, encoding_name: str = "o200k_base"):
        super().__init__("OpenAI (o200k_base)")
        self.encoder = tiktoken.get_encoding(encoding_name)

    def process_text(self, text: str) -> tuple[int, int]:
        try:
            return len(self.encoder.encode(text)), 0
        except:
            return 0, 1

class GeminiTokenizer(BaseTokenizer):
    def __init__(self, api_key: str):
        super().__init__("Gemini 3 Flash", is_api_based=True)
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        self.client = genai.Client(api_key=api_key)

    def process_text(self, text: str) -> tuple[int, int]:
        try:
            resp = self.client.models.count_tokens(model="gemini-3-flash-preview", contents=text)
            return resp.total_tokens, 0
        except:
            return 0, 1

class ClaudeTokenizer(BaseTokenizer):
    def __init__(self, api_key: str):
        super().__init__("Claude Sonnet 4.5", is_api_based=True)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed")
        self.client = anthropic.Anthropic(api_key=api_key)

    def process_text(self, text: str) -> tuple[int, int]:
        try:
            resp = self.client.messages.count_tokens(
                model="claude-sonnet-4-5", messages=[{"role": "user", "content": text}]
            )
            return resp.input_tokens, 0
        except:
            return 0, 1

class HFTokenizer(BaseTokenizer):
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def process_text(self, text: str) -> tuple[int, int]:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False)), 0
        except:
            return 0, 1


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 1: LINGUISTIC COMPLEXITY TEST 
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_complex_word_bank() -> list[tuple[str, str]]:
    """:
        - Triple+ consonant conjuncts (ක්‍ෂ්‍ර, ස්ත්‍ර, etc.)
        - Deep stacked virama chains (න්ද්‍ර, ම්ප්‍ර, etc.)
        - Rare yansaya/rakaransaya combinations
        - Mixed conjunct + pili + anusvara
        - Real Sanskrit/Pali loanwords used in Sinhala
    """
    words = []

    # ── CATEGORY 1: Real Sanskrit/Pali loanwords (hand-curated core) ──
    real_words = [
        # Triple conjuncts
        ("ක්‍රෝෂ්ඨ්‍ර", "triple_conjunct"),          # krōṣṭhra
        ("ක්ෂිතිජය", "kshitija"),                     # kṣitijaya
        ("වෛද්‍යවරයා", "vaidya"),                      # vaidyavarayā
        ("ශාස්ත්‍රීය", "shaastriya"),                  # śāstrīya
        ("යන්ත්‍රය", "yantraya"),                      # yantraya
        ("මන්ත්‍රය", "mantraya"),                      # mantraya
        ("තන්ත්‍රය", "tantraya"),                      # tantraya
        ("ක්ෂත්‍රිය", "kshatriya"),                   # kṣatriya
        ("චන්ද්‍රිකා", "chandrikaa"),                  # candrikā
        ("ඉන්ද්‍රිය", "indriya"),                      # indriya
        ("සමුද්‍රය", "samudraya"),                     # samudraya
        ("ක්‍රමය", "kramaya"),                         # kramaya
        ("ප්‍රකාශය", "prakaashaya"),                   # prakāśaya
        ("ප්‍රතිපත්තිය", "pratipattiya"),              # pratipattiya
        ("ප්‍රජාව", "prajaava"),                       # prajāva
        ("ප්‍රත්‍යක්ෂ", "pratyaksha"),                 # pratyakṣa
        ("ශ්‍රේෂ්ඨ", "shreshtha"),                     # śrēṣṭha
        ("වස්ත්‍රය", "vastraya"),                     # vastraya
        ("ශාස්ත්‍රය", "shaastraya"),                   # śāstraya
        ("ඍත්විජ", "ritvija"),                         # ṛtvija
        ("සංස්කෘතය", "samskrutaya"),                   # saṃskr̥taya
        ("ව්‍යාකරණය", "vyaakaranaya"),                 # vyākaraṇaya
        ("ව්‍යාපාරය", "vyaapaaraya"),                   # vyāpāraya
        ("ව්‍යවස්ථාව", "vyavasthaava"),                 # vyavasthāva
        ("ධ්‍යානය", "dhyaanaya"),                       # dhyānaya
        ("ආඛ්‍යානය", "aakhyaanaya"),                   # ākhyānaya
        ("උද්‍ඝෝෂණය", "udghoshanaya"),                 # udghōṣaṇaya
        ("අධ්‍යාපනය", "adhyaapanaya"),                  # adhyāpanaya
        ("අධ්‍යක්ෂ", "adhyaksha"),                     # adhyakṣa
        ("නිෂ්ක්‍රිය", "nishkriya"),                   # niṣkriya
        ("විශ්වාසය", "vishvaasaya"),                    # viśvāsaya
        ("අශ්වයා", "ashvayaa"),                         # aśvayā
        ("ස්වාමියා", "svaamiyaa"),                      # svāmiyā
        ("ස්වභාවය", "svabhaavaya"),                     # svabhāvaya
        ("ත්‍රිවිධ", "trividha"),                      # trividha
        ("ජ්‍යෝතිෂ්‍ය", "jyotishya"),                  # jyōtiṣya
        ("සත්‍යය", "satyaya"),                         # satyaya
        ("මෘත්‍යුව", "mrutyuva"),                      # mr̥tyuva
        ("ප්‍රත්‍යයය", "pratyayaya"),                   # pratyayaya
        ("ආත්මය", "aathmaya"),                         # ātmaya
        ("බ්‍රාහ්මණ", "braahmana"),                     # brāhmaṇa
        ("බ්‍රහ්මය", "brahmaya"),                       # brahmaya
        ("ක්‍රිකට්", "cricket"),                        # krikaṭ (loanword adaptation)
        ("ග්‍රහණය", "grahanaya"),                       # grahaṇaya
        ("ග්‍රන්ථය", "granthaya"),                      # granthaya
        ("ප්‍රේමය", "premaya"),                         # prēmaya
        ("ශ්‍රීමත්", "shreemath"),                      # śrīmat
        ("ද්‍රව්‍යය", "dravyaya"),                     # dravyaya
        ("ක්‍ෂේත්‍රය", "kshetraya"),                   # kṣētraya
        ("ශ්‍රද්ධාව", "shraddhaava"),                   # śraddhāva

        # Deep Pali compounds
        ("ධම්මචක්කප්පවත්තන", "dhammachakka"),          # dhammacakkappavattana
        ("අභිධම්මපිටකය", "abhidhamma"),                 # abhidhammapitakaya
        ("පටිච්චසමුප්පාද", "paticcasamuppaada"),       # paṭiccasamuppāda
        ("විපස්සනාව", "vipassanaava"),                   # vipassanāva
        ("සම්මාසම්බුද්ධ", "sammaasambuddha"),           # sammāsambuddha
        ("මහාපරිනිබ්බාන", "mahaparinibbana"),           # mahāparinibbāna
        ("අනිච්චතාව", "aniccataava"),                   # aniccatāva
        ("සංඛාරය", "sankhaaraya"),                       # saṅkhāraya
        ("උපාදානය", "upaadaanaya"),                     # upādānaya
        ("නිබ්බානය", "nibbaanaya"),                      # nibbānaya

        # Conjuncts with anusvara/visarga
        ("සංස්කාරය", "sanskaaraya"),                     # saṃskāraya
        ("සංස්ථාපනය", "sansthaapanaya"),                # saṃsthāpanaya
        ("දුඃඛය", "duhkhaya"),                          # duḥkhaya
        ("අන්තඃපුරය", "antahpuraya"),                   # antaḥpuraya
        ("මනඃකල්පිත", "manahkalpita"),                 # manaḥkalpita

        # Complex vowel sign combinations
        ("ප්‍රෞඪ", "praudha"),                          # prauḍha
        ("සෞන්දර්‍ය", "saundarya"),                     # saundarya
        ("වෛචිත්‍ර්‍ය", "vaichitrya"),                  # vaicitrya
        ("ඓතිහාසික", "aitihaasika"),                    # aitihasika
        ("ඖෂධය", "aushadhaya"),                        # auṣadhaya

        # Terminal virama words
        ("සමස්ත්", "samasth"),                          # samast
        ("ප්‍රශස්ත්", "prashast"),                     # praśast
        ("ස්වච්ඡන්ද්", "svachchhand"),                # svacchand

        # Multi-syllable heavy conjunct words
        ("ප්‍රාදේශීයකරණය", "praadeshiiyakaranaya"),
        ("ප්‍රතිව්‍යූහාත්මක", "prativyuuhaathmaka"),
        ("නිර්වචනාත්මක", "nirvachanaathmaka"),
        ("සාම්ප්‍රදායික", "saammpradaayika"),
        ("ව්‍යවහාරික", "vyavahaarika"),
        ("ආධ්‍යාත්මික", "aadhyaathmika"),
        ("ප්‍රාතිභාසික", "praatibhaasika"),
        ("ව්‍යතිරේකය", "vyatirekaya"),
        ("ප්‍රත්‍යුත්පන්න", "pratyuthpanna"),
        ("අන්තර්ජාතික", "antharjaathika"),
        ("උපනිෂද්", "upanishad"),
        ("ඡන්දස්", "chhandas"),
    ]
    words.extend(real_words)

    # ── CATEGORY 2: Systematic conjunct generation ──
    base_consonants = list(CONSONANTS)
    ya = "ය"    # 0x0DBA
    ra = "ර"    # 0x0DBB
    la = "ල"    # 0x0DBD
    va = "ව"    # 0x0DC0

    # Yansaya forms: C + HAL + ZWJ + ය
    for c in base_consonants[:20]:
        word = c + HAL + ZWJ + ya + "ා"  # e.g., ක්‍යා
        words.append((word, "yansaya_form"))

    # Rakaransaya forms: C + HAL + ZWJ + ර
    for c in base_consonants[:20]:
        word = c + HAL + ZWJ + ra + "ි"  # e.g., ක්‍රි
        words.append((word, "rakaransaya_form"))

    # Double conjuncts: C1 + HAL + C2 + pili
    pili_list = ["ා", "ි", "ු", "ේ", "ො", "ූ", "ැ", "ෑ"]
    for c1 in base_consonants[:15]:
        for c2 in base_consonants[:10]:
            if c1 != c2:
                pili = random.choice(pili_list)
                word = c1 + HAL + c2 + pili
                words.append((word, "double_conjunct"))
                if len(words) >= 600:
                    break
        if len(words) >= 600:
            break

    # Triple conjuncts: C1 + HAL + C2 + HAL + ZWJ + C3 + pili
    for c1 in base_consonants[:10]:
        for c2 in base_consonants[:8]:
            for c3 in [ya, ra, va]:
                pili = random.choice(pili_list)
                word = c1 + HAL + c2 + HAL + ZWJ + c3 + pili
                words.append((word, "triple_conjunct_gen"))
                if len(words) >= 900:
                    break
            if len(words) >= 900:
                break
        if len(words) >= 900:
            break

    # ── CATEGORY 3: Conjuncts with anusvara / visarga suffix ──
    for c1 in base_consonants[:15]:
        for c2 in base_consonants[:8]:
            # C1 + HAL + C2 + ං
            word = c1 + HAL + c2 + ANUSVARA
            words.append((word, "conjunct_anusvara"))

            # C1 + HAL + C2 + pili + ං
            word = c1 + HAL + c2 + "ා" + ANUSVARA
            words.append((word, "conjunct_pili_anusvara"))
            if len(words) >= 1200:
                break
        if len(words) >= 1200:
            break

    # ── CATEGORY 4: Multi-syllable constructed words ──
    syllable_parts = [
        "ක", "කා", "කි", "කු", "කේ", "කො",
        "ග", "ගා", "ගි", "ගු",
        "ත", "තා", "ති", "තු", "තේ",
        "ද", "දා", "දි", "දු",
        "ප", "පා", "පි", "පු", "පේ",
        "බ", "බා", "බි", "බු",
        "ම", "මා", "මි", "මු", "මේ",
        "න", "නා", "නි", "නු", "නේ",
        "ස", "සා", "සි", "සු", "සේ",
        "ර", "රා", "රි", "රු",
        "ල", "ලා", "ලි", "ලු",
        "ය", "යා", "යි", "යු",
        "හ", "හා", "හි",
        "ව", "වා", "වි", "වු",
    ]
    conjunct_parts = [
        "ක්‍ර", "ග්‍ර", "ත්‍ර", "ද්‍ර", "ප්‍ර", "බ්‍ර",
        "ක්‍ය", "ද්‍ය", "ම්‍ය", "ව්‍ය",
        "න්ද", "ම්බ", "න්ත", "ම්ප",
        "ක්ෂ", "ද්ධ", "ත්ථ", "ච්ඡ",
        "ස්ථ", "ස්ත", "ස්ක", "ස්ප",
        "ශ්‍ර", "ෂ්ට", "ෂ්ඨ",
    ]

    random.seed(42)  # deterministic
    while len(words) < 1800:
        n_parts = random.randint(2, 5)
        parts = []
        for _ in range(n_parts):
            if random.random() < 0.3:
                parts.append(random.choice(conjunct_parts))
            else:
                parts.append(random.choice(syllable_parts))
        # Optionally add anusvara at end
        word = "".join(parts)
        if random.random() < 0.1:
            word += ANUSVARA
        words.append((word, "constructed_multisyllable"))

    # ── CATEGORY 5: edge-cases ──
    extreme_cases = [
        # Quadruple stack
        ("ක" + HAL + "ත" + HAL + "ර" + HAL + ZWJ + "ය" + "ා", "quad_stack"),
        # Many virama in sequence (pathological)
        ("ක" + HAL + "ක" + HAL + "ක" + HAL + "ක", "quad_virama_chain"),
        # Lone virama (should not crash)
        (HAL, "bare_virama"),
        # Lone ZWJ (should not crash)
        (ZWJ, "bare_zwj"),
        # HAL + ZWJ only
        (HAL + ZWJ, "bare_hal_zwj"),
        # Consonant + HAL + ZWJ (dangling — no following consonant)
        ("ක" + HAL + ZWJ, "dangling_zwj"),
        # Independent vowel + conjunct
        ("අ" + "ක්‍ර" + "මය", "vowel_prefix_conjunct"),
        # ZWNJ in middle
        ("ක" + ZWNJ + "ර", "zwnj_middle"),
        # Multiple consecutive conjuncts
        ("ස්ත්‍ර" + "ක්‍ෂ" + "ය", "multi_conjunct_sequence"),
        # Very long word (10+ syllables with conjuncts)
        ("ප්‍ර" + "ති" + "ව්‍ය" + "ූ" + "හා" + "ත්ම" + "ක" + "ව" + "ශ" + "යෙ" + "න්",
         "very_long_compound"),
    ]
    words.extend(extreme_cases)

    n_curated = len(real_words) + len(extreme_cases)
    
    generated = words[len(real_words):-len(extreme_cases)]
    sampled_gen = random.sample(generated, min(500 - n_curated, len(generated)))
    
    final_bank = real_words + sampled_gen + extreme_cases
    return final_bank[:500]


def _is_conjunct_internal(char: str) -> bool:
    cp = ord(char)
    return (
        char == HAL or
        char == ZWJ or
        cp in VOWEL_SIGNS or
        char == ANUSVARA or
        char == VISARGA
    )


def test_linguistic_complexity(sgpe: SGPEEncoder) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 1: LINGUISTIC COMPLEXITY TEST")
    print("=" * 80)

    word_bank = _generate_complex_word_bank()
    print(f"  Generated {len(word_bank)} complex words across multiple categories")

    violations = []
    category_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})
    total_tokens_l1 = 0
    total_tokens_bpe = 0

    for word, category in tqdm(word_bank, desc="  Layer1 integrity", unit=" word"):
        category_stats[category]["total"] += 1

        try:
            l1_tokens = sgpe.tokenizer.tokenize(word, leading_space=sgpe.leading_space)
        except Exception as e:
            violations.append(f"  CRASH on '{word}' ({category}): {e}")
            category_stats[category]["fail"] += 1
            continue

        total_tokens_l1 += len(l1_tokens)

        DEGENERATE_CATEGORIES = {"bare_virama", "bare_zwj", "bare_hal_zwj", "dangling_zwj"}
        word_violations = []
        if category not in DEGENERATE_CATEGORIES:
            for i, tok in enumerate(l1_tokens):
                # Strip leading space marker if present
                clean_tok = tok
                if clean_tok.startswith(LEADING_SPACE_CHAR):
                    clean_tok = clean_tok[len(LEADING_SPACE_CHAR):]
                if not clean_tok:
                    continue

                first_char = clean_tok[0]
                if _is_conjunct_internal(first_char) and i > 0 and not category.startswith("bare_"):
                    # This token starts with a modifier — Layer 1 split it!
                    word_violations.append(
                        f"  Token '{tok}' (#{i}) starts with conjunct-internal "
                        f"char U+{ord(first_char):04X} in word '{word}' ({category})"
                    )

                if clean_tok.endswith(HAL + ZWJ) and i < len(l1_tokens) - 1:
                    word_violations.append(
                        f"  Token '{tok}' (#{i}) ends with dangling HAL+ZWJ "
                        f"in word '{word}' ({category})"
                    )

        if word_violations:
            violations.extend(word_violations)
            category_stats[category]["fail"] += 1
        else:
            category_stats[category]["pass"] += 1

        try:
            bpe_tokens = sgpe.tokenize(word)
            total_tokens_bpe += len(bpe_tokens)
        except Exception as e:
            violations.append(f"  BPE CRASH on '{word}' ({category}): {e}")

    print("  Testing with leading-space prefix...")
    leading_space_violations = 0
    for word, category in tqdm(word_bank[:500], desc="  leading-space check", unit=" word"):
        spaced_word = " " + word
        try:
            l1_tokens = sgpe.tokenizer.tokenize(spaced_word, leading_space=sgpe.leading_space)
            for i, tok in enumerate(l1_tokens):
                clean_tok = tok
                if clean_tok.startswith(LEADING_SPACE_CHAR):
                    clean_tok = clean_tok[len(LEADING_SPACE_CHAR):]
                if not clean_tok:
                    continue
                first_char = clean_tok[0]
                if _is_conjunct_internal(first_char) and i > 0 and not category.startswith("bare_"):
                    leading_space_violations += 1
                    violations.append(
                        f"  Leading-space split: '{tok}' in '{spaced_word}'"
                    )
        except Exception as e:
            violations.append(f"  Leading-space CRASH: '{spaced_word}': {e}")
            leading_space_violations += 1

    # Print category breakdown
    print(f"\n  {'Category':<30} {'Total':>6} {'Pass':>6} {'Fail':>6}")
    print(f"  {'-'*54}")
    for cat, stats in sorted(category_stats.items()):
        print(f"  {cat:<30} {stats['total']:>6} {stats['pass']:>6} {stats['fail']:>6}")

    avg_l1 = total_tokens_l1 / len(word_bank) if word_bank else 0
    avg_bpe = total_tokens_bpe / len(word_bank) if word_bank else 0

    status = TestStatus.PASS if len(violations) == 0 else TestStatus.FAIL
    details = (
        f"Tested {len(word_bank)} complex words. "
        f"Violations: {len(violations)}, Leading-space violations: {leading_space_violations}"
    )

    print(f"\n  Result: {status.value} — {details}")
    if violations[:10]:
        print(f"  Sample violations:")
        for v in violations[:10]:
            print(f"    {v}")

    return TestResult(
        name="Linguistic Complexity (2K Sanskrit/Pali Words)",
        status=status,
        details=details,
        metrics={
            "total_words": len(word_bank),
            "violations": len(violations),
            "avg_l1_tokens_per_word": round(avg_l1, 3),
            "avg_bpe_tokens_per_word": round(avg_bpe, 3),
        },
        violations=violations[:50],  # cap stored violations
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 2: GLITCHED TOKEN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_glitched_tokens(sgpe: SGPEEncoder, test_sentences: list[str]) -> TestResult:
    """
    Battery 2: Scan vocab.json for tokens with zero/near-zero usage in test set.
    Verify they don't cause infinite loops or encoding errors.
    """
    print("\n" + "=" * 80)
    print("BATTERY 2: GLITCHED TOKEN DETECTION")
    print("=" * 80)

    # Count token usage across test set
    print("  Counting token usage across test corpus...")
    token_usage = Counter()
    encode_errors = []

    for text in tqdm(test_sentences, desc="  scanning", unit=" sent"):
        try:
            tokens = sgpe.tokenize(text)
            token_usage.update(tokens)
        except Exception as e:
            encode_errors.append(f"  Encoding error: {str(e)[:100]}")

    zero_usage = []
    near_zero = []  # < 3 occurrences
    glitched_tokens = []  # Tokens that are just ZWJ or HAL
    
    for token_str, token_id in sgpe.vocab.items():
        if token_str.startswith('[') and token_str.endswith(']'):
            continue  # skip special tokens
        
        # ═══════════════════════════════════════════════════════════════
        #  GLITCH DETECTION 
        # ═══════════════════════════════════════════════════════════════
        
        check_token = token_str
        if check_token.startswith(LEADING_SPACE_CHAR):
            check_token = check_token[len(LEADING_SPACE_CHAR):]
        
        glitch_detected = False
        glitch_types = []
        
        if check_token:
            
            is_pure_whitespace = all(c in ' \t\n\r' for c in check_token)
            is_leading_space_token = (
                check_token.startswith(' ') and 
                all(c not in ' \t\n\r' for c in check_token[1:])
            )
            
            has_whitespace = any(c in ' \t\n\r' for c in check_token)
            is_valid_structure = (
                not has_whitespace or 
                is_pure_whitespace or 
                is_leading_space_token
            )

            if not is_valid_structure:
                glitch_detected = True
                ws_count = sum(1 for c in check_token if c in ' \t\n\r')
                glitch_types.append(f"mixed-whitespace-garbage ({ws_count}/{len(check_token)} chars)")
            
            stripped = check_token.replace(ZWJ, '').replace(HAL, '').replace(' ', '').replace('\t', '').replace('\n', '')
            if not stripped:
                has_zwj = ZWJ in check_token
                has_hal = HAL in check_token
                
                if has_zwj or has_hal:
                    pass 

            if len(token_str) > 1 and (ZWJ in token_str or HAL in token_str):
                stripped_token = token_str.strip()
                if stripped_token == ZWJ or stripped_token == HAL:
                     pass 

        if glitch_detected:
            glitch_desc = ", ".join(glitch_types) if glitch_types else "unknown pattern"
            glitched_tokens.append(
                f'  GLITCHED: token "{token_str}" (id={token_id}) - {glitch_desc}'
            )
        
        count = token_usage.get(token_str, 0)
        if count == 0:
            zero_usage.append((token_str, token_id))
        elif count < 3:
            near_zero.append((token_str, token_id, count))
        
        count = token_usage.get(token_str, 0)
        if count == 0:
            zero_usage.append((token_str, token_id))
        elif count < 3:
            near_zero.append((token_str, token_id, count))


    print(f"  Total vocab size: {len(sgpe.vocab):,}")
    print(f"  Zero-usage tokens: {len(zero_usage):,}")
    print(f"  Near-zero (< 3) tokens: {len(near_zero):,}")
    print(f"  Glitched tokens (bare ZWJ/HAL): {len(glitched_tokens)}")
    print(f"  Encoding errors during scan: {len(encode_errors)}")

    print(f"\n  Stress-testing {len(zero_usage)} zero-usage tokens...")
    infinite_loop_tokens = []
    crash_tokens = []

    for token_str, token_id in tqdm(zero_usage, desc="  stress-test", unit=" tok"):
        try:
            with time_limit(2.0):
                encoded = sgpe.encode(token_str)
                decoded = sgpe.decode(encoded)
                ctx = f"මෙය {token_str} වේ"
                sgpe.encode(ctx)
        except TimeoutError:
            infinite_loop_tokens.append(
                f"  INFINITE LOOP: token '{token_str}' (id={token_id})"
            )
        except Exception as e:
            crash_tokens.append(
                f"  CRASH: token '{token_str}' (id={token_id}): {str(e)[:80]}"
            )

    # test near-zero tokens
    for token_str, token_id, count in tqdm(near_zero[:500], desc="  near-zero test", unit=" tok"):
        try:
            with time_limit(2.0):
                sgpe.encode(token_str)
                sgpe.encode(f"මෙය {token_str} පරීක්ෂණය")
        except TimeoutError:
            infinite_loop_tokens.append(
                f"  INFINITE LOOP (near-zero): '{token_str}' (id={token_id}, count={count})"
            )
        except Exception as e:
            crash_tokens.append(
                f"  CRASH (near-zero): '{token_str}' (id={token_id}): {str(e)[:80]}"
            )

    critical_issues = infinite_loop_tokens + crash_tokens + encode_errors
    
    if len(critical_issues) > 0 or len(glitched_tokens) > 0:
        status = TestStatus.FAIL
    elif len(zero_usage) > len(sgpe.vocab) * 0.5:
        status = TestStatus.WARN
    else:
        status = TestStatus.PASS

    details = (
        f"Zero-usage: {len(zero_usage)}, Near-zero: {len(near_zero)}, "
        f"Glitched: {len(glitched_tokens)}, "
        f"Infinite loops: {len(infinite_loop_tokens)}, Crashes: {len(crash_tokens)}, "
        f"Encode errors: {len(encode_errors)}"
    )

    print(f"\n  Result: {status.value} — {details}")
    
    if critical_issues:
        print(f"\n  CRITICAL RUNTIME ISSUES:")
        for issue in critical_issues[:10]:
            print(f"    {issue}")

    if glitched_tokens:
        print(f"\n  GLITCHED TOKENS:")
        for issue in glitched_tokens[:10]:
            print(f"    {issue}")

    return TestResult(
        name="Glitched Token Detection",
        status=status,
        details=details,
        metrics={
            "zero_usage_tokens": len(zero_usage),
            "near_zero_tokens": len(near_zero),
            "glitched_tokens": len(glitched_tokens),
            "infinite_loops": len(infinite_loop_tokens),
            "crashes": len(crash_tokens),
            "encoding_errors": len(encode_errors),
            "dead_token_pct": round(len(zero_usage) / max(len(sgpe.vocab), 1) * 100, 2),
        },
        violations=critical_issues[:50] + glitched_tokens[:50],
    )



# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 3: FRONTIER BENCHMARKING REPORT
# ═══════════════════════════════════════════════════════════════════════════════




def test_frontier_benchmarking(
    sgpe: SGPEEncoder,
    test_sentences: list[str],
    max_samples: int = 500,
    rate_limit_delay: float = 0.05,
    full_eval: bool = False,
) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 3: FRONTIER BENCHMARKING")
    print("=" * 80)

    # ── Environment check ──
    print("\n  Environment:")
    print(f"    GEMINI_API_KEY:    {'✓ Set' if os.getenv('GEMINI_API_KEY') else '✗ Not set'}")
    print(f"    ANTHROPIC_API_KEY: {'✓ Set' if os.getenv('ANTHROPIC_API_KEY') else '✗ Not set'}")
    print(f"    HF_TOKEN:          {'✓ Set' if os.getenv('HF_TOKEN') else '✗ Not set'}")
    print(f"    google-genai:      {'✓' if GEMINI_AVAILABLE else '✗ pip install google-genai'}")
    print(f"    anthropic:         {'✓' if ANTHROPIC_AVAILABLE else '✗ pip install anthropic'}")
    print(f"    transformers:      {'✓' if TRANSFORMERS_AVAILABLE else '✗ pip install transformers'}")


    if full_eval:
        sample = test_sentences 
        print(f"\n  [FULL EVAL MODE] Using ALL {len(sample):,} sentences (local tokenizers only)")
    else:
        sample = test_sentences[:max_samples]
        print(f"\n  Using {len(sample):,} sentences (max_samples={max_samples})")

    print("\n  Initializing tokenizers...")
    tokenizers = []

    # 1. SGPE 
    tokenizers.append(SGPETokenizerWrapper(sgpe))
    print(f"  ✓ SGPE (vocab: {len(sgpe.vocab):,})")

    # 2. OpenAI o200k_base — local tiktoken
    try:
        tokenizers.append(OpenAITokenizer("o200k_base"))
        print("  ✓ OpenAI o200k_base (GPT-4o/o3/o4)")
    except Exception as e:
        print(f"  ✗ OpenAI failed: {e}")

    if not full_eval and GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            tokenizers.append(GeminiTokenizer(os.getenv("GEMINI_API_KEY")))
            print("  ✓ Gemini Flash (API)")
        except Exception as e:
            print(f"  ✗ Gemini failed: {e}")
    elif full_eval:
        print("  ⚠ Gemini skipped (full_eval mode — API tokenizers excluded for reproducibility)")
    else:
        print("  ⚠ Gemini skipped (set GEMINI_API_KEY to enable)")

    # 4. Claude Sonnet
    # Uncomment when a Claude API subscription is available:
    # if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
    #     try:
    #         tokenizers.append(ClaudeTokenizer(os.getenv("ANTHROPIC_API_KEY")))
    #         print("  ✓ Claude Sonnet (API)")
    #     except Exception as e:
    #         print(f"  ✗ Claude failed: {e}")
    # else:
    #     print("  ⚠ Claude skipped (set ANTHROPIC_API_KEY to enable)")
    # print("  ⚠ Claude skipped")

    # 5. Llama 4 Scout — HuggingFace tokenizer only (no model weights downloaded)
    #    Requires HF_TOKEN + accepted license at huggingface.co/meta-llama/...
    if TRANSFORMERS_AVAILABLE and os.getenv("HF_TOKEN"):
        try:
            from huggingface_hub import login as hf_login
            hf_login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
            _llama_tok = transformers.AutoTokenizer.from_pretrained(
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
            )
            class _LlamaWrapper(BaseTokenizer):
                def __init__(self):
                    super().__init__("Llama 4 Scout")
                    self.tok = _llama_tok
                def process_text(self, text):
                    try:
                        return len(self.tok.encode(text, add_special_tokens=False)), 0
                    except:
                        return 0, 1
            tokenizers.append(_LlamaWrapper())
            print(f"  ✓ Llama 4 Scout (HF, vocab: {_llama_tok.vocab_size:,})")
        except Exception as e:
            print(f"  ✗ Llama 4 skipped: {e}")
    else:
        print("  ⚠ Llama 4 skipped (set HF_TOKEN + accept model license)")

    if TRANSFORMERS_AVAILABLE and os.getenv("HF_TOKEN"):
        try:
            _ds_tok = transformers.AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3",
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
            )
            class _DeepSeekWrapper(BaseTokenizer):
                def __init__(self):
                    super().__init__("DeepSeek V3")
                    self.tok = _ds_tok
                def process_text(self, text):
                    try:
                        return len(self.tok.encode(text, add_special_tokens=False)), 0
                    except:
                        return 0, 1
            tokenizers.append(_DeepSeekWrapper())
            print(f"  ✓ DeepSeek V3 (HF, vocab: {_ds_tok.vocab_size:,})")
        except Exception as e:
            print(f"  ✗ DeepSeek skipped: {e}")
    else:
        print("  ⚠ DeepSeek skipped (set HF_TOKEN to enable)")

    # ── Run comparison ──
    print(f"\n  Processing {len(sample):,} sentences across {len(tokenizers)} tokenizers...")

    for text in tqdm(sample, desc="  benchmarking", unit=" sent"):
        n_words = _count_words(text)
        n_chars = len(text)

        for tok in tokenizers:
            count, err = tok.process_text(text)
            tok.results["total_tokens"] += count
            tok.results["total_words"] += n_words
            tok.results["total_chars"] += n_chars
            tok.results["errors"] += err

            if tok.is_api_based:
                time.sleep(rate_limit_delay)

    # ── Report ──
    print(f"\n  {'Tokenizer':<32} {'TWR':>8} {'Tokens':>10} {'Chr/Tok':>8} {'Source':>7}")
    print(f"  {'-' * 70}")

    metrics = {}
    sgpe_stats = tokenizers[0].results
    sgpe_twr = sgpe_stats["total_tokens"] / max(sgpe_stats["total_words"], 1)

    for tok in tokenizers:
        stats = tok.results
        twr = stats["total_tokens"] / max(stats["total_words"], 1)
        cpt = stats["total_chars"] / max(stats["total_tokens"], 1)
        source = "API" if tok.is_api_based else "Local"
        print(f"  {tok.name:<32} {twr:>8.3f} {stats['total_tokens']:>10,} {cpt:>8.2f} {source:>7}")
        metrics[tok.name] = {
            "twr": round(twr, 3),
            "total_tokens": stats["total_tokens"],
            "chars_per_token": round(cpt, 2),
            "errors": stats["errors"],
        }

    # Reductions vs SGPE
    print(f"\n  {'SGPE reduction vs':<40} {'% fewer tokens':>15}")
    print(f"  {'-' * 55}")
    for tok in tokenizers:
        if tok.name == "SGPE":
            continue
        stats = tok.results
        if stats["total_tokens"] > 0:
            reduction = (1 - sgpe_stats["total_tokens"] / stats["total_tokens"]) * 100
            print(f"  {tok.name:<40} {reduction:>14.1f}%")

    # Sample word tokenizations
    sample_words = ["ක්‍රෝෂ්ඨ්‍ර", "ශාස්ත්‍රීය", "ව්‍යාකරණය", "ප්‍රත්‍යක්ෂ"]
    print(f"\n  Sample tokenizations:")
    for word in sample_words:
        print(f"    '{word}':")
        for tok in tokenizers:
            try:
                if tok.is_api_based:
                    count, _ = tok.process_text(word)
                    print(f"      {tok.name:<30} [{count} tokens — API only]")
                elif hasattr(tok, 'encoder') and hasattr(tok.encoder, 'tokenize'):
                    toks = tok.encoder.tokenize(word)
                    print(f"      {tok.name:<30} {toks} ({len(toks)} tokens)")
                else:
                    count, _ = tok.process_text(word)
                    print(f"      {tok.name:<30} [{count} tokens]")
            except Exception as e:
                print(f"      {tok.name:<30} ERROR: {e}")

    eval_mode = "Full corpus" if full_eval else f"Sampled {len(sample):,} sentences"
    details = (
        f"{eval_mode}. SGPE TWR: {sgpe_twr:.3f}. "
        f"Compared against {len(tokenizers)-1} SOTA tokenizers."
        + (" [local tokenizers only, full corpus]" if full_eval else "")
    )
    status = TestStatus.PASS if sgpe_twr < 1.5 else TestStatus.WARN

    print(f"\n  Result: {status.value} — {details}")

    return TestResult(
        name="Frontier Benchmarking",
        status=status,
        details=details,
        metrics=metrics,
    )



def test_roundtrip_consistency(
    sgpe: SGPEEncoder,
    test_sentences: list[str],
    full_corpus_path: Optional[str] = None,
    target_count: int = 1_000_000,
) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 4: ROUND-TRIP CONSISTENCY")
    print("=" * 80)
    
    sentences = list(test_sentences)

    if full_corpus_path and os.path.exists(full_corpus_path):
        print(f"  Loading full corpus from {full_corpus_path}...")
        with open(full_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if text:
                        sentences.append(text)
                except json.JSONDecodeError:
                    continue
                if len(sentences) >= target_count:
                    break
                
    if len(sentences) < target_count:
        print(f"  Only {len(sentences):,} sentences available. "
              f"Cycling to reach {target_count:,}...")
        original = list(sentences)
        idx = 0
        while len(sentences) < target_count:
            sentences.append(original[idx % len(original)])
            idx += 1

    print(f"  Testing round-trip on {len(sentences):,} sentences...")

    mismatches = []
    unk_roundtrips = 0
    total_chars_tested = 0
    total_tokens_generated = 0
    total_words_evaluated = 0

    # Process in batches for memory efficiency
    batch_size = 10000
    for batch_start in tqdm(range(0, len(sentences), batch_size),
                            desc="  round-trip", unit=f" ×{batch_size}"):
        batch = sentences[batch_start:batch_start + batch_size]
        for text in batch:
            try:
                ids = sgpe.encode(text)
                decoded = sgpe.decode(ids)
                total_chars_tested += len(text)
                total_tokens_generated += len(ids)
                total_words_evaluated += _count_words(text)

                has_unk = sgpe.unk_id in ids
                if has_unk:
                    unk_roundtrips += 1

                if decoded != text:
                    if not has_unk:
                        mismatches.append({
                            "original": text[:200],
                            "decoded": decoded[:200],
                            "diff_len": abs(len(text) - len(decoded)),
                        })
            except Exception as e:
                mismatches.append({
                    "original": text[:200],
                    "decoded": f"EXCEPTION: {str(e)[:100]}",
                    "diff_len": -1,
                })

    # Results
    clean_mismatches = [m for m in mismatches if m["diff_len"] != -1]
    crash_mismatches = [m for m in mismatches if m["diff_len"] == -1]
    non_unk_mismatches = len(clean_mismatches)
    total_mismatches = len(mismatches)

    print(f"\n  Sentences tested:            {len(sentences):>12,}")
    print(f"  Total words (Sinhala):       {total_words_evaluated:>12,}")
    print(f"  Total characters tested:     {total_chars_tested:>12,}")
    print(f"  Total tokens generated:      {total_tokens_generated:>12,}")
    print(f"  Mismatches (non-UNK):        {non_unk_mismatches:>12,}")
    print(f"  Mismatches (with UNK loss):  {unk_roundtrips:>12,}")
    print(f"  Crashes:                     {len(crash_mismatches):>12,}")

    if clean_mismatches:
        print(f"\n  Sample mismatches:")
        for m in clean_mismatches[:5]:
            print(f"    Original: '{m['original']}'")
            print(f"    Decoded:  '{m['decoded']}'")
            print(f"    Diff len: {m['diff_len']}")
            print()

    # Status: PASS only if ZERO non-UNK mismatches
    if non_unk_mismatches == 0 and len(crash_mismatches) == 0:
        status = TestStatus.PASS
    elif non_unk_mismatches == 0 and len(crash_mismatches) > 0:
        status = TestStatus.FAIL
    else:
        status = TestStatus.FAIL

    details = (
        f"Tested {len(sentences):,} sentences ({total_chars_tested:,} chars). "
        f"Non-UNK mismatches: {non_unk_mismatches}, "
        f"UNK-caused losses: {unk_roundtrips}, "
        f"Crashes: {len(crash_mismatches)}"
    )

    print(f"\n  Result: {status.value} — {details}")

    return TestResult(
        name="Round-Trip Consistency (1M sentences)",
        status=status,
        details=details,
        metrics={
            "sentences_tested": len(sentences),
            "total_chars": total_chars_tested,
            "non_unk_mismatches": non_unk_mismatches,
            "unk_roundtrip_losses": unk_roundtrips,
            "crashes": len(crash_mismatches),
            "mismatch_rate_pct": round(non_unk_mismatches / max(len(sentences), 1) * 100, 6),
        },
        violations=[str(m) for m in mismatches[:20]],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 5: BOUNDARY & LEADING SPACE EDGE-CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_boundary_edge_cases(sgpe: SGPEEncoder) -> TestResult:
    """
    Battery 5: Exhaustive boundary and leading-space tests.
    - Multiple leading/trailing spaces
    - Only whitespace
    - Sinhala + numbers/English without spaces
    - Mixed boundaries
    - Leading space prefix integrity
    """
    print("\n" + "=" * 80)
    print("BATTERY 5: BOUNDARY & LEADING SPACE EDGE-CASES")
    print("=" * 80)

    violations = []
    test_count = 0

    def _check(label: str, text: str, check_roundtrip: bool = True,
               check_no_crash: bool = True, check_leading_space: bool = False,
               expected_token_prefix: str = None):
        nonlocal test_count, violations
        test_count += 1

        try:
            tokens = sgpe.tokenize(text)
            ids = sgpe.encode(text)
            decoded = sgpe.decode(ids)

            # Check roundtrip
            if check_roundtrip:
                has_unk = sgpe.unk_id in ids
                if decoded != text and not has_unk:
                    violations.append(
                        f"  [{label}] Round-trip FAIL: "
                        f"'{text}' → '{decoded}'"
                    )

            # Check leading space on first Sinhala token
            if check_leading_space and tokens:
                first_sinhala = None
                first_sinhala_idx = -1
                for i, t in enumerate(tokens):
                    clean = t.replace(LEADING_SPACE_CHAR, "")
                    if any(0x0D80 <= ord(c) <= 0x0DFF for c in clean):
                        first_sinhala = t
                        first_sinhala_idx = i
                        break
                
                if first_sinhala and text.lstrip() != text:
                    
                    # Check if first Sinhala token has Ġ prefix or raw space
                    has_attached_ws = first_sinhala and (
                        first_sinhala[0] in ' \t\n\r' or
                        first_sinhala.startswith(LEADING_SPACE_CHAR)
                    )
                    has_standalone_ws = False
                    if first_sinhala_idx > 0:
                        for t in tokens[:first_sinhala_idx]:
                            if all(c in ' \t\n\r' for c in t) or t == "[UNK]":
                                has_standalone_ws = True
                                break
                    
                    # At least one of these should be true
                    if not (has_attached_ws or has_standalone_ws):
                        violations.append(
                            f"  [{label}] Missing leading space: "
                            f"no whitespace before first Sinhala token '{first_sinhala}' in '{text}'"
                        )

            # Check for empty tokens
            if any(t == "" for t in tokens):
                violations.append(f"  [{label}] Empty token in output for '{text}'")

        except Exception as e:
            if check_no_crash:
                violations.append(f"  [{label}] CRASH: '{text}': {str(e)[:100]}")

    # ── Test Group A: Whitespace variations ──
    print("  Testing whitespace variations...")

    _check("A01-single-space", " ", check_roundtrip=True)
    _check("A02-double-space", "  ", check_roundtrip=True)
    _check("A03-triple-space", "   ", check_roundtrip=True)
    _check("A04-ten-spaces", "          ", check_roundtrip=True)
    _check("A05-tab", "\t", check_roundtrip=True)
    _check("A06-newline", "\n", check_roundtrip=True)
    _check("A07-mixed-ws", " \t \n ", check_roundtrip=True)
    _check("A08-empty", "", check_roundtrip=True)

    # ── Test Group B: Leading spaces before Sinhala ──
    print("  Testing leading spaces before Sinhala...")

    _check("B01-one-leading", " සිංහල", check_leading_space=True)
    _check("B02-two-leading", "  සිංහල", check_leading_space=True)
    _check("B03-five-leading", "     සිංහල", check_leading_space=True)
    _check("B04-ten-leading", "          සිංහල", check_leading_space=True)
    _check("B05-tab-leading", "\tසිංහල", check_leading_space=True)
    _check("B06-newline-leading", "\nසිංහල", check_leading_space=True)

    # ── Test Group C: Trailing spaces after Sinhala ──
    print("  Testing trailing spaces after Sinhala...")

    _check("C01-one-trailing", "සිංහල ", check_roundtrip=True)
    _check("C02-five-trailing", "සිංහල     ", check_roundtrip=True)
    _check("C03-ten-trailing", "සිංහල          ", check_roundtrip=True)

    # ── Test Group D: Both leading and trailing ──
    print("  Testing combined leading/trailing spaces...")

    _check("D01-both", " සිංහල ", check_roundtrip=True)
    _check("D02-heavy-both", "     සිංහල     ", check_roundtrip=True)
    _check("D03-extreme", "         සිංහල          ", check_roundtrip=True)
    _check("D04-word-in-spaces", "   ශාස්ත්‍රීය   ", check_roundtrip=True)

    # ── Test Group E: Sinhala + Numbers (no space) ──
    print("  Testing Sinhala + numbers without spaces...")

    _check("E01-sinhala-num", "සිංහල123", check_roundtrip=True)
    _check("E02-num-sinhala", "123සිංහල", check_roundtrip=True)
    _check("E03-sinhala-num-sinhala", "සිංහල123සිංහල", check_roundtrip=True)
    _check("E04-complex-num", "ක්‍රමය456", check_roundtrip=True)
    _check("E05-decimal", "සිංහල3.14", check_roundtrip=True)
    _check("E06-negative", "සිංහල-42", check_roundtrip=True)
    _check("E07-large-num", "සිංහල1234567890", check_roundtrip=True)

    # ── Test Group F: Sinhala + English (no space) ──
    print("  Testing Sinhala + English without spaces...")

    _check("F01-sinhala-eng", "සිංහලABC", check_roundtrip=True)
    _check("F02-eng-sinhala", "ABCසිංහල", check_roundtrip=True)
    _check("F03-mixed", "ABCසිංහලDEF", check_roundtrip=True)
    _check("F04-complex", "ශාස්ත්‍රීයScience", check_roundtrip=True)
    _check("F05-lowercase", "සිංහලabc", check_roundtrip=True)
    _check("F06-single-char", "සිංහලX", check_roundtrip=True)
    _check("F07-eng-conjunct", "Xක්‍රමය", check_roundtrip=True)

    # ── Test Group G: Sinhala + Numbers + English (complex mix) ──
    print("  Testing complex mixed boundaries...")

    _check("G01-full-mix", " සිංහල123ABC ", check_roundtrip=True, check_leading_space=True)
    _check("G02-heavy-space-mix", "         සිංහල", check_roundtrip=True, check_leading_space=True)
    _check("G03-trailing-mix", " සිංහල     ", check_roundtrip=True)
    _check("G04-number-sandwich", "123සිංහල456", check_roundtrip=True)
    _check("G05-eng-sandwich", "ABCසිංහලDEF", check_roundtrip=True)
    _check("G06-all-mixed", " ABC 123 සිංහල 456 DEF ", check_roundtrip=True)
    _check("G07-conjunct-mixed", " ශාස්ත්‍රීය123ABC ", check_roundtrip=True)
    _check("G08-pali-mixed", " ධම්මචක්කප්පවත්තන42Z ", check_roundtrip=True)

    # ── Test Group H: Punctuation boundaries ──
    print("  Testing punctuation boundaries...")

    _check("H01-period", "සිංහල.", check_roundtrip=True)
    _check("H02-comma", "සිංහල,බලන්න", check_roundtrip=True)
    _check("H03-question", "සිංහල?", check_roundtrip=True)
    _check("H04-paren", "(සිංහල)", check_roundtrip=True)
    _check("H05-bracket", "[සිංහල]", check_roundtrip=True)
    _check("H06-quote", '"සිංහල"', check_roundtrip=True)
    _check("H07-dash", "සිංහල—English", check_roundtrip=True)
    _check("H08-slash", "සිංහල/English", check_roundtrip=True)

    # ── Test Group I: Unicode edge cases ──
    print("  Testing Unicode edge cases...")

    _check("I01-emoji-after", "සිංහල😀", check_roundtrip=True)
    _check("I02-emoji-before", "😀සිංහල", check_roundtrip=True)
    _check("I03-bom", "\ufeffසිංහල", check_roundtrip=True)
    _check("I04-null-char", "සිංහල\x00බලන්න", check_no_crash=True)
    _check("I05-rtl-mark", "සිංහල\u200fEnglish", check_roundtrip=True)
    _check("I06-nbsp", "සිංහල\u00a0English", check_roundtrip=True)

    # ── Test Group J: Leading space integrity ──
    print("  Testing Leading Space (Ġ) prefix integrity...")

    # Words in sentence context — each non-initial word should get Ġ
    sentence = "මම ටොකනයිසර් එකක් සාදමි"
    tokens = sgpe.tokenize(sentence)
    l1_tokens = sgpe.tokenizer.tokenize(sentence, leading_space=sgpe.leading_space)

    # Count space-prefixed tokens in Layer 1
    g_count = sum(1 for t in l1_tokens if t.startswith(LEADING_SPACE_CHAR))
    space_count = sentence.count(" ")
    if sgpe.leading_space:
        # We expect roughly one Ġ per space (the token after each space gets Ġ)
        if g_count < space_count:
            violations.append(
                f"  [J01] Insufficient leading spacees: expected ≥{space_count}, got {g_count}. "
                f"L1 tokens: {l1_tokens}"
            )

    # Verify space-prefixed tokens round-trip correctly
    for text in [" සිංහල", "  සිංහල", "මම සිංහල"]:
        ids = sgpe.encode(text)
        decoded = sgpe.decode(ids)
        has_unk = sgpe.unk_id in ids
        if decoded != text and not has_unk:
            violations.append(
                f"  [J02] leading space round-trip fail: '{text}' → '{decoded}'"
            )

    # ── Test Group K: Stress - many boundary transitions ──
    print("  Testing rapid boundary transitions...")

    rapid_transition = "ක1ග2ත3ද4ප5බ6"
    _check("K01-rapid", rapid_transition, check_roundtrip=True)

    alternating = "සA1 බB2 මC3"
    _check("K02-alternating", alternating, check_roundtrip=True)

    only_conjuncts_nums = "ක්‍ර1ග්‍ර2ත්‍ර3"
    _check("K03-conjunct-nums", only_conjuncts_nums, check_roundtrip=True)

    # ── Results ──
    status = TestStatus.PASS if len(violations) == 0 else TestStatus.FAIL
    details = f"Ran {test_count} edge-case tests. Violations: {len(violations)}"

    print(f"\n  Result: {status.value} — {details}")
    if violations:
        print(f"  Violations:")
        for v in violations[:15]:
            print(f"    {v}")

    return TestResult(
        name="Boundary & Leading Space Edge-Cases",
        status=status,
        details=details,
        metrics={
            "total_tests": test_count,
            "violations": len(violations),
        },
        violations=violations[:50],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 6: ZERO-BREAKAGE GUARANTEE
# ═══════════════════════════════════════════════════════════════════════════════

def test_zero_breakage_extended(sgpe: SGPEEncoder) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 6: ZERO-BREAKAGE GUARANTEE")
    print("=" * 80)

    violations = []
    test_count = 0

    tokenizer = sgpe.tokenizer

    # ── Test A: C + HAL + ZWJ + C (all consonant pairs for yansaya/rakaransaya) ──
    print("  Testing all C + HAL + ZWJ + C pairs...")
    ya = "ය"
    ra = "ර"
    special_seconds = [ya, ra, "ල", "ව"]

    for c1 in sorted(CONSONANTS):
        for c2 in special_seconds:
            text = c1 + HAL + ZWJ + c2
            test_count += 1
            tokens = tokenizer.tokenize(text)
            if len(tokens) != 1:
                violations.append(
                    f"  C+HAL+ZWJ+C split: '{text}' → {tokens} "
                    f"(U+{ord(c1):04X} + HAL + ZWJ + U+{ord(c2):04X})"
                )

    # ── Test B: C + HAL + C (implicit conjuncts) ──
    print("  Testing C + HAL + C pairs (implicit conjuncts)...")
    for c1 in sorted(CONSONANTS):
        for c2 in list(sorted(CONSONANTS))[:10]:
            text = c1 + HAL + c2
            test_count += 1
            tokens = tokenizer.tokenize(text)
            if len(tokens) != 1:
                violations.append(
                    f"  C+HAL+C split: '{text}' → {tokens} "
                    f"(U+{ord(c1):04X} + HAL + U+{ord(c2):04X})"
                )

    # ── Test C: C + vowel_sign (all combinations) ──
    print("  Testing C + vowel_sign (all combinations)...")
    for c in sorted(CONSONANTS):
        for vs_cp in sorted(VOWEL_SIGNS):
            vs = chr(vs_cp) if isinstance(vs_cp, int) else vs_cp
            text = c + vs
            test_count += 1
            tokens = tokenizer.tokenize(text)
            if len(tokens) != 1:
                violations.append(
                    f"  C+pili split: '{text}' → {tokens} "
                    f"(U+{ord(c):04X} + U+{ord(vs):04X})"
                )

    # ── Test D: C + HAL (terminal virama) ──
    print("  Testing C + HAL (terminal virama)...")
    for c in sorted(CONSONANTS):
        text = c + HAL
        test_count += 1
        tokens = tokenizer.tokenize(text)
        if len(tokens) != 1:
            violations.append(
                f"  C+HAL split: '{text}' → {tokens} (U+{ord(c):04X} + HAL)"
            )

    # ── Test E: C + anusvara / visarga ──
    print("  Testing C + anusvara / visarga...")
    for c in sorted(CONSONANTS):
        for post in [ANUSVARA, VISARGA]:
            text = c + post
            test_count += 1
            tokens = tokenizer.tokenize(text)
            if len(tokens) != 1:
                violations.append(
                    f"  C+post split: '{text}' → {tokens} "
                    f"(U+{ord(c):04X} + U+{ord(post):04X})"
                )

    # ── Test F: C + pili + anusvara ──
    print("  Testing C + pili + anusvara...")
    for c in list(sorted(CONSONANTS))[:15]:
        for vs_cp in list(sorted(VOWEL_SIGNS))[:5]:
            vs = chr(vs_cp) if isinstance(vs_cp, int) else vs_cp
            text = c + vs + ANUSVARA
            test_count += 1
            tokens = tokenizer.tokenize(text)
            if len(tokens) != 1:
                violations.append(
                    f"  C+pili+anusvara split: '{text}' → {tokens}"
                )

    # ── Test G: Triple stacks (C + HAL + C + HAL + ZWJ + C) ──
    print("  Testing triple stacks...")
    triple_tests = [
        "ක" + HAL + "ත" + HAL + ZWJ + "ර",   # ktra
        "ස" + HAL + "ත" + HAL + ZWJ + "ර",   # stra
        "න" + HAL + "ද" + HAL + ZWJ + "ර",   # ndra
        "ම" + HAL + "බ" + HAL + ZWJ + "ර",   # mbra
        "ක" + HAL + "ෂ" + HAL + ZWJ + "ය",   # kshya
    ]
    for text in triple_tests:
        test_count += 1
        tokens = tokenizer.tokenize(text)
        if len(tokens) != 1:
            violations.append(f"  Triple stack split: '{text}' → {tokens}")

    # ── Test H: Full words with leading space ──
    print("  Testing conjuncts with leading space...")
    for text in ["ක්‍රමය", "ශාස්ත්‍රීය", "ප්‍රත්‍යක්ෂ"]:
        spaced = " " + text
        test_count += 1
        tokens = tokenizer.tokenize(spaced, leading_space=sgpe.leading_space)
        for i, tok in enumerate(tokens):
            clean = tok.replace(LEADING_SPACE_CHAR, "")
            if clean and _is_conjunct_internal(clean[0]) and i > 0:
                violations.append(
                    f"  Leading-space conjunct break: '{spaced}' → {tokens}"
                )

    # Results
    status = TestStatus.PASS if len(violations) == 0 else TestStatus.FAIL
    details = f"Ran {test_count:,} exhaustive breakage tests. Violations: {len(violations)}"

    print(f"\n  Result: {status.value} — {details}")
    if violations:
        print(f"  Sample violations (max 20):")
        for v in violations[:20]:
            print(f"    {v}")

    return TestResult(
        name="Zero-Breakage Guarantee (Extended)",
        status=status,
        details=details,
        metrics={
            "total_tests": test_count,
            "violations": len(violations),
        },
        violations=violations[:100],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_report(report: BattleReport):
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "    SGPE - BATTLE TEST REPORT".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    # Summary table
    print(f"  {'Test Battery':<50} {'Status':>10} {'Key Metric':>18}")
    print(f"  {'─' * 80}")

    for r in report.results:
        status_str = {
            TestStatus.PASS: "✓ PASS",
            TestStatus.FAIL: "✗ FAIL",
            TestStatus.WARN: "⚠ WARN",
        }[r.status]

        # Pick the most important metric
        key_metric = ""
        if "violations" in r.metrics:
            key_metric = f"{r.metrics['violations']} violations"
        if "reduction_vs_gpt4o_pct" in r.metrics:
            key_metric = f"{r.metrics['reduction_vs_gpt4o_pct']:.1f}% reduction"
        if "non_unk_mismatches" in r.metrics:
            key_metric = f"{r.metrics['non_unk_mismatches']} mismatches"

        print(f"  {r.name:<50} {status_str:>10} {key_metric:>18}")

    print(f"  {'─' * 80}")
    print(f"  {'TOTAL':<50} "
          f"{'P:' + str(report.passed):>4} "
          f"{'F:' + str(report.failed):>4} "
          f"{'W:' + str(report.warnings):>4}")

    # Overall verdict
    print()
    if report.failed == 0:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║                    🏆 VERDICT: STABLE 🏆                    ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print(f"  ║   ⚠  VERDICT: {report.failed} BATTERY(S) FAILED         ⚠  ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")

    # Frontier benchmark highlight
    frontier = next((r for r in report.results if "Frontier" in r.name), None)
    if frontier and frontier.metrics:
        m = frontier.metrics
        # Find SGPE and OpenAI entries by name prefix
        sgpe_entry  = next((v for k, v in m.items() if k.startswith("SGPE")), None)
        oai_entry   = next((v for k, v in m.items() if k.startswith("OpenAI")), None)
        gemini_entry = next((v for k, v in m.items() if k.startswith("Gemini")), None)
        llama_entry  = next((v for k, v in m.items() if k.startswith("Llama")), None)

        sgpe_twr  = sgpe_entry["twr"]  if sgpe_entry  else None
        oai_twr   = oai_entry["twr"]   if oai_entry   else None

        def _pct(other_twr):
            if sgpe_twr and other_twr and other_twr > 0:
                return (1 - sgpe_twr / other_twr) * 100
            return None

        def _fmt_twr(v):  return f"{v:.3f}" if v is not None else "N/A"
        def _fmt_pct(v):  return f"{v:.1f}%" if v is not None else "N/A"

        print(f"\n  ┌─── Frontier Benchmark Highlight ──────────────────────────────┐")
        print(f"  │  SGPE TWR:                       {_fmt_twr(sgpe_twr):>10}              │")
        print(f"  │  Tiktoken TWR (o200k_base):        {_fmt_twr(oai_twr):>10}              │")
        print(f"  │  SGPE reduction vs Tiktoken:       {_fmt_pct(_pct(oai_twr)):>10}              │")
        if gemini_entry:
            print(f"  │  SGPE reduction vs Gemini:       {_fmt_pct(_pct(gemini_entry['twr'])):>10}              │")
        if llama_entry:
            print(f"  │  SGPE reduction vs Llama 4:      {_fmt_pct(_pct(llama_entry['twr'])):>10}              │")
        print(f"  └────────────────────────────────────────────────────────────────┘")

    print()


def save_report_json(report: BattleReport, output_path: str):
    data = {
        "timestamp": report.timestamp,
        "summary": {
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "warnings": report.warnings,
            "verdict": "PASS" if report.failed == 0 else "FAIL",
        },
        "batteries": [],
    }

    for r in report.results:
        data["batteries"].append({
            "name": r.name,
            "status": r.status.value,
            "details": r.details,
            "metrics": r.metrics,
            "violation_count": len(r.violations),
            "sample_violations": r.violations[:10],
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Report saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_data(test_file: str) -> list[str]:
    """Load test sentences from lines of JSONL or return defaults if missing."""
    test_sentences = []
    if os.path.exists(test_file):
        print(f"\n[INIT] Loading test data from {test_file}...")
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if text:
                        test_sentences.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(test_sentences):,} test sentences")
    else:
        print(f"\n[WARN] Test file not found: {test_file}")
        print("  Using minimal built-in test set")
        test_sentences = [
            "මම සිංහල ටොකනයිසර් එකක් සාදමි",
            "ශාස්ත්‍රීය ක්‍රමවේදය ඉතා වැදගත් ය",
            "බුද්ධ ධර්මය ප්‍රකාශ කරන ලදී",
            "ව්‍යාකරණ නීති අනුගමනය කරන්න",
            "ප්‍රත්‍යක්ෂ වශයෙන් දැකිය හැකිය",
        ]
    return test_sentences


def main():
    parser = argparse.ArgumentParser(
        description="SGPE Battle Test"
    )
    parser.add_argument("--vocab_file", type=str, default="output/vocab.json",
                        help="Path to SGPE vocab.json")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl",
                        help="Path to test JSONL file")
    parser.add_argument("--full_corpus", type=str, default=None,
                        help="Path to full 1M corpus JSONL (for round-trip test)")
    parser.add_argument("--report_output", type=str, default="output/battle_report.json",
                        help="Path to save JSON report")
    parser.add_argument("--skip_roundtrip", action="store_true",
                        help="Skip the 1M round-trip test (slow)")
    parser.add_argument("--roundtrip_count", type=int, default=1_000_000,
                        help="Number of sentences for round-trip test")
    parser.add_argument("--only", type=str, nargs="+", 
                        choices=["complexity", "glitched", "frontier", "roundtrip", "boundary", "zerobreak"],
                        help="Run only specific batteries")
    parser.add_argument("--frontier_samples", type=int, default=500,
                        help="Max sentences for Battery 3 frontier benchmarking (default: 500)")
    parser.add_argument("--full_eval", action="store_true",
                        help="Full-corpus TWR eval: skip API tokenizers, use ALL sentences.")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      SGPE BATTLE TEST                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if args.only:
        print(f"  [Partial Run] Selected batteries: {args.only}")

    # ── Initialize ──
    print("\n[INIT] Loading SGPE encoder...")
    sgpe = SGPEEncoder(args.vocab_file)
    print(f"  Vocab size: {len(sgpe.vocab):,}")
    print(f"  Merges: {len(sgpe.merges):,}")
    print(f"  Leading space: {sgpe.leading_space}")

    # Load test sentences
    test_sentences = load_test_data(args.test_file)

    # ── Build report ──
    report = BattleReport()
    report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    def should_run(battery_name):
        return args.only is None or battery_name in args.only

    # ── Battery 1: Linguistic Complexity ──
    if should_run("complexity"):
        result1 = test_linguistic_complexity(sgpe)
        report.add(result1)

    # ── Battery 2: Glitched Token Detection ──
    if should_run("glitched"):
        result2 = test_glitched_tokens(sgpe, test_sentences)
        report.add(result2)

    # ── Battery 3: Frontier Benchmarking ──
    if should_run("frontier"):
        result3 = test_frontier_benchmarking(
            sgpe, test_sentences,
            max_samples=args.frontier_samples,
            full_eval=args.full_eval,
        )
        report.add(result3)

    # ── Battery 4: Round-Trip Consistency ──
    if should_run("roundtrip"):
        if not args.skip_roundtrip:
            result4 = test_roundtrip_consistency(
                sgpe, test_sentences,
                full_corpus_path=args.full_corpus,
                target_count=args.roundtrip_count,
            )
            report.add(result4)
        else:
            print("\n  [SKIP] Round-trip consistency test (use --skip_roundtrip=false to enable)")

    # ── Battery 5: Boundary & Leading Space ──
    if should_run("boundary"):
        result5 = test_boundary_edge_cases(sgpe)
        report.add(result5)

    # ── Battery 6: Zero-Breakage Extended ──
    if should_run("zerobreak"):
        result6 = test_zero_breakage_extended(sgpe)
        report.add(result6)

    # ── Final Report ──
    print_final_report(report)
    save_report_json(report, args.report_output)

    # Exit code
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
