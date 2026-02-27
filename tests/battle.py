"""
WWHO — Battle Test Suite
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from battle_core import (
    SGPEEncoder,
    BattleReport,
    TestResult,
    TestStatus,
    load_test_data,
    test_linguistic_complexity,
    test_glitched_tokens,
    test_roundtrip_consistency,
    test_boundary_edge_cases,
    test_zero_breakage_extended,
    print_final_report,
    save_report_json,
    LEADING_SPACE_CHAR,
    HAL, ZWJ, ANUSVARA,
)

from linguis_trie import build_devanagari_linguis_trie, LinguisTrie
from router import CodeSwitchSegmenter, Script

try:
    from encoder import WWHOMetaEncoder
    _META_OK = True
except ImportError:
    _META_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# DEVANAGARI CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

VIRAMA_DEV     = "\u094D"   # Devanagari halant
ANUSVARA_DEV   = "\u0902"   # Devanagari anusvara (ं)
VISARGA_DEV    = "\u0903"   # Devanagari visarga (ः)
CHANDRABINDU   = "\u0901"   # Devanagari chandrabindu (ँ)
NUKTA          = "\u093C"   # Nukta

DEV_VOWEL_SIGNS = {chr(c) for c in range(0x093E, 0x094D)}  # ा-ौ
DEV_CONSONANTS  = {chr(c) for c in range(0x0915, 0x093A)}  # क-ह


def _is_dev_conjunct_internal(ch: str) -> bool:
    return (
        ch == VIRAMA_DEV or
        ch == ANUSVARA_DEV or
        ch == VISARGA_DEV or
        ch == CHANDRABINDU or
        ch == NUKTA or
        ch in DEV_VOWEL_SIGNS
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 7: DEVANAGARI LINGUISTIC COMPLEXITY
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_devanagari_word_bank() -> list[tuple[str, str]]:
    words = []

    # Level 1-2: Simple and basic-matra words
    simple = [
        ("घर",         "simple"), ("फल",        "simple"), ("कमल",       "simple"),
        ("आम",         "simple"), ("किताब",     "matra"),  ("सूरज",      "matra"),
        ("विद्यालय",   "matra"),  ("हृदय",      "conjunct"),
    ]
    words.extend(simple)

    # Level 3-4: Conjuncts including anusvara absorption
    conjuncts = [
        ("प्रकृति",    "conjunct"), ("संस्कृति",  "anusvara"),
        ("आशीर्वाद",   "conjunct"), ("उज्ज्वल",  "double_conjunct"),
        ("युधिष्ठिर",  "complex"),  ("वैयक्तिकता","complex"),
        ("स्वातन्त्र्योत्तर", "very_complex"),
    ]
    words.extend(conjuncts)

    # Level 5: Sanskrit compounds
    sanskrit = [
        ("अष्टविंशति", "sanskrit"), ("तितिक्षा",   "sanskrit"),
        ("स्थितप्रज्ञ","sanskrit"),  ("निरन्तरान्धकारित", "sanskrit"),
        ("मत्तमातङ्गगामिनी", "super_compound"),
    ]
    words.extend(sanskrit)

    # Systematic: C + VIRAMA + C conjuncts
    consonant_list = sorted(list(DEV_CONSONANTS))
    matras = ["\u093E", "\u093F", "\u0940", "\u0941", "\u0942", "\u0947", "\u0948"]
    import random
    random.seed(42)
    for c1 in consonant_list:
        for c2 in consonant_list:
            if c1 != c2:
                for _ in range(3):
                    matra = random.choice(matras)
                    words.append((c1 + VIRAMA_DEV + c2 + matra, "double_conjunct_gen"))

    # Anusvara absorption: C + ANUSVARA sequences
    for c in consonant_list:
        words.append((c + ANUSVARA_DEV + "स्कृ" + "ति", "anusvara_prefix"))
        words.append((c + VIRAMA_DEV + "त" + ANUSVARA_DEV + "र", "conjunct_anusvara"))

    # Extreme: long compound
    words.append((
        "निरन्तरान्धकारितदिगन्तरकन्दलदमन्दसुधारसबिन्दुसान्द्रतरघनाघनवृन्दसन्देहकर",
        "extreme_compound"
    ))

    curated = simple + conjuncts + sanskrit
    extreme = words[-1:]
    generated = words[len(curated):-1]
    
    sampled_gen = random.sample(generated, min(500 - len(curated) - len(extreme), len(generated)))
    final_bank = curated + sampled_gen + extreme
    return final_bank[:500]


def test_devanagari_complexity(dfa: LinguisTrie | None = None) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 7: DEVANAGARI LINGUISTIC COMPLEXITY")
    print("=" * 80)

    if dfa is None:
        dfa = build_devanagari_linguis_trie()

    word_bank = _generate_devanagari_word_bank()
    print(f"  Generated {len(word_bank)} Devanagari test words")

    violations = []
    from collections import defaultdict
    category_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})

    for word, category in tqdm(word_bank, desc="  Devanagari Layer1", unit=" word"):
        category_stats[category]["total"] += 1
        try:
            tokens = dfa.tokenize(word, leading_space=False)
        except Exception as e:
            violations.append(f"  CRASH on '{word}' ({category}): {e}")
            category_stats[category]["fail"] += 1
            continue

        word_violations = []
        for i, tok in enumerate(tokens):
            clean = tok.lstrip(LEADING_SPACE_CHAR)
            if not clean:
                continue
            if _is_dev_conjunct_internal(clean[0]) and i > 0:
                word_violations.append(
                    f"  Token '{tok}' (#{i}) starts with conjunct-internal "
                    f"char U+{ord(clean[0]):04X} in word '{word}' ({category})"
                )

        if word_violations:
            violations.extend(word_violations)
            category_stats[category]["fail"] += 1
        else:
            category_stats[category]["pass"] += 1

    print(f"\n  {'Category':<28} {'Total':>6} {'Pass':>6} {'Fail':>6}")
    print(f"  {'-'*52}")
    for cat, stats in sorted(category_stats.items()):
        print(f"  {cat:<28} {stats['total']:>6} {stats['pass']:>6} {stats['fail']:>6}")

    status = TestStatus.PASS if len(violations) == 0 else TestStatus.FAIL
    details = f"Tested {len(word_bank)} Devanagari words. Violations: {len(violations)}"
    print(f"\n  Result: {status.value} — {details}")
    if violations[:10]:
        for v in violations[:10]:
            print(f"    {v}")

    return TestResult(
        name="Devanagari Linguistic Complexity",
        status=status,
        details=details,
        metrics={"total_words": len(word_bank), "violations": len(violations)},
        violations=violations[:50],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 8: CODE-SWITCHING INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

_SINHALA_RE  = re.compile(r'[\u0D80-\u0DFF\u200D]')
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
_LATIN_RE    = re.compile(r'[A-Za-z0-9]')

CODE_SWITCH_TEST_CASES = [
    # (text, description)
    ("Hello, ශ්‍රී ලංකාව!",              "simple_sinhala_english"),
    ("const x = ප්‍රකාශය;",             "code_sinhala"),
    ("मेरा नाम है और I love Python",     "devanagari_english"),
    ("function foo() { return 'ශ්‍රී'; }","code_sinhala_mixed"),
    ("ශ‍්‍රී ලංකාව is beautiful",        "sinhala_english_mixed"),
    ("print('नमस्ते') # Say Hello",       "python_devanagari_comment"),
    ("ඒ කියන්නේ, GPE Tokenizer English","sinhala_english_complex"),
    ("for i in range(10): # ලූපය",       "python_sinhala_comment"),
    ("SELECT * FROM users WHERE नाम='राम'", "sql_devanagari"),
    ("const create_func = (p1, p2) => { return 'ශ්‍රී' + p1; }", "arrow_fn_sinhala"),
    ("",                                  "empty_string"),
    ("   ",                               "whitespace_only"),
    ("123 + 456 = ෆ",                    "math_sinhala"),
]


def test_code_switching_integrity(sgpe: SGPEEncoder) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 8: CODE-SWITCHING INTEGRITY")
    print("=" * 80)

    segmenter = CodeSwitchSegmenter()
    violations = []
    crash_count = 0
    total = len(CODE_SWITCH_TEST_CASES)

    for text, desc in CODE_SWITCH_TEST_CASES:
        if not text.strip():
            continue
        try:
            segments = segmenter.segment(text)
            for seg in segments:
                if seg.script == Script.LATIN:
                    if _SINHALA_RE.search(seg.text) or _DEVANAGARI_RE.search(seg.text):
                        violations.append(
                            f"  ROUTING: Indic chars in Latin segment '{seg.text}' ({desc})"
                        )
                elif seg.script in (Script.SINHALA, Script.DEVANAGARI):
                    pure_latin = re.findall(r'[A-Za-z]', seg.text)
                    if pure_latin:
                        violations.append(
                            f"  ROUTING: Latin chars {pure_latin} in Indic segment '{seg.text}' ({desc})"
                        )

            # 2. Full pipeline
            tokens = sgpe.tokenize(text)
            ids    = sgpe.encode(text)
            if not tokens:
                violations.append(f"  EMPTY: no tokens for '{text}' ({desc})")

            print(f"  [{desc:<35}] {len(tokens):>3} tokens | {tokens[:8]}")

        except Exception as e:
            crash_count += 1
            violations.append(f"  CRASH [{desc}]: {e}")
            print(f"  [{desc:<35}] CRASH: {e}")

    status = TestStatus.PASS if len(violations) == 0 else TestStatus.FAIL
    details = (f"Tested {total} code-switching cases. "
               f"Violations: {len(violations)}, Crashes: {crash_count}")
    print(f"\n  Result: {status.value} — {details}")
    if violations:
        for v in violations[:10]:
            print(f"    {v}")

    return TestResult(
        name="Code-Switching Integrity",
        status=status,
        details=details,
        metrics={"total_cases": total, "violations": len(violations), "crashes": crash_count},
        violations=violations,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY 9: META-VOCAB ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════════

def test_meta_vocab_roundtrip(
    vocab_path: str,
    test_sentences: list[str],
    full_corpus_path: Optional[str] = None,
    target_count: int = 100_000,
    tiktoken_model: str = "o200k_base",
) -> TestResult:
    print("\n" + "=" * 80)
    print("BATTERY 9: META-VOCAB ROUND-TRIP (WWHOMetaEncoder)")
    print("=" * 80)

    if not _META_OK:
        return TestResult(
            name="Meta-Vocab Round-Trip",
            status=TestStatus.WARN,
            details="WWHOMetaEncoder not available (encoder.py import failed)",
        )

    try:
        meta = WWHOMetaEncoder(vocab_path, tiktoken_model=tiktoken_model)
        print(f"  Meta vocab: {meta._meta.total_size:,} IDs "
              f"(tiktoken={meta._tik.n_vocab:,} + SGPE={len(meta._meta._sgpe_raw):,})")
    except Exception as e:
        return TestResult(
            name="Meta-Vocab Round-Trip",
            status=TestStatus.FAIL,
            details=f"Failed to initialize WWHOMetaEncoder: {e}",
        )

    # Build sentence list — load full corpus if available
    sentences = list(test_sentences)
    if full_corpus_path and os.path.exists(full_corpus_path):
        print(f"  Loading full corpus from {full_corpus_path}...")
        import json
        with open(full_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "").strip()
                    if text:
                        sentences.append(text)
                except json.JSONDecodeError:
                    continue
                if target_count > 0 and len(sentences) >= target_count:
                    break

    # Cycle if we have fewer than target
    if target_count > 0:
        if len(sentences) < target_count:
            import math
            reps = math.ceil(target_count / len(sentences))
            sentences = (sentences * reps)[:target_count]
        else:
            sentences = sentences[:target_count]

    print(f"  Testing round-trip on {len(sentences):,} sentences...")

    failures = []
    total_tokens = 0
    unk_count = 0

    for text in tqdm(sentences, desc="  meta round-trip", unit=" sent"):
        try:
            ids = meta.encode(text)
            decoded = meta.decode(ids)
            total_tokens += len(ids)

            # Count UNKs
            unk_id = meta._meta.encode_sgpe_token("[UNK]", 1)
            unk_count += ids.count(unk_id)

            if decoded != text:
                has_unk = unk_id in ids if unk_id is not None else False
                if not has_unk:
                    failures.append({
                        "original": text[:80],
                        "decoded":  decoded[:80],
                        "token_count": len(ids),
                    })
        except Exception as e:
            failures.append({"original": text[:80], "error": str(e)})

    lossless_pct = 100 * (1 - len(failures) / len(sentences))
    avg_tokens = total_tokens / len(sentences) if sentences else 0
    unk_rate = unk_count / total_tokens * 100 if total_tokens else 0

    print(f"\n  Sentences:     {len(sentences):,}")
    print(f"  Round-trip failures: {len(failures):,} "
          f"({lossless_pct:.2f}% lossless)")
    print(f"  Avg tokens/sentence: {avg_tokens:.1f}")
    print(f"  UNK rate: {unk_rate:.2f}%")

    if failures[:3]:
        print("  Sample failures:")
        for f in failures[:3]:
            print(f"    orig:    {f.get('original','')!r}")
            print(f"    decoded: {f.get('decoded','')!r}")

    hard_failures = [f for f in failures if "error" in f]
    status = TestStatus.FAIL if hard_failures else (
        TestStatus.WARN if failures else TestStatus.PASS
    )
    details = (f"Tested {len(sentences):,} sentences. "
               f"Failures: {len(failures)}, Crashes: {len(hard_failures)}, "
               f"Lossless: {lossless_pct:.2f}%, UNK rate: {unk_rate:.2f}%")
    print(f"\n  Result: {status.value} — {details}")

    return TestResult(
        name="Meta-Vocab Round-Trip (WWHOMetaEncoder)",
        status=status,
        details=details,
        metrics={
            "sentences_tested": len(sentences),
            "failures": len(failures),
            "lossless_pct": round(lossless_pct, 2),
            "avg_tokens_per_sentence": round(avg_tokens, 1),
            "unk_rate_pct": round(unk_rate, 2),
        },
        violations=[str(f) for f in failures[:20]],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════



def test_glitched_tokens(enc_mixed, test_sentences: list[str]) -> TestResult:
    import collections
    from tqdm import tqdm
    from battle_core import time_limit, TestStatus, TestResult, ZWJ, HAL
    print("\n" + "=" * 80)
    print("BATTERY 2: GLITCHED TOKEN DETECTION ")
    print("=" * 80)

    print("  Counting token usage across mixed corpus...")
    token_usage = collections.Counter()
    for text in tqdm(test_sentences, desc="  scanning", unit=" sent"):
        ids = enc_mixed.encode(text)
        token_usage.update(ids)
        
    zero_usage = []
    near_zero = []
    glitched_tokens = []
    encode_errors = []

    sgpe_vocab = enc_mixed.vocab
    for token_str, raw_id in sgpe_vocab.items():
        uid = enc_mixed._meta._sgpe_offset.get(token_str)
        count = token_usage.get(uid, 0)
        
        if count == 0:
            zero_usage.append((token_str, uid))
        elif count < 3:
            near_zero.append((token_str, uid, count))
            
        # ZWJ/HAL glitched check
        if len(token_str) > 1 and (ZWJ in token_str or HAL in token_str):
            stripped = token_str.strip()
            if stripped == ZWJ or stripped == HAL:
                glitched_tokens.append(f"Bare Artifact: '{token_str}'")

    print(f"  Total unified vocab size: {enc_mixed.vocab_size:,} (SGPE component: {len(sgpe_vocab):,})")
    print(f"  Zero-usage SGPE tokens: {len(zero_usage):,}")
    print(f"  Near-zero (< 3) tokens: {len(near_zero):,}")
    
    status = TestStatus.PASS if len(zero_usage) < len(sgpe_vocab) * 0.5 else TestStatus.WARN
    if glitched_tokens:
        status = TestStatus.FAIL

    details = f"Zero: {len(zero_usage)}, Near-Zero: {len(near_zero)}, Glitched: {len(glitched_tokens)}"
    print(f"\n  Result: {status.value} — {details}")
    
    return TestResult(
        name="Glitched Token Detection",
        status=status,
        details=details,
        metrics={
            "zero_usage": len(zero_usage),
            "near_zero": len(near_zero),
            "glitched_tokens": len(glitched_tokens),
        },
        violations=glitched_tokens[:50],
    )


def test_stratified_benchmarking() -> TestResult:
    from battle_core import TestStatus, TestResult
    print("\n" + "=" * 80)
    print("BATTERY 3: FRONTIER BENCHMARKING")
    print("=" * 80)
    
    import stratified_benchmark
    
    try:
        stratified_benchmark.main()
        status = TestStatus.PASS
        details = "Stratified benchmark completed across all scripts."
    except Exception as e:
        status = TestStatus.FAIL
        details = f"Benchmark crashed: {e}"
        
    return TestResult(
        name="Frontier Benchmarking",
        status=status,
        details=details,
        metrics={},
    )


def test_boundary_edge_cases(enc_mixed) -> TestResult:
    from battle_core import TestStatus, TestResult, LEADING_SPACE_CHAR
    print("\n" + "=" * 80)
    print("BATTERY 5: BOUNDARY & LEADING SPACE EDGE-CASES ")
    print("=" * 80)

    violations = []
    def _check(label: str, text: str):
        try:
            ids = enc_mixed.encode(text)
            decoded = enc_mixed.decode(ids)
            meta_unk_id = enc_mixed._meta.sgpe_unk_id(enc_mixed._raw_unk_id)
            ok = decoded == text or meta_unk_id in ids
            icon = "✓" if ok else "✗"
            print(f"  [{icon}] [{label:<28}] {text!r} -> {decoded!r}")
            if not ok:
                violations.append(f"[{label}] Roundtrip mismatch: {text!r} -> {decoded!r}")
        except Exception as e:
            violations.append(f"[{label}] CRASH: {text!r} {str(e)[:50]}")
            print(f"  [CRASH] [{label}] {e}")

    # Sinhala boundaries
    _check("B01-Sinhala-leading-space",     " සිංහල")
    _check("B02-Sinhala-no-leading-space",  "සිංහල")
    _check("B03-Sinhala-trailing-punct",    "සිංහල.")
    _check("B04-Sinhala-multi-word",        "දරුවන් පාසලට")

    # Devanagari boundaries
    _check("D01-Devanagari-leading-space",  " हिंदी")
    _check("D02-Devanagari-no-leading",     "नमस्ते")
    _check("D03-Devanagari-trailing-danda", "नमस्ते।")
    _check("D04-Devanagari-multi-word",     "भारत देश")
    _check("D05-Devanagari-anusvara",       "संस्कृत")

    # Cross-script boundaries
    _check("F01-SinhalaEng",               "සිංහලදABC")
    _check("F02-DevanagariEng",            "हिंदीDEF")
    _check("F03-Sinhala-Devanagari",       "සිංහල हिंदी")
    _check("G01-Mixed-3-scripts",          " සිංහල123ABCहिंदी ")

    status = TestStatus.PASS if not violations else TestStatus.FAIL
    print(f"\n  Result: {status.value} — Violations: {len(violations)}")
    return TestResult("Boundary Edge-Cases", status, f"Violations: {len(violations)}", {}, violations)


def test_roundtrip_consistency(
    enc_mixed,
    test_sentences: list[str],
    full_corpus_path: Optional[str] = None,
    target_count: int = 1_000_000,
) -> TestResult:
    """Battery 4: round-trip encode→decode over the mixed-script corpus.
    Uses WWHOMetaEncoder; resolves unk_id from the unified meta-vocab so we
    never crash on `sgpe.unk_id` which only exists on the V1 SGPEEncoder.
    """
    import os, json
    from tqdm import tqdm
    from battle_core import TestStatus, TestResult, _count_words

    print("\n" + "=" * 80)
    print("BATTERY 4: ROUND-TRIP CONSISTENCY ")
    print("=" * 80)

    try:
        unk_id = enc_mixed._meta.sgpe_unk_id(enc_mixed._raw_unk_id)
    except Exception:
        unk_id = None

    sentences = list(test_sentences)
    if full_corpus_path and os.path.exists(full_corpus_path):
        print(f"  Loading full corpus from {full_corpus_path}...")
        with open(full_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line).get("text", "").strip()
                    if text:
                        sentences.append(text)
                except json.JSONDecodeError:
                    continue
                if target_count > 0 and len(sentences) >= target_count:
                    break

    if target_count > 0:
        if len(sentences) < target_count:
            original = list(sentences)
            while len(sentences) < target_count:
                sentences.append(original[len(sentences) % len(original)])
        sentences = sentences[:target_count]
    print(f"  Testing round-trip on {len(sentences):,} sentences...")

    mismatches, crashes = [], []
    unk_roundtrips = 0
    total_chars = total_tokens = total_words = 0

    batch_size = 10_000
    for batch_start in tqdm(range(0, len(sentences), batch_size),
                            desc="  round-trip", unit=f" ×{batch_size}"):
        for text in sentences[batch_start: batch_start + batch_size]:
            try:
                ids = enc_mixed.encode(text)
                decoded = enc_mixed.decode(ids)
                total_chars  += len(text)
                total_tokens += len(ids)
                total_words  += _count_words(text)

                has_unk = (unk_id is not None) and (unk_id in ids)
                if has_unk:
                    unk_roundtrips += 1
                if decoded != text and not has_unk:
                    mismatches.append({"original": text[:200], "decoded": decoded[:200]})
            except Exception as e:
                crashes.append({"original": text[:200], "error": str(e)[:100]})

    print(f"\n  Sentences tested:            {len(sentences):>12,}")
    print(f"  Total words:                 {total_words:>12,}")
    print(f"  Total characters tested:     {total_chars:>12,}")
    print(f"  Total tokens generated:      {total_tokens:>12,}")
    print(f"  Mismatches (non-UNK):        {len(mismatches):>12,}")
    print(f"  Mismatches (with UNK loss):  {unk_roundtrips:>12,}")
    print(f"  Crashes:                     {len(crashes):>12,}")

    if mismatches:
        print("\n  Sample mismatches:")
        for m in mismatches[:3]:
            print(f"    Original: {m['original']!r}")
            print(f"    Decoded:  {m['decoded']!r}")

    status = (
        TestStatus.PASS if (len(mismatches) == 0 and len(crashes) == 0)
        else TestStatus.FAIL
    )
    details = (
        f"Tested {len(sentences):,} sentences ({total_chars:,} chars). "
        f"Non-UNK mismatches: {len(mismatches)}, "
        f"UNK-caused losses: {unk_roundtrips}, Crashes: {len(crashes)}"
    )
    print(f"\n  Result: {status.value} — {details}")
    return TestResult(
        name="Round-Trip Consistency",
        status=status,
        details=details,
        metrics={
            "sentences_tested":    len(sentences),
            "total_chars":         total_chars,
            "total_tokens":        total_tokens,
            "non_unk_mismatches":  len(mismatches),
            "unk_roundtrips":      unk_roundtrips,
            "crashes":             len(crashes),
        },
        violations=[str(m) for m in (mismatches + crashes)[:20]],
    )


def test_zero_breakage_extended(enc_mixed) -> TestResult:
    from battle_core import test_zero_breakage_extended, TestStatus, TestResult
    
    print("\n" + "=" * 80)
    print("BATTERY 6: ZERO-BREAKAGE GUARANTEE")
    print("=" * 80)

    violations = []
    dfa = enc_mixed._devanagari_dfa
    
    print("  Testing Devanagari C + HAL + C pairs (implicit conjuncts)...")
    for c1 in list(sorted(DEV_CONSONANTS))[:15]:
        for c2 in list(sorted(DEV_CONSONANTS))[:15]:
            text = c1 + VIRAMA_DEV + c2
            tokens = dfa.tokenize(text, leading_space=False)
            if len(tokens) != 1:
                violations.append(f"  Devanagari C+HAL+C split: '{text}' → {tokens}")

    print("  Testing Devanagari C + vowel_sign...")
    for c in DEV_CONSONANTS:
        for vs in DEV_VOWEL_SIGNS:
            text = c + vs
            tokens = dfa.tokenize(text, leading_space=False)
            if len(tokens) != 1:
                violations.append(f"  Devanagari C+matra split: '{text}' → {tokens}")

    print("  Testing Devanagari C + HAL (terminal virama)...")
    for c in DEV_CONSONANTS:
        text = c + VIRAMA_DEV
        tokens = dfa.tokenize(text, leading_space=False)
        if len(tokens) != 1:
            violations.append(f"  Devanagari C+HAL split: '{text}' → {tokens}")

    print("  Testing Devanagari C + anusvara / visarga / chandrabindu...")
    for c in DEV_CONSONANTS:
        for mod in [ANUSVARA_DEV, VISARGA_DEV, CHANDRABINDU]:
            text = c + mod
            tokens = dfa.tokenize(text, leading_space=False)
            if len(tokens) != 1:
                violations.append(f"  Devanagari C+mod split: '{text}' → {tokens}")

    print("  Testing Devanagari C + vowel_sign + modifier...")
    for c in list(sorted(DEV_CONSONANTS))[:10]:
        for vs in list(sorted(DEV_VOWEL_SIGNS))[:5]:
            for mod in [ANUSVARA_DEV, VISARGA_DEV, CHANDRABINDU]:
                text = c + vs + mod
                tokens = dfa.tokenize(text, leading_space=False)
                if len(tokens) != 1:
                    violations.append(f"  Devanagari C+matra+mod split: '{text}' → {tokens}")

    status = TestStatus.PASS if not violations else TestStatus.FAIL
    print(f"\n  Result: {status.value} — Devanagari Violations: {len(violations)}")
    
    return TestResult(
        name="Zero-Breakage Guarantee",
        status=status,
        details=f"Violations: {len(violations)}",
        metrics={},
        violations=violations,
    )


def main():
    parser = argparse.ArgumentParser(description="WWHO Battle Test Suite")
    parser.add_argument("--vocab_file", type=str, default="output/vocab.json")
    parser.add_argument("--test_file", type=str, default="dataset/test.jsonl")
    parser.add_argument("--full_corpus", type=str, default=None)
    parser.add_argument("--report_output", type=str, default="output/battle_report.json")
    parser.add_argument("--tiktoken_model", type=str, default="o200k_base")
    parser.add_argument("--skip_roundtrip", action="store_true")
    parser.add_argument("--roundtrip_count", type=int, default=1_000_000)
    parser.add_argument("--meta_roundtrip_count", type=int, default=100_000)
    parser.add_argument("--frontier_samples", type=int, default=500)
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument(
        "--only", type=str, nargs="+",
        choices=[
            "complexity", "glitched", "frontier", "roundtrip",
            "boundary", "zerobreak",
            "devanagari", "codeswitching", "metavocab",
        ],
        help="Run only specific batteries",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               WWHO v1.0.0 BATTLE TEST SUITE                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if args.only:
        print(f"  [Partial Run] Selected: {args.only}")

    def should_run(name: str) -> bool:
        return args.only is None or name in args.only

    print("\n[INIT] Loading SGPE encoder (Sinhala-only, Layer 1+2)...")
    sgpe = SGPEEncoder(args.vocab_file)
    print(f"  Vocab size: {len(sgpe.vocab):,}")
    print(f"  Merges: {len(sgpe.merges):,}")

    meta_enc = None
    if _META_OK:
        try:
            meta_enc = WWHOMetaEncoder(args.vocab_file, tiktoken_model=args.tiktoken_model)
        except Exception as e:
            print(f"\n[WARN] WWHOMetaEncoder init failed: {e}")
            print("       Batteries 3/4/8 will fall back to SGPEEncoder (Sinhala-only — results will be skewed)")

    if meta_enc is not None:
        print(f"\n[INIT] WWHOMetaEncoder ready "
              f"(tiktoken={meta_enc.tiktoken_size:,} SGPE={len(meta_enc.vocab):,} "
              f"total={meta_enc.vocab_size:,} unified IDs)")
    else:
        print("[WARN] WWHOMetaEncoder unavailable — using SGPEEncoder fallback (results will be skewed)")
    enc_mixed = meta_enc if meta_enc is not None else sgpe

    test_sentences = load_test_data(args.test_file)

    report = BattleReport()
    report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # ── Original 6 Sinhala Batteries ────────────────────────────────────────
    if should_run("complexity"):
        report.add(test_linguistic_complexity(sgpe))

    if should_run("glitched"):
        report.add(test_glitched_tokens(enc_mixed, test_sentences))

    if should_run("frontier"):
        report.add(test_stratified_benchmarking())

    if should_run("roundtrip") and not args.skip_roundtrip:
        report.add(test_roundtrip_consistency(
            enc_mixed, test_sentences,
            full_corpus_path=args.full_corpus,
            target_count=args.roundtrip_count,
        ))

    if should_run("boundary"):
        report.add(test_boundary_edge_cases(enc_mixed))

    if should_run("zerobreak"):
        report.add(test_zero_breakage_extended(sgpe))
        report.add(test_zero_breakage_extended(enc_mixed))

    # ── New Multi-Script Batteries ───────────────────────────────────────────
    if should_run("devanagari"):
        dfa = build_devanagari_linguis_trie()
        report.add(test_devanagari_complexity(dfa))

    if should_run("codeswitching"):
        report.add(test_code_switching_integrity(enc_mixed))

    if should_run("metavocab"):
        report.add(test_meta_vocab_roundtrip(
            args.vocab_file, test_sentences,
            full_corpus_path=args.full_corpus,
            target_count=args.meta_roundtrip_count,
            tiktoken_model=args.tiktoken_model,
        ))

    print_final_report(report)
    save_report_json(report, args.report_output)
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
