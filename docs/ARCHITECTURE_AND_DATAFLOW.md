# SGPE Architecture & Data Flow Reference

## Syllable-Aware Grapheme Pair Encoding — Internal Architecture Document

*Every statement in this document is derived by reading source code line by line.
Pseudocode mirrors actual control flow. Nothing is inferred from comments.*

Files covered: `linguis_trie.py`, `gpe_trainer.py`, `encoder.py`, `export.py`

---

## 1. The Problem This Architecture Solves

### 1.1 Sinhala Script Structure

A Sinhala consonant conjunct looks like this at the Unicode level:

```
Visible glyph:   ශ්‍රී   (one rendered syllable: "shri")

Unicode codepoints:
  U+0DC1  ශ   SINHALA LETTER TAALUJA SAYANNA  (consonant)
  U+0DCA  ්   SINHALA SIGN AL-LAKUNA           (HAL / virama)
  U+200D  ‍   ZERO WIDTH JOINER
  U+0DBB  ර   SINHALA LETTER RAYANNA           (consonant)
  U+0DD3  ී   SINHALA VOWEL SIGN DIGA IS       (pili / dependent vowel sign)

Total: 5 codepoints. Renders as: 1 syllable.
```

A purely frequency-based tokenizer sees a byte or codepoint sequence with no
knowledge of which positions form an indivisible glyph cluster. The HAL+ZWJ+consonant
sequence that produces a conjunct gets split wherever the BPE frequency statistics
happen to draw a boundary.

### 1.2 What "Split" Looks Like

```
BPE tokenisation of "ශ්‍රී ලංකාව" (OpenAI o200k_base):
  ['ශ්', 'රී', ' ලංක', 'ාව']

  ශ්   ← consonant + dangling virama with no following consonant in this token
  රී   ← ZWJ consumed into rendering engine; ර consonant + ී pili with no base
  ' ලංක' ← three-codepoint run that bisects the word mid-character
  'ාව'  ← pili ා separated from its base consonant ක

SGPE tokenisation of "ශ්‍රී ලංකාව":
  ['ශ්‍රී', ' ලංකාව']   ← 2 tokens for the entire phrase

  ශ්‍රී  = ශ + ් (HAL) + ‍ (ZWJ) + ර + ී  — one complete syllable, never split
  ' ලංකාව' = space prefix + full word merged by GPE
```

---

## 2. Layer 1 — LinguisTrie (`linguis_trie.py`)

### 2.1 Character Sets

LinguisTrie uses five module-level Python sets and two literal comparisons.
Every classification is an `in` test or `==` test — O(1) hash lookup.

```python
HAL = '\u0DCA'    # string literal; used as: ch == HAL
ZWJ = '\u200D'    # string literal; used as: ch == ZWJ

CONSONANTS   = {chr(c) for c in range(0x0D9A, 0x0DC7)}
# range() is exclusive of end → 0x0D9A to 0x0DC6 inclusive → 45 characters

VOWELS = {'\u0D85', '\u0D86', ..., '\u0D96'}
# 18 explicit codepoints. NOT a contiguous range — there are gaps.
# Cannot be replaced with a range() expression.

VOWEL_SIGNS = {'\u0DCF', ..., '\u0DDF', '\u0DF2', '\u0DF3'}
# 19 codepoints (17 from U+0DCF–U+0DDF + 2 from U+0DF2–U+0DF3)

POST_MODIFIERS = {'\u0D82', '\u0D83'}
# 2 codepoints: anusvara (ං) and visarga (ඃ)
```

**`_is_sinhala()` is defined but never called inside `tokenize()`.**
The six predicates `_is_consonant`, `_is_vowel`, `_is_vowel_sign`,
`_is_post_modifier`, `_is_hal`, `_is_zwj` are the only classification functions
used in the main loop.

**Bare ZWJ (U+200D) with no preceding HAL:** ZWJ is not in any of the five sets and
does not equal HAL or any other tested literal. It therefore falls through all
branches to the non-Sinhala passthrough and is emitted as a
single-character token. It is not dropped silently.

### 2.2 The `tokenize()` Main Loop

`LinguisTrie.tokenize(text, leading_space=False)` is a single `while pos < n` loop.
through `if / elif / else` branches determines how each character is handled.

```python
def tokenize(text, leading_space=False):
    tokens = []
    n = len(text)
    pos = 0
    pending_space = ""       # holds at most one " " to prepend to the next token

    while pos < n:
        ch = text[pos]

        # ── BRANCH 1: Whitespace ─────────────────────────────────────────
        # Only active when leading_space=True.
        # When leading_space=False, all whitespace falls through to Branch 5.
        if leading_space and ch in (' ', '\t', '\n', '\r'):
            ws_buffer = ""
            while pos < n and text[pos] in (' ', '\t', '\n', '\r'):
                ws_buffer += text[pos]
                pos += 1

            if ws_buffer.endswith(' '):
                # Emit every character except the trailing space as individual tokens
                for ws_char in ws_buffer[:-1]:
                    tokens.append(ws_char)
                # Save the trailing space to prepend to the next Sinhala token
                pending_space = " "
            else:
                # Buffer ends with \t or \n — emit ALL chars individually, no pending
                for ws_char in ws_buffer:
                    tokens.append(ws_char)
                pending_space = ""
            continue

        # ── BRANCH 2: Consonant-initiated syllable ───────────────────────
        if _is_consonant(ch):
            start = pos
            pos += 1

            # Inner conjunct loop: absorb (HAL [ZWJ] Consonant)* extensions.
            # Loop condition: HAL must be at current pos.
            while pos < n and _is_hal(text[pos]):

                if pos + 1 < n and _is_zwj(text[pos + 1]):
                    # Next two chars are ZWJ then a consonant → explicit conjunct
                    if pos + 2 < n and _is_consonant(text[pos + 2]):
                        pos += 3        # absorb HAL + ZWJ + C; continue loop
                        continue
                    else:
                        # HAL + ZWJ but no consonant follows — absorb both, stop
                        pos += 2
                        break

                elif pos + 1 < n and _is_consonant(text[pos + 1]):
                    pos += 2            # absorb HAL + C (implicit conjunct); continue
                    continue

                else:
                    # HAL with nothing valid after it (end-of-string, or next char
                    # is neither ZWJ nor a consonant). Leave HAL at pos; exit loop.
                    break

            # Post-cluster modifiers — absorbed after the inner loop exits.
            # if/elif: pili and terminal HAL are mutually exclusive.
            if pos < n and _is_vowel_sign(text[pos]):
                pos += 1               # absorb pili (dependent vowel sign)
            elif pos < n and _is_hal(text[pos]):
                pos += 1               # absorb terminal HAL (virama)

            # Separate if — post-modifier is independent of the pili/HAL choice.
            if pos < n and _is_post_modifier(text[pos]):
                pos += 1               # absorb anusvara or visarga

            tokens.append(pending_space + text[start:pos])
            pending_space = ""
            continue

        # ── BRANCH 3: Independent vowel ──────────────────────────────────
        if _is_vowel(ch):
            start = pos
            pos += 1
            if pos < n and _is_post_modifier(text[pos]):
                pos += 1               # absorb anusvara/visarga (e.g. අං)
            tokens.append(pending_space + text[start:pos])
            pending_space = ""
            continue

        # ── BRANCH 4: Orphan ─────────────────────────────────────────────
        # Condition covers three character types:
        #   _is_post_modifier(ch)  — bare anusvara / visarga
        #   _is_hal(ch)            — bare HAL with no preceding consonant
        #   _is_vowel_sign(ch)     — bare pili with no preceding consonant
        # Each is emitted as a single-character token.
        if _is_post_modifier(ch) or _is_hal(ch) or _is_vowel_sign(ch):
            tokens.append(pending_space + ch)
            pending_space = ""
            pos += 1
            continue

        # ── BRANCH 5: Non-Sinhala passthrough ────────────────────────────
        # Everything not caught above: Latin, digits, punctuation, bare ZWJ,
        # whitespace when leading_space=False, etc.
        if pending_space:
            tokens.append(pending_space + ch)
            pending_space = ""
        else:
            tokens.append(ch)
        pos += 1

    # Flush any pending space left at end of input (trailing whitespace with no
    # following token to attach to).
    if pending_space:
        tokens.append(pending_space)

    return tokens
```

### 2.3 Whitespace Handling in Detail

The whitespace branch consumes ALL contiguous whitespace into
`ws_buffer` in one inner loop, then decides what to emit based on whether the buffer
ends with a plain space character `' '`.

| ws_buffer content | endswith(' ')? | Action |
|---|---|---|
| `"  \t "` | Yes | Emit `" "`, `" "`, `"\t"` individually; `pending_space = " "` |
| `"\t\n "` | Yes | Emit `"\t"`, `"\n"` individually; `pending_space = " "` |
| `"\t\n"` | No | Emit `"\t"`, `"\n"` individually; `pending_space = ""` |
| `" "` (single space) | Yes | Emit nothing from loop; `pending_space = " "` |
| `"\t"` (single tab) | No | Emit `"\t"`; `pending_space = ""` |

Only a trailing `' '` (ordinary space) triggers `pending_space`. Tabs and newlines
are never attached to a following token.

**When `leading_space=False`:** The entire Branch 1 block is skipped. All whitespace
falls to Branch 5 and is emitted as individual passthrough characters with no
`pending_space` accumulation.

### 2.4 The Conjunct Extension Loop in Detail

The inner `while` loop has exactly four execution paths. For each
path, `pos` points at a HAL character when the path begins.

**Path 1 — Explicit ZWJ conjunct (HAL + ZWJ + C):**
- `text[pos+1]` is ZWJ and `text[pos+2]` is a consonant.
- `pos += 3`. Loop continues.
- All three codepoints are absorbed into the current syllable.

**Path 2 — Stray HAL+ZWJ (HAL + ZWJ + non-C or end-of-string):**
- `text[pos+1]` is ZWJ but `text[pos+2]` is NOT a consonant (or pos+2 >= n).
- `pos += 2`. `break`.
- Both HAL and ZWJ are absorbed into the current syllable token; extension stops.

**Path 3 — Implicit conjunct (HAL + C, no ZWJ):**
- `text[pos+1]` is NOT ZWJ but IS a consonant.
- `pos += 2`. Loop continues.
- Both HAL and the following consonant are absorbed; cluster extends.

**Path 4 — Terminal HAL (nothing valid follows):**
- `text[pos+1]` is neither ZWJ nor a consonant, OR `pos+1 >= n`.
- `break`. pos stays pointing at HAL.
- HAL is NOT absorbed by the inner loop. It is then absorbed by the
  `elif pos < n and _is_hal(text[pos]): pos += 1` check in the post-cluster section.

### 2.5 Character Handling Reference Table

| Character | Consonant branch | Post-cluster | Orphan branch | Passthrough |
|-----------|-----------------|-------------|--------------|-------------|
| Consonant (C) | Base of syllable; starts cluster | — | — | — |
| HAL — after cluster as ZWJ+C | Inner loop Path 1: absorbed | — | — | — |
| HAL — after cluster as C | Inner loop Path 3: absorbed | — | — | — |
| HAL — stray (HAL+ZWJ+non-C) | Inner loop Path 2: absorbed, break | — | — | — |
| HAL — terminal | Inner loop Path 4: break, left at pos | Absorbed as terminal virama | — | — |
| HAL — at syllable start (no base) | — | — | Orphan (Branch 4) | — |
| ZWJ — inside HAL+ZWJ+C | Inner loop Path 1: absorbed | — | — | — |
| ZWJ — bare (no preceding HAL) | — | — | — | Passthrough (Branch 5) |
| Pili after cluster | — | Absorbed (if/elif) | — | — |
| Pili at syllable start | — | — | Orphan (Branch 4) | — |
| Anusvara/visarga after cluster | — | Absorbed (if) | — | — |
| Anusvara/visarga at syllable start | — | — | Orphan (Branch 4) | — |
| Independent vowel | — | — | — (Branch 3 handles it) | — |
| Latin, digit, punctuation | — | — | — | Passthrough (Branch 5) |
| Whitespace (`leading_space=True`) | — | — | — | Branch 1 |
| Whitespace (`leading_space=False`) | — | — | — | Passthrough (Branch 5) |

---

## 3. Layer 2 — GPE Trainer (`gpe_trainer.py`)

### 3.1 Module-Level Constants

```python
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# List length = 5. Order determines IDs in the final vocab:
# [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

_SINHALA_RE = re.compile(r'[\u0D80-\u0DFF\u200D]')
# Used only in _is_boundary_token(). Not used in tokenize().
```

**`_is_boundary_token(token)`**:
```python
def _is_boundary_token(token):
    if not token:
        return True
    return not bool(_SINHALA_RE.search(token))
```
Returns `True` if the token contains no Sinhala Unicode characters and no ZWJ.
A leading-space Sinhala token like `" ක"` returns `False` because it contains
U+0D9A, which matches `[\u0D80-\u0DFF]`.

**`segment_into_words(syllables)`**: Splits the Layer 1 syllable list
into word groups. The exact logic from the code:

```python
for tok in syllables:
    if _is_boundary_token(tok):
        # Non-Sinhala token: flush current Sinhala run, emit this as its own group
        if current:
            words.append(current); current = []
        words.append([tok])
    else:
        # Sinhala token: check if it starts a new word
        if tok[0] in (' ', '\t', '\n', '\r') and current:
            # Leading-space prefix means this is the start of a new word
            words.append(current); current = []
        current.append(tok)
if current:
    words.append(current)
```

A Sinhala token whose first character is whitespace (i.e., it carries a
`pending_space` prefix from Layer 1) terminates the previous word, starts a new
word group, and is itself **included** as the first element of that new group
(the token is appended after the flush). Consecutive non-prefixed
Sinhala tokens remain in the same word group.

### 3.2 Parallel Pre-tokenisation

Training corpus is read from a `.jsonl` file where each line is `{"text": "..."}`.

```python
BATCH = 8192          # lines per pool.map call
chunksize = 256       # pool.map chunksize
num_workers = max(1, cpu_count() - 1)   # default
```

Each worker is initialised with `_init_worker()` which creates a single
`LinguisTrie()` instance stored in the process-global `_worker_tokenizer`.
Each task calls `_pretokenize_line(text)` → `_worker_tokenizer.tokenize(text, leading_space=True)`.

The main process then calls `segment_into_words()` and updates counters
**in the main process**, not in the workers. Workers only run
`tokenize()`.

### 3.3 Core Data Structures

**`word_counter`** (type: `Counter[tuple[str, ...]]`)
Maps each unique syllable-tuple (word type) to its total corpus frequency.
Built during phase 1 pre-tokenisation.

**`boundary_tokens`** (type: `set[str]`)
All non-Sinhala tokens seen during pre-tokenisation. They are added to the
`SymbolTable` first, in sorted order, before any syllable strings:
```python
for bt in sorted(boundary_tokens):
    self.symbols.get_or_add(bt)
```
This means boundary token IDs are assigned before syllable token IDs in
`SymbolTable`.

**`SymbolTable`**: Bidirectional mapping.
- `_str_to_id: dict[str, int]`
- `_id_to_str: list[str]`
- `get_or_add(token)` → assigns `new_id = len(_id_to_str)` before append.
- `add_merged(a_id, b_id)` → concatenates `_id_to_str[a_id] + _id_to_str[b_id]`,
  then calls `get_or_add` on the result.

**`word_types`** (type: `list[list[int]]`) and **`word_freqs`** (type: `list[int]`)
Two parallel lists produced by `build_word_types()`. `word_types[i]` is the
integer-ID representation of word type `i`. `word_freqs[i]` is its corpus count.
Rare syllables (frequency < `prune_freq`) are replaced with the sentinel integer
`-1` (defined as `UNK_SENTINEL = -1`). The string `"[UNK]"` is NOT
used internally — only the integer `-1`.

**`token_index`** (type: `dict[int, set[int]]`)
Maps symbol ID → set of `word_types` indices that contain that symbol.
Built by `build_token_index()`. Does not include UNK sentinel (`-1`) entries.

**`pair_counts`** (type: `dict[tuple[int, int], int]`)
Maps every adjacent symbol-ID pair to its weighted corpus frequency. UNK-adjacent
pairs are excluded (`if a < 0 or b < 0: continue`).

**`heap`** (type: `list[tuple[int, tuple[int, int]]]`)
Python `heapq` min-heap storing `(-freq, pair)` tuples — negated for max-heap
behaviour. Uses **lazy deletion**: stale entries are not removed on update. `_pop_best()`
pops entries and discards any whose stored frequency doesn't match
the current `pair_counts` value:
```python
actual = pair_counts.get(pair, 0)
if actual <= 0:
    continue            # entry is stale: pair was fully consumed
if actual != -neg_freq:
    self._heap_push(heap, pair, actual)   # re-push with current count
    continue
return pair, actual     # valid entry
```

### 3.4 Merge Budget Calculation

From `train()`:
```python
total_vocab_needed = self.target_vocab_size - len(SPECIAL_TOKENS)
# e.g. vocab_size=100_000 → total_vocab_needed = 99_995

if base_vocab >= total_vocab_needed:
    num_merges = 0
else:
    num_merges = total_vocab_needed - base_vocab
```
`base_vocab = len(self.symbols)` at the point after `build_word_types()` completes —
it includes boundary tokens and surviving syllable tokens, but NOT the 5 special
tokens (those are added later in `_save_output`).

### 3.5 Merge Loop

For each step:

1. `_pop_best(heap, pair_counts)` — returns the pair with the highest current
   frequency, skipping stale heap entries.
2. Stop if `best_pair is None or freq < self.min_freq`.
   Note: the condition is strict `<`. A pair with freq == `min_freq` still merges.
3. `self.symbols.add_merged(a_id, b_id)` — concatenates strings, assigns new ID.
4. `self.merges.append(best_pair)` — stored as `(int, int)` ID pair.
5. `merge_and_update(...)` — applies the merge and updates all affected data.
6. Log progress if `step <= 10 or step % 1000 == 0`.
7. Save checkpoint if `step % checkpoint_every == 0`.

### 3.6 `merge_and_update()` in Detail

Only word types containing BOTH `a` AND `b` are scanned — via set intersection:
```python
candidates = list(token_index.get(a, set()) & token_index.get(b, set()))
```

**Before the candidate loop**, the merged pair itself is immediately removed from
`pair_counts`:
```python
pair_counts.pop(pair, None)
```
This happens unconditionally, even if no candidates are found. The pair `(a, b)` is
gone from `pair_counts` from this point forward; it is not handled through
`dirty_pairs`.

For each candidate word type, the function walks `wt` left-to-right:

```
When wt[i] == a and wt[i+1] == b:
  left  = new_wt[-1]   if new_wt is non-empty
  right = wt[i+2]      if i+2 < len(wt)

  Decrement pair_counts[(left, a)]    if left >= 0
  Decrement pair_counts[(b, right)]   if right >= 0

  Append merged_id to new_wt

  Increment pair_counts[(new_wt[-2], merged_id)]  if new_wt[-2] >= 0
  Increment pair_counts[(merged_id, right)]        if right >= 0

  i += 2    (skip both a and b)

When wt[i] != a or wt[i+1] != b:
  Append wt[i] to new_wt
  i += 1
```

All changed pair counts are collected in `dirty_pairs`. After the word loop:
- `dirty_pairs` entries with count <= 0 are removed from `pair_counts`.
- `dirty_pairs` entries with count > 0 are pushed to the heap.
- `token_index` is updated: `merged_id` is added; `a` and `b` are removed from
  any word type where they no longer appear.
- Empty `token_index` entries for `a` and `b` are deleted.

The function returns `total_applied` — the sum of `word_freqs` for all word types
where the merge was applied (not the count of individual positions replaced).

### 3.7 Vocabulary Output Format (`vocab.json`)

Written by `_save_output()`:

```json
{
  "version": "sgpe_v1.0.0",
  "vocab_size": <int>,
  "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
  "num_merges": <int>,
  "prune_freq": <int>,
  "leading_space": true,
  "merges": [["syllable_a", "syllable_b"], ...],
  "vocab": {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "most_frequent_token": 5,
    ...
  }
}
```

**Vocab ID assignment order:**
1. The 5 special tokens get IDs 0–4 (in `SPECIAL_TOKENS` list order).
2. All remaining tokens ordered by `final_freq.most_common()` — highest corpus
   frequency first. **Note:** boundary tokens that had zero raw corpus frequency are
   assigned `freq = 1` before `most_common()` runs, so they appear
   here, not in step 3.
3. Any symbols in `SymbolTable` that still have no entry in `final_freq` — meaning
   they are neither boundary tokens nor appeared in the final BPE corpus — are
   appended last in `SymbolTable` insertion order.

**`"leading_space": true`** is hardcoded in `_save_output`. It is not
read from any config; it always writes `True` in the output.

**Merges are stored as string pairs** `[str, str]`, not integer pairs. The integer
representation used during training is not written to disk.

Also written to disk in the same call:
- `checkpoint_<N>.json` — final checkpoint
- `tokenizer.json` — HuggingFace format via `export_hf_tokenizer()`

### 3.8 Checkpoint Format

```json
{
  "step": <int>,
  "merges": [["a", "b"], ...],
  "elapsed_seconds": <float, rounded to 1 decimal>
}
```

On resume: the checkpoint is loaded, merges are replayed via
`replay_merges()` using a `dummy_heap = []` (the heap is not maintained during
replay — only `pair_counts`, `word_types`, and `token_index` are updated). After
replay, `pair_counts` is fully rebuilt from scratch.

### 3.9 Training Phase Summary

| Phase label | What it does | Key output |
|---|---|---|
| `[1/5]` Pre-tokenisation | Stream jsonl, run Layer 1 in parallel workers (batch=8192, chunk=256), `segment_into_words()` in main process | `word_counter`, `boundary_tokens` |
| `[2/5]` Build ID corpus | Compute syllable freqs; prune (sentinel `-1`); add boundary tokens to SymbolTable (sorted); convert word tuples to int lists | `word_types`, `word_freqs`, `SymbolTable` |
| `[3/5]` Index & count | Build `token_index`; count all adjacent pairs (skip `-1`) | `token_index`, `pair_counts` |
| `[4/5]` Merge loop | Pop best pair; create merged symbol; `merge_and_update`; checkpoint every N steps | `self.merges`, updated `word_types` |
| `[5/5]` Save | Count final token frequencies; assign IDs (specials first, then by freq); write `vocab.json`, `checkpoint_N.json`, `tokenizer.json` | Files on disk |

---

## 4. Inference — `encoder.py`

### 4.1 Initialisation (`SGPEEncoder.__init__`)

Reads `vocab.json` and constructs:

```python
self.vocab: dict[str, int]          # string → ID
self.merges: list[tuple[str, str]]  # ordered merge rules from data["merges"]
self.special_tokens: list[str]      # from data["special_tokens"]
self.tokenizer = LinguisTrie()      # fresh instance; no state
self.unk_id = self.vocab.get("[UNK]", 1)   # fallback if [UNK] missing
self.leading_space: bool = data.get("leading_space", False)

self._merge_priority: dict[tuple[str, str], int] = {
    (a, b): rank for rank, (a, b) in enumerate(self.merges)
}
# rank 0 = first merge learned = highest priority (most frequent)
# rank N = last merge learned = lowest priority
```

### 4.2 Full Tokenisation Flow (`tokenize()`)

```python
def tokenize(self, text):
    # Step 1: Layer 1 — deterministic syllable segmentation
    syllables = self.tokenizer.tokenize(text, leading_space=self.leading_space)

    # Step 2: Group syllables into word spans
    words = segment_into_words(syllables)

    result = []
    for word_tokens in words:

        # Step 3a: Boundary tokens bypass everything
        # Emitted as-is, even if not present in self.vocab.
        # (encode() will map them to unk_id later if missing from vocab.)
        if len(word_tokens) == 1 and _is_boundary_token(word_tokens[0]):
            result.append(word_tokens[0])
            continue

        # Step 3b: OOV syllable replacement — string level
        # Each syllable string is checked against self.vocab (the string→int dict).
        # Syllables in vocab: kept as-is.
        # Syllables not in vocab: replaced with the string "[UNK]".
        cleaned = [t if t in self.vocab else "[UNK]" for t in word_tokens]

        # Step 4: Apply merge rules within this word
        result.extend(self._apply_merges_to_word(cleaned))

    return result   # list of token strings, not yet IDs
```

### 4.3 Merge Application (`_apply_merges_to_word`)

```python
def _apply_merges_to_word(self, tokens):
    if len(tokens) <= 1:
        return tokens

    while True:
        best_rank = len(self.merges)   # sentinel larger than any real rank
        best_idx = -1

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = self._merge_priority.get(pair)   # None if pair not in rules
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_idx = i

        if best_idx == -1:
            break    # no applicable merge found anywhere in tokens

        # Apply the single best merge found in this pass
        merged = tokens[best_idx] + tokens[best_idx + 1]
        tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

    return tokens
```

Each iteration of the outer `while True` loop finds **the single highest-priority
applicable merge** across the entire token list and applies it. This repeats until no
merge rule matches any adjacent pair. Applying one merge can create new adjacent pairs
that match rules, so the outer loop may run many times.

### 4.4 `encode()` and `decode()`

**`encode(text)`**:
```python
def encode(self, text):
    tokens = self.tokenize(text)
    return [self.vocab.get(t, self.unk_id) for t in tokens]
```
Maps each string token to its integer ID. Tokens not found in `self.vocab` map to
`self.unk_id`. This includes boundary tokens that were passed through without OOV
checking in `tokenize()`.

**`decode(ids)`**:
```python
def decode(self, ids):
    id_to_token = {v: k for k, v in self.vocab.items()}
    return "".join(id_to_token.get(i, "") for i in ids)
```
Builds the reverse mapping on every call — not cached. IDs not found in the reverse
mapping produce an empty string `""` (not a placeholder or error character).

---

## 5. Export — `export.py`

`export_hf_tokenizer()` converts the SGPE vocabulary to a HuggingFace
`tokenizer.json` compatible with the `tokenizers` library's `BPE` backend.

```python
# Merge format conversion:
hf_merges = [f"{a} {b}" for a, b in merges]
# e.g. ["ශ්‍රී", " ලං"] → "ශ්‍රී  ලං"  (space-joined)

# CLS and SEP IDs come from vocab lookup with fallbacks:
cls_id = vocab.get("[CLS]", 2)
sep_id = vocab.get("[SEP]", 3)
```

The output `tokenizer.json` structure includes:
- `"version": "1.0"` — this is the HuggingFace tokenizer schema version, distinct
  from the `"sgpe_v1.0.0"` version string written to `vocab.json`.
- `"pre_tokenizer": null` — no pre-tokenizer is registered. The HuggingFace
  tokenizer will NOT run LinguisTrie automatically. To use SGPE correctly through
  the HuggingFace interface, a custom pre-tokenizer wrapping LinguisTrie must be
  registered separately, or `SGPEEncoder` must be used directly.
- `"model": {"type": "BPE", "byte_fallback": false, "fuse_unk": false, ...}`
- Post-processor: `TemplateProcessing` with `[CLS]...[SEP]` wrapping for single
  sequences and `[CLS]...[SEP]...[SEP]` for pairs.

---

## 6. End-to-End Data Flow: Traced Example

**Input:** `"ශ්‍රී ලංකාව"` with `leading_space=True`

**Codepoint breakdown:**

```
Index  Char   CP      Class
  0    ශ    U+0DC1   consonant (CONSONANTS set)
  1    ්    U+0DCA   HAL
  2    ‍    U+200D   ZWJ
  3    ර    U+0DBB   consonant
  4    ී    U+0DD3   pili (VOWEL_SIGNS)
  5    (sp) U+0020   whitespace
  6    ල    U+0DBD   consonant
  7    ං    U+0D82   post-modifier (POST_MODIFIERS: anusvara)
  8    ක    U+0D9A   consonant
  9    ා    U+0DCF   pili (VOWEL_SIGNS)
 10    ව    U+0DC0   consonant
```

**Step 1 — Layer 1 execution:**

```
pos=0: ch='ශ' → _is_consonant → True
  start=0, pos=1
  Inner loop:
    text[1]='්' → _is_hal → True → enter loop body
    text[2]='‍' → _is_zwj → True
    text[3]='ර' → _is_consonant → True   ✓ Path 1
    pos += 3 → pos=4; continue
    text[4]='ී' → _is_hal → False → exit loop (while condition fails)
  Post-cluster:
    text[4]='ී' → _is_vowel_sign → True → pos=5
    text[5]=' ' → _is_post_modifier → False
  Emit: "" + text[0:5] = "ශ්‍රී"
  pending_space = ""

pos=5: ch=' ' → leading_space=True, ch in whitespace → Branch 1
  ws_buffer = " "  (inner while: text[5]=' ', text[6]='ල'∉whitespace → stop)
  ws_buffer.endswith(' ') → True
  ws_buffer[:-1] = "" → emit nothing from loop
  pending_space = " "

pos=6: ch='ල' → _is_consonant → True
  start=6, pos=7
  Inner loop:
    text[7]='ං' → _is_hal → False → exit loop immediately
  Post-cluster:
    text[7]='ං' → _is_vowel_sign → False
    text[7]='ං' → _is_hal → False
    text[7]='ං' → _is_post_modifier → True → pos=8
  Emit: " " + text[6:8] = " ලං"
  pending_space = ""

pos=8: ch='ක' → _is_consonant → True
  start=8, pos=9
  Inner loop:
    text[9]='ා' → _is_hal → False → exit loop immediately
  Post-cluster:
    text[9]='ා' → _is_vowel_sign → True → pos=10
    text[10]='ව' → _is_post_modifier → False
  Emit: "" + text[8:10] = "කා"
  pending_space = ""

pos=10: ch='ව' → _is_consonant → True
  start=10, pos=11
  Inner loop:
    pos=11, n=11 → while condition (pos < n) fails → no iterations
  Post-cluster:
    pos=11 >= n=11 → all checks skipped
  Emit: "" + text[10:11] = "ව"
  pending_space = ""

Layer 1 output: ["ශ්‍රී", " ලං", "කා", "ව"]
```

**Step 2 — `segment_into_words()`:**

```
"ශ්‍රී"  → _is_boundary_token → False; tok[0]='ශ' not whitespace → current=["ශ්‍රී"]
" ලං"  → _is_boundary_token → False; tok[0]=' ' IS whitespace, current non-empty
          → flush ["ශ්‍රී"] as word 1; current=[" ලං"]
"කා"   → _is_boundary_token → False; tok[0]='ක' not whitespace → current=[" ලං","කා"]
"ව"    → _is_boundary_token → False; tok[0]='ව' → current=[" ලං","කා","ව"]
End   → flush [" ලං","කා","ව"] as word 2

words = [["ශ්‍රී"], [" ලං", "කා", "ව"]]
```

**Step 3 — `encoder.py` processing:**

```
Word 1: ["ශ්‍රී"]
  len==1 and _is_boundary_token("ශ්‍රී") → False → not the boundary bypass
  cleaned = ["ශ්‍රී"] if in vocab else ["[UNK]"]
  _apply_merges_to_word(["ශ්‍රී"]) → len==1 → return immediately
  result: ["ශ්‍රී"]  (or ["[UNK]"] if OOV)

Word 2: [" ලං", "කා", "ව"]
  Not boundary bypass.
  cleaned: check each against self.vocab; replace OOV with "[UNK]"
  _apply_merges_to_word([" ලං", "කා", "ව"]):
    Pass 1: scan pairs:
      (" ලං", "කා") → rank = self._merge_priority.get((" ලං","කා"))
      ("කා", "ව") → rank = self._merge_priority.get(("කා","ව"))
    If (" ලං","කා") has the lower rank (higher priority):
      merged = " ලංකා"; tokens = [" ලංකා", "ව"]
    Pass 2: scan pairs:
      (" ලංකා", "ව") → check priority
    If this pair is also in merge rules:
      merged = " ලංකාව"; tokens = [" ලංකාව"]
    Pass 3: no adjacent pairs → break
  result extended with [" ලංකාව"]

Final tokenize() output: ["ශ්‍රී", " ලංකාව"]
```

**Step 4 — `encode()`:**

```python
["ශ්‍රී", " ලංකාව"]
→ [self.vocab.get("ශ්‍රී", unk_id), self.vocab.get(" ලංකාව", unk_id)]
→ e.g. [4821, 1203]
```

---

## 7. Correctness Properties Grounded in Code

| Property | Code location enforcing it | Empirical result |
|----------|---------------------------|-----------------|
| HAL+ZWJ+C always absorbed as one unit | Inner loop Path 1: `pos += 3; continue` | Battery 6: 1,703 tests, 0 violations |
| HAL+ZWJ without C absorbed, not split | Inner loop Path 2: `pos += 2; break` | Battery 1: `bare_hal_zwj` passes |
| Terminal HAL absorbed into syllable | Post-cluster `elif _is_hal: pos += 1` | Battery 6, 0 violations |
| Pili cannot be a token start | Branch 4 (orphan) catches bare pili | Battery 6, 0 violations |
| Bare ZWJ not dropped, emitted as passthrough | Falls to Branch 5; emitted as single character | Battery 1: `bare_zwj` passes |
| UNK sentinel never enters pair counts | `count_all_pairs` excludes `if a < 0 or b < 0` | Structural |
| Boundary tokens bypass OOV replacement | `encode.py` boundary check before `cleaned` | Structural |
| Round-trip lossless (non-UNK) | Every input char emitted in exactly one token; `decode()` concatenates | Battery 4: 59.3M chars, 0 non-UNK mismatches |
| O(N) inference (Layer 1) | `pos` strictly monotone-increasing; each char examined at most once by inner loop | Measured: 32,899 words/s |
