# Syllable is the Token: SGPE - Syllable-Aware Grapheme Pair Encoding

**Remeinium Research**  
[remeinium.com](https://remeinium.com) | [Paper](https://arxiv.org/abs/...) | [Tokenizer](https://huggingface.co/remeinium/SGPE-Tokenizer) | [Dataset](https://huggingface.co/datasets/remeinium/SGPE_Cleaned)

---

## The Next Architectural Primitive in Tokenization

Large language models remain linguistically blind to Abugida scripts. Byte-Pair Encoding and its descendants routinely shatter complex conjuncts — atomic multi-codepoint grapheme clusters that constitute the fundamental phonetic units of Indic and Southeast Asian writing systems — into meaningless sub-character fragments. The result is degraded reasoning, inflated inference costs, and a systemic “Token Tax” that disproportionately burdens more than one billion speakers.

**SGPE introduces the clean separation of concerns the field has been missing.**

**Layer 1 (LinguisTrie)** enforces linguistic integrity by construction: a deterministic $O(N)$ finite automaton segments raw Unicode into well-formed syllables with a formal zero-breakage guarantee.  
**Layer 2 (GPE)** then performs statistical pair merging exclusively over this linguistically sound stream, inheriting the guarantee by design.

Sinhala serves as the high-complexity proof-of-concept. The same architecture generalizes directly to Devanagari, Tamil, Khmer, Myanmar, and the broader Abugida family through script-specific character-class mappings and conjunct rules.

---

## Results on 59.3 Million Characters

| Tokenizer              | TWR ↓   | Tokens      | Chars/Token ↑ | Reduction vs SGPE |
|------------------------|---------|-------------|---------------|-------------------|
| **SGPE (ours)**        | **1.438** | **13.26 M** | **4.48**      | —                 |
| OpenAI o200k_base      | 3.515   | 32.39 M     | 1.83          | 59.1 %            |
| Llama 4 Scout          | 3.673   | 33.85 M     | 1.75          | 60.8 %            |
| DeepSeek V3            | 5.965   | 54.98 M     | 1.08          | 75.8 %            |

- **Zero-Breakage Guarantee** validated on 1,703 exhaustive conjunct formations (0 violations).  
- Full-corpus round-trip reconstruction: 0 non-UNK mismatches.  
- UNK rate: 0.46 % (rare compounds only; no structural errors).

SGPE reclaims more than half the context window for Abugida text while preserving perfect orthographic and semantic integrity.

---

## Architecture

SGPE is deliberately bimodal:

1. **LinguisTrie (Layer 1)**  
   Deterministic finite automaton operating in a single left-to-right pass with constant-time transitions and $O(1)$ auxiliary space. Guarantees that no conjunct, pili, virama, or ZWJ sequence is ever fragmented.

2. **Grapheme Pair Encoding (Layer 2)**  
   Standard BPE performed exclusively on the atomic syllable stream, with three critical constraints:
   - Syllabic initialization (base vocabulary consists of linguistically valid units)  
   - Boundary-aware scoping (merges restricted to within-word spans)  
   - Frequency pruning (rare syllables mapped to [UNK] sentinel before merging)

The decoupling is the core scientific contribution: linguistic correctness is enforced by construction rather than hoped for statistically.

---

## Quick Start with Hugging Face

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("remeinium/SGPE")
text = "ආයුබෝවන් ශ්‍රී ලංකා"

tokens = tokenizer.tokenize(text)
# ['ආයුබෝවන්', ' ශ්‍රී', ' ලංකා']
print(tokenizer.encode(text))
```

---

## Resources

- **Research Paper**: “The Syllable is the Token: Breaking the Token Tax with SGPE” (Remeinium Research, February 2026)  
- **Pre-trained Tokenizer**: [Hugging Face](https://huggingface.co/remeinium/SGPE-Tokenizer)  
- **Cleaned Training Corpus**: [Hugging Face](https://huggingface.co/datasets/remeinium/SGPE_Cleaned)  
- **Full Code & Evaluation Harness**: [GitHub](https://github.com/remeinium/SGPE)  

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

**Remeinium Research | Remeinium AI | Intelligence for a Greater Tomorrow**  

---