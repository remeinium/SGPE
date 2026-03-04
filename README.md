# Separate before you Compress

<!-- **Remeinium Research**  
[remeinium.com](https://remeinium.com) | [Paper](https://arxiv.org/abs/...) | [Tokenizer](https://huggingface.co/remeinium/WWHO) | [Dataset](https://huggingface.co/datasets/remeinium/WWHO_Cleaned_30m) 

--- -->

## The Next Architectural Primitive in Tokenization

Large language models remain linguistically blind to Abugida scripts. Byte-Pair Encoding and its descendants routinely shatter complex conjuncts — atomic multi-codepoint grapheme clusters that constitute the fundamental phonetic units of Indic and Southeast Asian writing systems — into meaningless sub-character fragments. The result is degraded reasoning, inflated inference costs, and a systemic “Token Tax” that disproportionately burdens more than one billion speakers.

**WWHO (Where-What-How Often) introduces the clean separation of concerns the field has been missing.**

By decoupling linguistic structural constraints from statistical compression, WWHO builds a unified meta-vocabulary space:

1. **Layer 1 (Where): Code-Switching Router**  
   A linear $O(N)$ block scanner that evaluates characters in $O(1)$ time to inherently identify script boundaries, routing Latin text to proven frontier tokenizers (like `o200k_base`) while sending Abugida text for specialized processing.
2. **Layer 2 (What): LinguisTrie**  
   Enforces linguistic integrity by construction: a DFA based syllabifier segments raw Unicode into well-formed syllables with a formal zero-breakage guarantee.  
3. **Layer 3 (How Often): SGPE & Meta-Vocabulary**  
   Performs statistical pair merging exclusively over this linguistically sound stream, safely projecting the resulting tokens into a unified, mathematically offset ID space.

Sinhala and Devanagari serve as the high-complexity proofs-of-concept. The same architecture generalizes directly to Tamil, Khmer, Myanmar, and the broader Abugida family.

---

## Multi-Script Stratified Benchmarks (122.2M Characters)

We evaluated WWHO against frontier models across a 1.5 million sentence code-switched corpus containing Sinhala, Hindi (Devanagari), and English. 

### 1. Sinhala Efficiency
|Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
|----------------------------------------------------------------------
|**SGPE(WWHO)         |    6,654,288 |   1.274 |    4.83 |            -**
|OpenAI (o200k_base)  |   17,360,196 |   3.324 |    1.85 |        61.7%
|Llama 4 Scout        |   18,157,707 |   3.476 |    1.77 |        63.4%
|DeepSeek V3          |   29,152,698 |   5.581 |    1.10 |        77.2%

### 2. Hindi (Devanagari) Efficiency
|Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
|----------------------------------------------------------------------
|**SGPE(WWHO)         |   13,433,554 |   1.181 |    4.29 |            -**
|OpenAI (o200k_base)  |   18,394,075 |   1.617 |    3.13 |        27.0%
|Llama 4 Scout        |   19,566,121 |   1.720 |    2.94 |        31.3%
|DeepSeek V3          |   31,682,218 |   2.786 |    1.82 |        57.6%

### 3. English
|Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
|----------------------------------------------------------------------
|**SGPE(WWHO)         |    7,240,147 |   1.330 |    4.46 |            -**
|OpenAI (o200k_base)  |    7,420,527 |   1.364 |    4.35 |         2.4%
|Llama 4 Scout        |    7,512,843 |   1.381 |    4.30 |         3.6%
|DeepSeek V3          |    7,904,670 |   1.453 |    4.09 |         8.4%

*(Note: Because WWHO routes Latin text directly to the native Tiktoken sequence, English performance is mathematically identical. The minor delta in total tokens emerges solely from boundary crossing mechanics.)*

### 4. Overall (Mixed-Script)
|Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
|----------------------------------------------------------------------
|**SGPE(WWHO)         |   27,327,989 |   1.240 |    4.47 |            -**
|OpenAI (o200k_base)  |   43,174,798 |   1.959 |    2.83 |        36.7%
|Llama 4 Scout        |   45,236,671 |   2.053 |    2.70 |        39.6%
|DeepSeek V3          |   68,739,586 |   3.119 |    1.78 |        60.2%

- **Zero-Breakage Guarantee**: Validated through exhaustive testing permutations across all supported Abugida scripts (0 violations).  
- **Full-corpus reconstruction**: 1.5M code-switched sentences encoded and decoded with 0 non-UNK mismatches.  
- **UNK rate**: 0.08 % (restricted strictly to rare compounds without violating structural boundaries).

WWHO radically compresses the context window for Abugida text, effectively ending the Token Tax without penalizing existing state-of-the-art programming and reasoning capabilities.

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

<!-- 
- **Research Paper**: “The Syllable is the Token: Breaking the Token Tax with SGPE” (Remeinium Research, February 2026)   -->
- **Pre-trained Tokenizer**: [Hugging Face](https://huggingface.co/remeinium/WWHO)  
- **Cleaned Training Corpus**: [Hugging Face](https://huggingface.co/datasets/remeinium/WWHO_30m)  
- **Full Code & Evaluation Harness**: [GitHub](https://github.com/remeinium/WWHO)  


---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

**Remeinium Research | Remeinium AI | Intelligence for a Greater Tomorrow**  

---