# Syllable is the Token

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
   Enforces linguistic integrity by construction: a deterministic finite state machine segments raw Unicode into well-formed syllables with a formal zero-breakage guarantee.  
3. **Layer 3 (How Often): SGPE & Meta-Vocabulary**  
   Performs statistical pair merging exclusively over this linguistically sound stream, safely projecting the resulting tokens into a unified, mathematically offset ID space.

Sinhala and Devanagari serve as the high-complexity proofs-of-concept. The same architecture generalizes directly to Tamil, Khmer, Myanmar, and the broader Abugida family.

---

## Multi-Script Stratified Benchmarks (122.2M Characters)

We evaluated WWHO against frontier models across a 1.5 million sentence code-switched corpus containing Sinhala, Hindi (Devanagari), and English. 

### 1. Sinhala Efficiency
| Tokenizer              | TWR ↓   | Tokens      | Chars/Token ↑ | Reduction vs WWHO |
|------------------------|---------|-------------|---------------|-------------------|
| **WWHO (ours)**        | **1.276** | **6.66 M**  | **4.83**      | —                 |
| OpenAI o200k_base      | 3.324   | 17.36 M     | 1.85          | 61.6 %            |
| Llama 4 Scout          | 3.476   | 18.15 M     | 1.77          | 63.3 %            |
| DeepSeek V3            | 5.581   | 29.15 M     | 1.10          | 77.1 %            |

### 2. Hindi (Devanagari) Efficiency
| Tokenizer              | TWR ↓   | Tokens      | Chars/Token ↑ | Reduction vs WWHO |
|------------------------|---------|-------------|---------------|-------------------|
| **WWHO (ours)**        | **1.181** | **13.43 M** | **4.29**      | —                 |
| OpenAI o200k_base      | 1.617   | 18.39 M     | 3.13          | 27.0 %            |
| Llama 4 Scout          | 1.720   | 19.56 M     | 2.94          | 31.3 %            |
| DeepSeek V3            | 2.786   | 31.68 M     | 1.82          | 57.6 %            |

### 3. Native English / Code Retention
| Tokenizer              | TWR ↓   | Tokens      | Chars/Token ↑ | Reduction vs WWHO |
|------------------------|---------|-------------|---------------|-------------------|
| **WWHO (ours)**        | **1.330** | **7.24 M**  | **4.46**      | —                 |
| OpenAI o200k_base      | 1.364   | 7.42 M      | 4.35          | 2.4 %             |

*(Note: Because WWHO routes Latin text directly to the native Tiktoken sequence, English performance is mathematically identical. The minor delta in total tokens emerges solely from boundary crossing mechanics.)*

### 4. Overall Pipeline (Mixed-Script)
| Tokenizer              | TWR ↓   | Tokens      | Chars/Token ↑ | Reduction vs WWHO |
|------------------------|---------|-------------|---------------|-------------------|
| **WWHO (ours)**        | **1.241** | **27.33 M** | **4.47**      | —                 |
| OpenAI o200k_base      | 1.959   | 43.17 M     | 2.83          | 36.7 %            |
| Llama 4 Scout          | 2.053   | 45.23 M     | 2.70          | 39.6 %            |
| DeepSeek V3            | 3.119   | 68.73 M     | 1.78          | 60.2 %            |

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
- **Research Paper**: “The Syllable is the Token: Breaking the Token Tax with SGPE” (Remeinium Research, February 2026)  
- **Pre-trained Tokenizer**: [Hugging Face](https://huggingface.co/remeinium/WWHO)  
- **Cleaned Training Corpus**: [Hugging Face](https://huggingface.co/datasets/remeinium/WWHO_Cleaned_30m)  
- **Full Code & Evaluation Harness**: [GitHub](https://github.com/remeinium/WWHO)  


--- -->

## License

Apache License 2.0 — see [LICENSE](LICENSE).

**Remeinium Research | Remeinium AI | Intelligence for a Greater Tomorrow**  

---