import argparse
import json

from linguis_trie import LinguisTrie
from gpe_trainer import segment_into_words, _is_boundary_token


class SGPEEncoder:

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab: dict[str, int] = data["vocab"]
        self.merges: list[tuple[str, str]] = [tuple(m) for m in data["merges"]]
        self.special_tokens: list[str] = data["special_tokens"]
        self.tokenizer = LinguisTrie()
        self.unk_id = self.vocab.get("[UNK]", 1)
        self.leading_space: bool = data.get("leading_space", False)

        self._merge_priority: dict[tuple[str, str], int] = {
            (a, b): rank for rank, (a, b) in enumerate(self.merges)
        }

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def _apply_merges_to_word(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= 1:
            return tokens

        while True:
            best_rank = len(self.merges)
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return tokens

    def tokenize(self, text: str) -> list[str]:
        syllables = self.layer1_tokenize(text)
        words = segment_into_words(syllables)

        result: list[str] = []
        for word_tokens in words:
            if len(word_tokens) == 1 and _is_boundary_token(word_tokens[0]):
                result.append(word_tokens[0])
                continue

            cleaned = [t if t in self.vocab else "[UNK]" for t in word_tokens]
            result.extend(self._apply_merges_to_word(cleaned))

        return result

    def layer1_tokenize(self, text: str) -> list[str]:
        """Layer 1: Deterministic LinguisTrie pre-tokenization (Syllables)."""
        return self.tokenizer.tokenize(text, leading_space=self.leading_space)

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_token.get(i, "") for i in ids)


def main():
    parser = argparse.ArgumentParser(description="SGPE Encoder")
    parser.add_argument("--vocab", type=str, default="output/vocab.json")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    enc = SGPEEncoder(args.vocab)
    tokens = enc.tokenize(args.text)
    ids = enc.encode(args.text)
    print(f"tokens : {tokens}")
    print(f"ids    : {ids}")
    print(f"count  : {len(tokens)}")


if __name__ == "__main__":
    main()
