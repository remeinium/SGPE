import argparse
import json
import os


def export_hf_tokenizer(vocab: dict[str, int], merges: list[tuple[str, str]],
                         special_tokens: list[str], output_path: str):
    added = []
    for st in special_tokens:
        added.append({
            "id": vocab[st],
            "content": st,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })

    hf_merges = [f"{a} {b}" for a, b in merges]

    cls_id = vocab.get("[CLS]", 2)
    sep_id = vocab.get("[SEP]", 3)

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ],
            "special_tokens": {
                "[CLS]": {"id": str(cls_id), "ids": [cls_id], "tokens": ["[CLS]"]},
                "[SEP]": {"id": str(sep_id), "ids": [sep_id], "tokens": ["[SEP]"]},
            },
        },
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": hf_merges,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  exported: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export SGPE vocab to HuggingFace tokenizer.json")
    parser.add_argument("--vocab", type=str, default="output/vocab.json")
    parser.add_argument("--out", type=str, default="output/tokenizer.json")
    args = parser.parse_args()

    with open(args.vocab, "r", encoding="utf-8") as f:
        data = json.load(f)

    export_hf_tokenizer(
        vocab=data["vocab"],
        merges=[tuple(m) for m in data["merges"]],
        special_tokens=data["special_tokens"],
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
