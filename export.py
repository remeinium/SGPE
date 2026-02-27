"""
WWHO — SGPE Export
"""

import argparse
import json
import os

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False


def export_hf_tokenizer(
    vocab: dict[str, int],
    merges: list,
    special_tokens: list[str],
    output_path: str,
    script_mode: str = "mixed",
):
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
        "wwho_version": "1.0.0",
        "script_mode": script_mode,
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
    print(f"  [SGPE HF]  exported: {output_path} ({size_mb:.2f} MB)")


def export_meta_tokenizer(
    sgpe_vocab: dict[str, int],
    sgpe_merges: list,
    special_tokens: list[str],
    output_dir: str,
    tiktoken_model: str = "o200k_base",
    script_mode: str = "mixed",
):
    if not _TIKTOKEN_OK:
        print("  [META] tiktoken not available, skipping meta export.")
        return

    tik = tiktoken.get_encoding(tiktoken_model)
    tik_size = tik.n_vocab
    sgpe_size = len(sgpe_vocab)

    offset_vocab: dict[str, int] = {}
    for tok_str, raw_id in sgpe_vocab.items():
        offset_vocab[tok_str] = raw_id + tik_size

    hf_merges_offset = [f"{a} {b}" for a, b in sgpe_merges]

    added = []
    for st in special_tokens:
        if st in offset_vocab:
            added.append({
                "id": offset_vocab[st],
                "content": st,
                "single_word": False,
                "lstrip": False, "rstrip": False,
                "normalized": False, "special": True,
            })

    meta_hf = {
        "version": "1.0",
        "sgpe_version": "2.0.0",
        "tiktoken_model": tiktoken_model,
        "tiktoken_vocab_size": tik_size,
        "sgpe_vocab_size": sgpe_size,
        "total_vocab_size": tik_size + sgpe_size,
        "sgpe_id_offset": tik_size,
        "script_mode": script_mode,
        "note": (
            "IDs [0, tiktoken_vocab_size) are tiktoken tokens. "
            "IDs [tiktoken_vocab_size, total_vocab_size) are SGPE tokens "
            "(SGPE raw_id + tiktoken_vocab_size)."
        ),
        "added_tokens": added,
        "model": {
            "type": "BPE",
            "unk_token": "[UNK]",
            "vocab": offset_vocab,
            "merges": hf_merges_offset,
        },
    }

    meta_path = os.path.join(output_dir, "tokenizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_hf, f, ensure_ascii=False, indent=2)
    meta_mb = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"  [META HF]  exported: {meta_path} ({meta_mb:.2f} MB)")
    print(f"             vocab = tiktoken({tik_size:,}) + SGPE({sgpe_size:,}) "
          f"= {tik_size + sgpe_size:,} total IDs")

    cfg_path = os.path.join(output_dir, "meta_config.json")
    meta_config = {
        "tiktoken_model": tiktoken_model,
        "tiktoken_vocab_size": tik_size,
        "sgpe_vocab_size": sgpe_size,
        "sgpe_id_offset": tik_size,
        "script_mode": script_mode,
        "sgpe_vocab_path": "vocab.json",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(meta_config, f, indent=2)
    print(f"  [META CFG] exported: {cfg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WWHO — Export SGPE vocab to HF tokenizer format"
    )
    parser.add_argument("--vocab", type=str, default="output/vocab.json")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: same dir as vocab)")
    parser.add_argument("--tiktoken_model", type=str, default="o200k_base",
                        help="Tiktoken model for meta-vocab offset calculation")
    parser.add_argument("--no_meta", action="store_true",
                        help="Skip meta-vocab export (export raw SGPE only)")
    args = parser.parse_args()

    if not os.path.exists(args.vocab):
        print(f"Error: vocab file not found: {args.vocab}")
        return

    with open(args.vocab, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    merges = [tuple(m) for m in data["merges"]]
    special_tokens = data["special_tokens"]
    script_mode = data.get("script_mode", "mixed")

    out_dir = args.out_dir or os.path.dirname(args.vocab) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. Raw SGPE HF tokenizer
    hf_path = os.path.join(out_dir, "tokenizer.json")
    export_hf_tokenizer(vocab, merges, special_tokens, hf_path,
                           script_mode=script_mode)

    # 2. Meta-vocab tokenizer + config (for unified ID space)
    if not args.no_meta:
        export_meta_tokenizer(
            vocab, merges, special_tokens, out_dir,
            tiktoken_model=args.tiktoken_model,
            script_mode=script_mode,
        )


if __name__ == "__main__":
    main()
