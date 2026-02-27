"""
SGPE — Dataset Acquisition
============================
Downloads the polyglots/MADLAD_CulturaX_cleaned Sinhala subset from
HuggingFace and splits it into train (95%) / test (5%) JSONL files.

Usage:
    python data/download_dataset.py [--n_sentences 10000]
    python data/download_dataset.py --n_sentences 0   # all ~10M
"""

import argparse
import json
import os
import random
import time

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download MADLAD/CulturaX Sinhala data")
    parser.add_argument("--n_sentences", type=int, default=10_000,
                        help="Number of sentences to download (0 = all)")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help="Fraction held out for test (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    limit = args.n_sentences if args.n_sentences > 0 else None
    label = f"first {limit:,}" if limit else "all"
    print(f"[1/3] Streaming dataset ({label} sentences)...")
    t0 = time.time()

    ds = load_dataset(
        "polyglots/MADLAD_CulturaX_cleaned",
        split="train",
        streaming=True,
    )

    sentences: list[str] = []
    pbar = tqdm(ds, total=limit, unit=" sent", desc="  downloading")

    for i, sample in enumerate(pbar):
        if limit and i >= limit:
            break
        text = sample["text"].strip()
        if text:
            sentences.append(text)

    pbar.close()
    elapsed = time.time() - t0
    rate = len(sentences) / elapsed
    print(f"  {len(sentences):,} sentences in {elapsed:.1f}s ({rate:,.0f} sent/s)")

    # shuffle and split
    print(f"\n[2/3] Shuffling (seed={args.seed}) and splitting "
          f"{1 - args.test_ratio:.0%} train / {args.test_ratio:.0%} test...")
    random.seed(args.seed)
    random.shuffle(sentences)

    split_idx = int(len(sentences) * (1 - args.test_ratio))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    del sentences  # free ~3 GB

    print(f"  Train: {len(train_sentences):,}  |  Test: {len(test_sentences):,}")

    # write JSONL
    print(f"\n[3/3] Writing to {out_dir}/...")

    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")

    for path, data, split_label in [
        (train_path, train_sentences, "train"),
        (test_path, test_sentences, "test"),
    ]:
        with open(path, "w", encoding="utf-8", buffering=1 << 20) as f:
            for text in tqdm(data, desc=f"  {split_label}", unit=" sent"):
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {split_label}: {path} ({len(data):,} lines, {size_mb:.2f} MB)")

    print(f"\nDone. Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
