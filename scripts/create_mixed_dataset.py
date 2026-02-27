import json
import os
import random
import re
from tqdm import tqdm

# Configuration
TARGET_SINHALA = 10_500_000
TARGET_HINDI = 13_500_000
TARGET_ENGLISH = 6_000_000

BASE_DIR = "/home/opc/SGPE"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(DATASET_DIR, "mixed_dataset_30m.jsonl")

# Sentence splitting regex
# Hindi punctuation: । (U+0964), ? (U+003F), ! (U+0021)
# English/Sinhala punctuation: . (U+002E), ? (U+003F), ! (U+0021)
RE_SENT_SPLIT = re.compile(r'([।\.?!])\s*')

def split_into_sentences(text):
    if not text:
        return []
    # Split by punctuation followed by optional whitespace
    parts = RE_SENT_SPLIT.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = (parts[i] + parts[i+1]).strip()
        if sent:
            sentences.append(sent)
    # Add trailing part if it exists
    if len(parts) % 2 == 1:
        last = parts[-1].strip()
        if last:
            sentences.append(last)
    return sentences

def stream_sentences_from_jsonl(filepath, limit):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                # Some files might have multiple lines in one 'text' field
                for subtext in text.split('\n'):
                    sents = split_into_sentences(subtext)
                    for s in sents:
                        yield s
                        count += 1
                        if limit and count >= limit:
                            return
            except Exception:
                continue

def main():
    final_dataset = []

    # 1. Process Sinhala (35% -> 10.5M)
    print("Processing Sinhala...")
    sinhala_sentences = []
    
    # Take ALL from sinhala_cm.jsonl
    print("  Loading sinhala_cm.jsonl...")
    for sent in stream_sentences_from_jsonl(os.path.join(DATASET_DIR, "sinhala_cm.jsonl"), None):
        sinhala_sentences.append(sent)
    print(f"  Got {len(sinhala_sentences):,} from sinhala_cm")

    # Take rest from test.jsonl and train.jsonl
    remaining_sinhala = TARGET_SINHALA - len(sinhala_sentences)
    if remaining_sinhala > 0:
        print(f"  Loading {remaining_sinhala:,} more from test.jsonl and train.jsonl...")
        for filepath in ["test.jsonl", "train.jsonl"]:
            if remaining_sinhala <= 0:
                break
            for sent in stream_sentences_from_jsonl(os.path.join(DATASET_DIR, filepath), remaining_sinhala):
                sinhala_sentences.append(sent)
                remaining_sinhala -= 1
    
    print(f"Total Sinhala collected: {len(sinhala_sentences):,}")
    final_dataset.extend(sinhala_sentences)
    del sinhala_sentences

    # 2. Process Hindi (45% -> 13.5M)
    print("Processing Hindi...")
    hindi_sentences = []
    
    # Take ALL from hindi_cm.jsonl
    print("  Loading hindi_cm.jsonl...")
    for sent in stream_sentences_from_jsonl(os.path.join(DATASET_DIR, "hindi_cm.jsonl"), None):
        hindi_sentences.append(sent)
    print(f"  Got {len(hindi_sentences):,} from hindi_cm")

    # Take rest from hi.jsonl
    remaining_hindi = TARGET_HINDI - len(hindi_sentences)
    if remaining_hindi > 0:
        print(f"  Loading {remaining_hindi:,} more from hi.jsonl...")
        for sent in stream_sentences_from_jsonl(os.path.join(DATASET_DIR, "hi.jsonl"), remaining_hindi):
            hindi_sentences.append(sent)
            remaining_hindi -= 1
            if len(hindi_sentences) % 1000000 == 0:
                print(f"    ...collected {len(hindi_sentences):,} Hindi sentences")

    print(f"Total Hindi collected: {len(hindi_sentences):,}")
    final_dataset.extend(hindi_sentences)
    del hindi_sentences

    # 3. Process English (20% -> 6M)
    print("Processing English...")
    english_sentences = []
    
    # Take from local en.jsonl
    print("  Loading en.jsonl...")
    for sent in stream_sentences_from_jsonl(os.path.join(DATASET_DIR, "en.jsonl"), TARGET_ENGLISH):
        english_sentences.append(sent)
        if len(english_sentences) % 1000000 == 0:
            print(f"    ...collected {len(english_sentences):,} English sentences")

    print(f"Total English collected: {len(english_sentences):,}")
    final_dataset.extend(english_sentences)
    del english_sentences

    # 4. Mixing and Saving
    print(f"Mixing {len(final_dataset):,} sentences...")
    random.shuffle(final_dataset)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sent in tqdm(final_dataset, desc="  Writing"):
            f.write(json.dumps({"text": sent}, ensure_ascii=False) + "\n")

    print("Success! Dataset created.")

if __name__ == "__main__":
    main()
