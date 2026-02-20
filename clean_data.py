import json
import unicodedata
import re
from tqdm import tqdm
import os

# Modifiers: HAL, Pili, and Anusvara/Visarga
MODIFIERS = r'\u0DCA\u0DCF-\u0DDF\u0DF2\u0DF3\u0D82\u0D83'
# Base characters: Consonants and Vowels
SINHALA_BASE = r'[\u0D85-\u0D96\u0D9A-\u0DC6]'

def clean_orphans(text):
    # 1. Standardize representation
    text = unicodedata.normalize('NFC', text)
    
    # 2. Iteratively remove orphans until no more changes 
    # (Handles chained orphans like [space][HAL][ZWJ])
    while True:
        orig = text
        # Remove ZWJs that are not following a HAL
        text = re.sub(r'(?<!\u0DCA)\u200D+', '', text)
        # Remove HAL/Pili/Post-modifiers that are not preceded by a base or another modifier
        text = re.sub(f'(?<!{SINHALA_BASE}|\u0DCA)[{MODIFIERS}]+', '', text)
        if text == orig:
            break
            
    # 3. Remove trailing HAL/ZWJ at the end of a word
    text = re.sub(r'\u0DCA\u200D?(?=\s|$)', '', text)
    
    return text

def main():
    files = [
        ('data/train.jsonl', 'data/train_clean.jsonl'),
        ('data/test.jsonl', 'data/test_clean.jsonl')
    ]
    
    for input_path, output_path in files:
        if not os.path.exists(input_path):
            print(f"Skipping {input_path} (not found)")
            continue

        print(f"Processing {input_path} -> {output_path}...")
        
        # Count lines
        with open(input_path, 'r', encoding='utf-8') as f:
            total = sum(1 for _ in f)

        count = 0
        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, total=total, unit=' lines', desc=os.path.basename(input_path)):
                    try:
                        obj = json.loads(line)
                        cleaned = clean_orphans(obj['text'])
                        if cleaned.strip():
                            obj['text'] = cleaned
                            f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                            count += 1
                    except Exception:
                        continue
        print(f"  Saved {count:,} lines to {output_path}")



if __name__ == "__main__":
    main()