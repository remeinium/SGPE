import json
import unicodedata
import re
from tqdm import tqdm
import os

SINHALA_BASE = r'\u0D85-\u0D96\u0D9A-\u0DC6'
SINHALA_MODIFIERS = r'\u0DCA\u0DCF-\u0DDF\u0DF2\u0DF3\u0D82\u0D83'

DEV_BASE = r'\u0904-\u0939\u0958-\u095F\u0960-\u0961\u0972-\u097F'
DEV_MODIFIERS = r'\u0900-\u0903\u093A-\u094F\u0951-\u0957\u0962-\u0963'
DEV_HALANT = r'\u094D'
DEV_NUKTA = r'\u093C'

def clean_orphans(text):
    text = unicodedata.normalize('NFC', text)
    while True:
        orig = text
        # ZWJ cleanup - keep only if preceded by Sinhala Al-pillam or Dev Halant
        text = re.sub(r'(?<!\u0DCA)(?<!\u094D)\u200D+', '', text)
        
        # Sinhala orphaned modifiers
        text = re.sub(f'(?<![{SINHALA_BASE}\u0DCA])[{SINHALA_MODIFIERS}]+', '', text)
        
        # Devanagari orphaned modifiers (can follow Base, Halant, or Nukta)
        text = re.sub(f'(?<![{DEV_BASE}{DEV_HALANT}{DEV_NUKTA}])[{DEV_MODIFIERS}{DEV_HALANT}{DEV_NUKTA}]+', '', text)
        
        if text == orig:
            break
    
    return text

def main():
    input_path = 'dataset/mixed_dataset_30m.jsonl'
    train_output = 'dataset/mixed_train.jsonl'
    test_output = 'dataset/mixed_test.jsonl'

    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (not found)")
        return

    print(f"Processing {input_path} -> 95% train, 5% test")
    
    total_lines = 30000000
    train_target = int(total_lines * 0.95)
    
    train_count = 0
    test_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(train_output, 'w', encoding='utf-8') as f_train, \
         open(test_output, 'w', encoding='utf-8') as f_test:
        
        for i, line in enumerate(tqdm(f_in, total=total_lines, unit=' lines', desc='Cleaning & Splitting')):
            try:
                obj = json.loads(line)
                cleaned = clean_orphans(obj['text'])
                if cleaned.strip():
                    obj['text'] = cleaned
                    out_line = json.dumps(obj, ensure_ascii=False) + '\n'
                    if i < train_target:
                        f_train.write(out_line)
                        train_count += 1
                    else:
                        f_test.write(out_line)
                        test_count += 1
            except Exception:
                continue

    print(f"\nDone! Saved {train_count:,} lines to {train_output}, {test_count:,} lines to {test_output}")

if __name__ == "__main__":
    main()