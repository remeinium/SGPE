import os
import sys
import json
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder import WWHOMetaEncoder
import tiktoken
import transformers
import re

def _count_words(text: str) -> int:
    return len([w for w in text.split() if w.strip()])

def get_dominant_script(text: str) -> str:
    sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', text))
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    m = max(sinhala_chars, devanagari_chars, latin_chars)
    if m == 0:
        return "Other"
    if m == sinhala_chars: return "Sinhala"
    if m == devanagari_chars: return "Hindi"
    return "English"

def main():
    console = Console()
    console.print("\n[bold cyan]Initializing Tokenizers...[/bold cyan]")
    
    tokenizers = {}
    
    visual_tokenizers = {}
    
    # WWHO
    wwho = WWHOMetaEncoder("output/vocab.json")
    tokenizers["SGPE"] = lambda t: len(wwho.encode(t))
    visual_tokenizers["SGPE"] = lambda t: [(tk, id) for tk, id in zip(wwho.tokenize(t), wwho.encode(t))]
    console.print("  ✓ SGPE")
    
    # OpenAI
    tik = tiktoken.get_encoding("o200k_base")
    tokenizers["OpenAI (o200k_base)"] = lambda t: len(tik.encode(t))
    visual_tokenizers["OpenAI (o200k_base)"] = lambda t: [(tik.decode([id]), id) for id in tik.encode(t)]
    console.print("  ✓ OpenAI")
    
    # Llama 4
    if os.getenv("HF_TOKEN"):
        try:
            llama = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", token=os.getenv("HF_TOKEN"), trust_remote_code=True)
            tokenizers["Llama 4 Scout"] = lambda t: len(llama.encode(t, add_special_tokens=False))
            visual_tokenizers["Llama 4 Scout"] = lambda t: [(llama.decode([id]), id) for id in llama.encode(t, add_special_tokens=False)]
            console.print("  ✓ Llama 4 Scout")
        except Exception as e:
            console.print(f"  ✗ Llama 4 skipped: {e}")
            
        try:
            ds = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", token=os.getenv("HF_TOKEN"), trust_remote_code=True)
            tokenizers["DeepSeek V3"] = lambda t: len(ds.encode(t, add_special_tokens=False))
            visual_tokenizers["DeepSeek V3"] = lambda t: [(ds.decode([id]), id) for id in ds.encode(t, add_special_tokens=False)]
            console.print("  ✓ DeepSeek V3")
        except Exception as e:
            console.print(f"  ✗ DeepSeek skipped: {e}")
            
            
    # --- VISUAL EXAMPLES ---
    console.print("\n[bold cyan]1. Tokenization Anatomy [/bold cyan]")
    test_words = [
        "ව්යාකරණය",         
        "ශ්‍රී ලංකාව",     
        "अंतर्राष्ट्रीय",     
        "कृत्रिम बुद्धिमत्ता", 
    ]
    
    for word in test_words:
        console.print(f"\n[bold yellow]'{word}':[/bold yellow]")
        for name, viz_func in visual_tokenizers.items():
            try:
                tokens = viz_func(word)
                tok_strs = [t[0] for t in tokens]
                count = len(tokens)
                console.print(f"  {name:<30} {str(tok_strs):<45} [dim]({count} tokens)[/dim]")
            except Exception as e:
                console.print(f"  {name:<30} [red]Error: {e}[/red]")
    
    sentences = []
    with open("dataset/mixed_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "").strip()
                if text: sentences.append(text)
            except: pass

    stats = {
        lang: {
            "words": 0,
            "chars": 0,
            "tok_counts": {name: 0 for name in tokenizers}
        } for lang in ["Sinhala", "Hindi", "English"]
    }

    console.print(f"\n[bold cyan]Evaluating {len(sentences):,} sentences...[/bold cyan]")
    for text in tqdm(sentences):
        lang = get_dominant_script(text)
        if lang not in stats: continue
        
        words = _count_words(text)
        if words == 0: continue
        
        stats[lang]["words"] += words
        stats[lang]["chars"] += len(text)
        
        for name, func in tokenizers.items():
            try:
                stats[lang]["tok_counts"][name] += func(text)
            except:
                pass

    for lang in ["Sinhala", "Hindi", "English"]:
        if stats[lang]["words"] == 0: continue
        
        print(f"\n====== {lang} Results ======")
        print(f"{'Tokenizer':<20} | {'Tokens':>12} | {'TWR':>7} | {'Chr/Tok':>7} | {'% Reduction':>12}")
        print("-" * 70)
        
        sgpe_twr = stats[lang]["tok_counts"]["SGPE"] / stats[lang]["words"]
        
        for name in tokenizers:
            toks = stats[lang]["tok_counts"][name]
            twr = toks / stats[lang]["words"]
            cpt = stats[lang]["chars"] / toks if toks > 0 else 0
            
            if name == "SGPE":
                red = "-"
            else:
                red_val = ((toks - stats[lang]["tok_counts"]["SGPE"]) / toks) * 100
                red = f"{red_val:.1f}%"
                
            print(f"{name:<20} | {toks:>12,} | {twr:>7.3f} | {cpt:>7.2f} | {red:>12}")
            
    # --- OVERALL SUMMARY ---
    total_words = sum(stats[lang]["words"] for lang in ["Sinhala", "Hindi", "English"])
    if total_words > 0:
        total_chars = sum(stats[lang]["chars"] for lang in ["Sinhala", "Hindi", "English"])
        
        print("\n" + "=" * 25 + " OVERALL Results " + "=" * 25)
        print(f"{'Tokenizer':<20} | {'Tokens':>12} | {'TWR':>7} | {'Chr/Tok':>7} | {'% Reduction':>12}")
        print("-" * 70)
        
        overall_sgpe_toks = sum(stats[lang]["tok_counts"]["SGPE"] for lang in ["Sinhala", "Hindi", "English"])
        
        for name in tokenizers:
            overall_toks = sum(stats[lang]["tok_counts"].get(name, 0) for lang in ["Sinhala", "Hindi", "English"])
            
            twr = overall_toks / total_words if total_words > 0 else 0
            cpt = total_chars / overall_toks if overall_toks > 0 else 0
            
            if name == "SGPE":
                red = "-"
            else:
                red_val = ((overall_toks - overall_sgpe_toks) / overall_toks) * 100 if overall_toks > 0 else 0
                red = f"{red_val:.1f}%"
                
            print(f"{name:<20} | {overall_toks:>12,} | {twr:>7.3f} | {cpt:>7.2f} | {red:>12}")

if __name__ == "__main__":
    main()
