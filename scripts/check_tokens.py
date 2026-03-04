import os
import sys
from encoder import WWHOMetaEncoder
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

def display_tokens(text, enc, console):
    ids = enc.encode(text)
    tokens = [enc.decode([i]) for i in ids]
    char_count = len(text)
    token_count = len(ids)
    
    table = Table(title=f"Tokenization Result", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    
    table.add_row("Input Text", f"'{text}'")
    table.add_row("Tokens", str(tokens))
    table.add_row("Token Count", str(token_count))
    table.add_row("Character Count", str(char_count))
    
    console.print(table)
    console.print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description="SGPE Tokenizer Checker")
    parser.add_argument("words", nargs="*", help="Words or sentences to tokenize")
    parser.add_argument("--file", "-f", help="File containing a word list (one per line)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive mode")
    parser.add_argument("--vocab", default="output/vocab.json", help="Path to vocab.json")
    args = parser.parse_args()

    console = Console()
    
    if not os.path.exists(args.vocab):
        console.print(f"[bold red]Error:[/bold red] Vocabulary file '{args.vocab}' not found.")
        return

    console.print("[yellow]Loading SGPE Tokenizer...[/yellow]")
    try:
        enc = WWHOMetaEncoder(args.vocab)
        console.print("[green]Tokenizer loaded successfully.[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to load encoder:[/bold red] {e}")
        return

    test_string = "ඔයා 1 special अद्भुत" #"AGI (General කෘත්රිම बुद्धिमत्ता) Ultimate लक्ष्य එක සපුරා ගැනීමට නම්, to anyone from ඕනෑම භාෂාවකින් समान गुणवत्ता සහ depth of knowledge සහිතව respond කිරීමට හැකි වන පරිදි LLMs प्रशिक्षित කළ යුතුයි....!!!!"
    console.print(f"[bold cyan]Tokenizing hardcoded test string...[/bold cyan]\n")
    display_tokens(test_string, enc, console)

if __name__ == "__main__":
    import argparse
    main()
