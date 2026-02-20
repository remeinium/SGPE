import os
import sys
from encoder import SGPEEncoder
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

def display_tokens(text, enc, console):
    tokens = enc.tokenize(text)
    char_count = len(text)
    token_count = len(tokens)
    
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
        enc = SGPEEncoder(args.vocab)
        console.print("[green]Tokenizer loaded successfully.[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to load encoder:[/bold red] {e}")
        return

    # Process file if provided
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        display_tokens(word, enc, console)
        else:
            console.print(f"[bold red]File not found:[/bold red] {args.file}")

    # Process positional arguments
    if args.words:
        for word in args.words:
            display_tokens(word, enc, console)

    # Interactive mode
    if args.interactive or (not args.words and not args.file):
        console.print("[bold blue]Interactive mode enabled.[/bold blue]")
        console.print("Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.\n")
        
        while True:
            try:
                text = Prompt.ask("Enter text to tokenize")
                if text.lower() in ["exit", "quit"]:
                    break
                if not text.strip():
                    continue
                    
                display_tokens(text, enc, console)
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                break

if __name__ == "__main__":
    import argparse
    main()
