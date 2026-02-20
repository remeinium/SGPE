"""
SGPE Battle Test Orchestrator
"""

import os
import sys
import json
import time
from typing import Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project root is the parent of the tests/ directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich import print as rprint

from battle import (
    SGPEEncoder,
    BattleReport,
    TestStatus,
    load_test_data,
    test_linguistic_complexity,
    test_glitched_tokens,
    test_frontier_benchmarking,
    test_roundtrip_consistency,
    test_boundary_edge_cases,
    test_zero_breakage_extended,
    print_final_report,
    save_report_json
)

CONFIG_FILE = ".battle_config.json"


@dataclass
class BattleConfig:
    vocab_path: str = os.path.join(PROJECT_ROOT, "output/vocab.json")
    test_path: str = os.path.join(PROJECT_ROOT, "data/test.jsonl")
    corpus_path: str = os.path.join(PROJECT_ROOT, "data/full_1m.jsonl")
    report_output: str = os.path.join(PROJECT_ROOT, "output/battle_report.json")


class BattleOrchestrator:
    
    def __init__(self):
        self.console = Console()
        self.config = BattleConfig()
        self.sgpe: Optional[SGPEEncoder] = None
        self.test_sentences: list[str] = []
        self.loaded = False

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.config = BattleConfig(**data)
                self.console.print(f"[green]Loaded config from {CONFIG_FILE}[/green]")
            except Exception as e:
                self.console.print(f"[red]Config load failed: {e}[/red]")

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config.__dict__, f, indent=2)
            self.console.print(f"[green]Config saved to {CONFIG_FILE}[/green]")
        except Exception as e:
            self.console.print(f"[red]Config save failed: {e}[/red]")

    def setup(self):
        if self.loaded:
            return

        self.console.rule("[bold blue]Initializing Test Environment[/bold blue]")
        
        if not os.path.exists(self.config.vocab_path):
            self.console.print(f"[bold red]Vocab not found: {self.config.vocab_path}[/bold red]")
            if Confirm.ask("Update path?"):
                self.edit_config()
                return self.setup()
            return

        try:
            with self.console.status("Loading SGPE..."):
                self.sgpe = SGPEEncoder(self.config.vocab_path)
            self.console.print(f"[green]✓ SGPE loaded[/green] (Vocab: {len(self.sgpe.vocab):,})")

            with self.console.status(f"Loading test data..."):
                self.test_sentences = load_test_data(self.config.test_path)
            self.console.print(f"[green]✓ Test data loaded[/green] ({len(self.test_sentences):,} sentences)")
            
            self.loaded = True

        except Exception as e:
            self.console.print(f"[bold red]Init failed: {e}[/bold red]")
            self.console.print_exception()

    def edit_config(self):
        self.console.rule("[bold yellow]Configuration[/bold yellow]")
        
        self.config.vocab_path = Prompt.ask("Vocab JSON", default=self.config.vocab_path)
        self.config.test_path = Prompt.ask("Test JSONL", default=self.config.test_path)
        self.config.corpus_path = Prompt.ask("Full Corpus", default=self.config.corpus_path)
        self.config.report_output = Prompt.ask("Report Output", default=self.config.report_output)
        
        self.save_config()
        self.loaded = False 
        self.sgpe = None

    def display_menu(self):
        self.console.clear()
        self.console.rule("[bold cyan]SGPE Battle Test Orchestrator[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Test")
        
        table.add_row("1", "Linguistic Complexity (2K Sanskrit/Pali words)")
        table.add_row("2", "Glitched Token Detection")
        table.add_row("3", "Frontier Benchmarking")
        table.add_row("4", "Round-Trip Consistency")
        table.add_row("5", "Boundary & Leading Space Edge-Cases")
        table.add_row("6", "Zero-Breakage Guarantee (Extended)")
        table.add_row("7", "[bold yellow]Run ALL Tests[/bold yellow]")
        table.add_row("C", "Configure Paths")
        table.add_row("Q", "Quit")
        
        self.console.print(table)
        
        if self.loaded:
            self.console.print(f"[dim]Vocab: {self.config.vocab_path}[/dim]")
        else:
            self.console.print("[bold red]⚠ Not initialized[/bold red]")

    def run(self):
        self.load_config()
        
        while True:
            self.display_menu()
            choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5", "6", "7", "c", "C", "q", "Q"])
            
            if choice.lower() == 'q':
                break
            
            if choice.lower() == 'c':
                self.edit_config()
                continue

            if not self.loaded:
                self.setup()
                if not self.loaded:
                    Prompt.ask("Press Enter...")
                    continue

            if choice == '7':
                self.run_all()
            else:
                self.run_single(int(choice))
            
            Prompt.ask("\nPress Enter to continue...")

    def run_single(self, test_id: int):
        report = BattleReport()
        report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        try:
            if test_id == 1:
                report.add(test_linguistic_complexity(self.sgpe))
            elif test_id == 2:
                report.add(test_glitched_tokens(self.sgpe, self.test_sentences))
            elif test_id == 3:
                self.console.print("\n  [bold]Frontier Benchmarking Mode:[/bold]")
                self.console.print("    [cyan]1[/cyan] — Sampled (500 sentences, includes Gemini API, fast)")
                self.console.print("    [cyan]2[/cyan] — [bold green]Full Corpus ✦ Recommended[/bold green] (all sentences, local only.)")
                mode = Prompt.ask("  Select mode", choices=["1", "2"], default="2")
                full_eval = (mode == "2")
                report.add(test_frontier_benchmarking(
                    self.sgpe, self.test_sentences, full_eval=full_eval
                ))
            elif test_id == 4:
                count = IntPrompt.ask("Sentence count?", default=100000)
                report.add(test_roundtrip_consistency(
                    self.sgpe, 
                    self.test_sentences, 
                    full_corpus_path=self.config.corpus_path,
                    target_count=count
                ))
            elif test_id == 5:
                report.add(test_boundary_edge_cases(self.sgpe))
            elif test_id == 6:
                report.add(test_zero_breakage_extended(self.sgpe))
            
            print_final_report(report)
            
        except Exception as e:
            self.console.print(f"[bold red]Test failed: {e}[/bold red]")
            self.console.print_exception()

    def run_all(self):
        report = BattleReport()
        report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        try:
            report.add(test_linguistic_complexity(self.sgpe))
            report.add(test_glitched_tokens(self.sgpe, self.test_sentences))
            # Default to full_eval=True (Recommended) for run_all — paper-quality TWR
            report.add(test_frontier_benchmarking(self.sgpe, self.test_sentences, full_eval=True))
            
            if Confirm.ask("Run 1M round-trip?", default=False):
                report.add(test_roundtrip_consistency(
                    self.sgpe, 
                    self.test_sentences, 
                    full_corpus_path=self.config.corpus_path,
                    target_count=1_000_000
                ))
            
            report.add(test_boundary_edge_cases(self.sgpe))
            report.add(test_zero_breakage_extended(self.sgpe))
            
            print_final_report(report)
            save_report_json(report, self.config.report_output)
            
        except Exception as e:
            self.console.print(f"[bold red]Suite failed: {e}[/bold red]")
            self.console.print_exception()


if __name__ == "__main__":
    try:
        orch = BattleOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        rprint("\n[red]Exiting...[/red]")
        sys.exit(0)
