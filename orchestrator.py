import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

CONFIG_FILE = ".sgpe_config.json"


@dataclass
class PipelineConfig:
    train_file: str = "data/train.jsonl"
    test_file: str = "data/test.jsonl"
    vocab_file: str = "output/vocab.json"
    output_dir: str = "output"
    export_out: str = "output/tokenizer.json"
    vocab_size: int = 100_000
    min_freq: int = 2
    prune_freq: int = 100
    checkpoint_every: int = 5000


class SGPEOrchestrator:

    def __init__(self):
        self.console = Console()
        self.config = PipelineConfig()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                self.config = PipelineConfig(**{k: v for k, v in data.items()
                                                if k in PipelineConfig.__dataclass_fields__})
            except Exception:
                pass

        # Auto-switch to clean data if available and using defaults (use clean_data.py to clean the data)
        if self.config.train_file == "data/train.jsonl" and os.path.exists("data/train_clean.jsonl"):
            self.console.print("[dim]Note: Auto-switching to cleaned training data (data/train_clean.jsonl)[/dim]")
            self.config.train_file = "data/train_clean.jsonl"
            
        if self.config.test_file == "data/test.jsonl" and os.path.exists("data/test_clean.jsonl"):
            self.console.print("[dim]Note: Auto-switching to cleaned test data (data/test_clean.jsonl)[/dim]")
            self.config.test_file = "data/test_clean.jsonl"

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def display_menu(self):
        self.console.clear()
        self.console.rule("[bold cyan]SGPE Pipeline Orchestrator[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        table.add_column("", style="dim", width=4)
        table.add_column("Step", min_width=30)
        table.add_column("Description", style="dim")

        table.add_row("1", "Train", "Run GPE trainer → outputs vocab.json + tokenizer.json")
        table.add_row("2", "Evaluate", "Benchmark SGPE vs Other frontier Tokenizers on test set")
        table.add_row("3", "Export", "Re-export vocab.json → HuggingFace tokenizer.json")
        table.add_row("4", "Run Tests", "Launch battle test suite")
        table.add_row("─" * 2, "─" * 30, "─" * 40)
        table.add_row("5", "[bold yellow]Full Pipeline[/bold yellow]", "Train → Evaluate → Export → Tests")
        table.add_row("C", "Configure", "Edit paths and training params")
        table.add_row("Q", "Quit", "")

        self.console.print(table)
        self.console.print(
            f"\n[dim]vocab: {self.config.vocab_file}  |  "
            f"output: {self.config.output_dir}[/dim]"
        )

    def edit_config(self):
        self.console.rule("[bold yellow]Configuration[/bold yellow]")
        c = self.config

        c.train_file = Prompt.ask("Train JSONL", default=c.train_file)
        c.test_file = Prompt.ask("Test JSONL", default=c.test_file)
        c.vocab_file = Prompt.ask("Vocab JSON", default=c.vocab_file)
        c.output_dir = Prompt.ask("Output dir", default=c.output_dir)
        c.export_out = Prompt.ask("HF export path", default=c.export_out)
        c.vocab_size = int(Prompt.ask("Vocab size", default=str(c.vocab_size)))
        c.min_freq = int(Prompt.ask("Min freq", default=str(c.min_freq)))
        c.prune_freq = int(Prompt.ask("Prune freq", default=str(c.prune_freq)))
        c.checkpoint_every = int(Prompt.ask("Checkpoint every N merges", default=str(c.checkpoint_every)))

        self.save_config()
        self.console.print("[green]Config saved.[/green]")

    def _run(self, cmd: list[str]) -> bool:
        self.console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        result = subprocess.run(cmd)
        return result.returncode == 0

    def step_train(self) -> bool:
        self.console.rule("[bold blue]Step 1 — Train[/bold blue]")
        cmd = [
            sys.executable, "gpe_trainer.py",
            "--train_file", self.config.train_file,
            "--vocab_size", str(self.config.vocab_size),
            "--min_freq", str(self.config.min_freq),
            "--prune_freq", str(self.config.prune_freq),
            "--output_dir", self.config.output_dir,
            "--checkpoint_every", str(self.config.checkpoint_every),
        ]
        return self._run(cmd)

    def step_evaluate(self) -> bool:
        self.console.rule("[bold blue]Step 2 — Evaluate[/bold blue]")
        self.console.print("\n  [bold]Frontier Benchmarking Mode:[/bold]")
        self.console.print("    [cyan]1[/cyan] — Sampled (500 sentences, includes Gemini API, fast)")
        self.console.print("    [cyan]2[/cyan] — [bold green]Full Corpus ✦ Recommended[/bold green] (all sentences, local only.)")
        mode = Prompt.ask("  Select mode", choices=["1", "2"], default="2")

        cmd = [
            sys.executable, "tests/battle.py",
            "--test_file", self.config.test_file,
            "--vocab_file", self.config.vocab_file,
            "--only", "frontier",
        ]
        if mode == "2":
            cmd.append("--full_eval")
        return self._run(cmd)

    def step_export(self) -> bool:
        self.console.rule("[bold blue]Step 3 — Export[/bold blue]")
        cmd = [
            sys.executable, "export.py",
            "--vocab", self.config.vocab_file,
            "--out", self.config.export_out,
        ]
        return self._run(cmd)

    def step_tests(self) -> bool:
        self.console.rule("[bold blue]Step 4 — Tests (Automated)[/bold blue]")
        # Run all batteries except frontier 
        cmd = [
            sys.executable, "tests/battle.py",
            "--test_file", self.config.test_file,
            "--vocab_file", self.config.vocab_file,
            "--only", "complexity", "glitched", "roundtrip", "boundary", "zerobreak",
            "--skip_roundtrip",  # skip 1M round-trip in automated run (too slow)
        ]
        return self._run(cmd)

    def manual_test_menu(self):
        self.console.rule("[bold blue]Manual Test Menu[/bold blue]")
        cmd = [sys.executable, "tests/orchestrator.py"]
        # Use subprocess.run directly to keep interactive terminal
        try:
            subprocess.run(cmd)
        except Exception as e:
            self.console.print(f"[red]Failed to launch tests: {e}[/red]")

    def run_full_pipeline(self):
        self.console.rule("[bold green]Full Pipeline[/bold green]")
        auto = Confirm.ask("Auto-proceed through all steps without confirmation?", default=False)

        steps = [
            ("Train", self.step_train),
            ("Evaluate", self.step_evaluate),
            ("Export", self.step_export),
            ("Tests", self.step_tests),
        ]

        for name, fn in steps:
            if not auto:
                if not Confirm.ask(f"Run: [bold]{name}[/bold]?", default=True):
                    self.console.print(f"[dim]Skipped {name}.[/dim]")
                    continue

            ok = fn()
            if not ok:
                self.console.print(f"[bold red]{name} failed. Stopping pipeline.[/bold red]")
                return

        self.console.print("\n[bold green]Pipeline complete.[/bold green]")

    def run(self):
        self.load_config()

        while True:
            self.display_menu()
            choice = Prompt.ask(
                "Select",
                choices=["1", "2", "3", "4", "5", "c", "C", "q", "Q"],
            )

            if choice.lower() == "q":
                break

            if choice.lower() == "c":
                self.edit_config()
                Prompt.ask("\nPress Enter to continue...")
                continue

            if choice == "1":
                self.step_train()
            elif choice == "2":
                self.step_evaluate()
            elif choice == "3":
                self.step_export()
            elif choice == "4":
                self.manual_test_menu()
            elif choice == "5":
                self.run_full_pipeline()

            Prompt.ask("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        orch = SGPEOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        rprint("\n[red]Exiting.[/red]")
        sys.exit(0)
