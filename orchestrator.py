"""
WWHO — Pipeline Orchestrator
===================================
"""

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

CONFIG_FILE = ".wwho_config.json"


@dataclass
class PipelineConfig:
    train_file:       str = "dataset/mixed_train.jsonl"
    test_file:        str = "dataset/mixed_test.jsonl"
    vocab_file:       str = "output/vocab.json"
    output_dir:       str = "output"
    export_out_dir:   str = "output"
    tiktoken_model:   str = "o200k_base"
    vocab_size:       int = 128_000
    min_freq:         int = 2
    prune_freq:       int = 100
    checkpoint_every: int = 20_000
    script_mode:      str = "mixed"   # sinhala | devanagari | english | mixed
    num_workers:      int = 0         # 0 = auto (cpu_count - 1)
    frontier_samples: int = 2_000


class WWHOOrchestrator:

    def __init__(self):
        self.console = Console()
        self.config  = PipelineConfig()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                self.config = PipelineConfig(**{
                    k: v for k, v in data.items()
                    if k in PipelineConfig.__dataclass_fields__
                })
            except Exception:
                pass


    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def display_menu(self):
        self.console.clear()
        self.console.rule("[bold cyan]WWHO Orchestrator[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        table.add_column("", style="dim", width=4)
        table.add_column("Step", min_width=28)
        table.add_column("Description", style="dim")

        table.add_row("1", "Train",
                      "gpe_trainer.py — 30M mixed corpus (Sinhala + Hindi + English)")
        table.add_row("2", "Evaluate",
                      "Frontier benchmark — SGPE vs GPT-4o / Gemini / Llama / DeepSeek")
        table.add_row("3", "Export",
                      "export.py — WWHO HF tokenizer + unified meta-vocab")
        table.add_row("4", "Run Tests",
                      "battle.py — all 9 batteries (Sinhala + Devanagari + code-switch)")
        table.add_row("5", "Schema Check",
                      "DFA self-test: sinhala.json + devanagari.json")
        table.add_row("─" * 2, "─" * 28, "─" * 44)
        table.add_row("6", "[bold yellow]Full Pipeline[/bold yellow]",
                      "Train → Evaluate → Export → Tests → Schema")
        table.add_row("C", "Configure", "Edit paths and training params")
        table.add_row("Q", "Quit", "")

        self.console.print(table)
        c = self.config
        self.console.print(
            f"\n[dim]train: {c.train_file}  |  vocab_size: {c.vocab_size:,}  |  "
            f"script: {c.script_mode}  |  workers: {c.num_workers or 'auto'}[/dim]"
        )

    def edit_config(self):
        self.console.rule("[bold yellow]Configuration[/bold yellow]")
        c = self.config

        c.train_file       = Prompt.ask("Train JSONL",          default=c.train_file)
        c.test_file        = Prompt.ask("Test JSONL",           default=c.test_file)
        c.vocab_file       = Prompt.ask("Vocab JSON",           default=c.vocab_file)
        c.output_dir       = Prompt.ask("Output dir",           default=c.output_dir)
        c.export_out_dir   = Prompt.ask("Export dir",           default=c.export_out_dir)
        c.tiktoken_model   = Prompt.ask("Tiktoken model",       default=c.tiktoken_model)
        c.vocab_size       = int(Prompt.ask("Vocab size",       default=str(c.vocab_size)))
        c.min_freq         = int(Prompt.ask("Min freq",         default=str(c.min_freq)))
        c.prune_freq       = int(Prompt.ask("Prune freq",       default=str(c.prune_freq)))
        c.checkpoint_every = int(Prompt.ask("Checkpoint every", default=str(c.checkpoint_every)))
        c.frontier_samples = int(Prompt.ask("Frontier samples", default=str(c.frontier_samples)))
        c.num_workers      = int(Prompt.ask("Workers (0=auto)", default=str(c.num_workers)))
        c.script_mode      = Prompt.ask("Script mode",
                                        choices=["sinhala", "devanagari", "mixed"],
                                        default=c.script_mode)
        self.save_config()
        self.console.print("[green]Config saved.[/green]")

    def _run(self, cmd: list[str]) -> bool:
        self.console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        result = subprocess.run(cmd)
        return result.returncode == 0

    def step_train(self) -> bool:
        self.console.rule("[bold blue]Step 1 — Train[/bold blue]")
        c = self.config
        cmd = [
            sys.executable, "gpe_trainer.py",
            "--train_file", c.train_file,
            "--vocab_size", str(c.vocab_size),
            "--min_freq",   str(c.min_freq),
            "--prune_freq", str(c.prune_freq),
            "--output_dir", c.output_dir,
            "--checkpoint_every", str(c.checkpoint_every),
            "--script_mode", c.script_mode,
        ]
        if c.num_workers > 0:
            cmd += ["--num_workers", str(c.num_workers)]
        return self._run(cmd)

    def step_evaluate(self) -> bool:
        self.console.rule("[bold blue]Step 2 — Evaluate[/bold blue]")
        c = self.config
        cmd = [
            sys.executable, "tests/battle.py",
            "--vocab_file", c.vocab_file,
            "--test_file",  c.test_file,
            "--only", "frontier",
            "--frontier_samples", str(c.frontier_samples),
            "--full_eval",
        ]
        return self._run(cmd)

    def step_export(self) -> bool:
        self.console.rule("[bold blue]Step 3 — Export[/bold blue]")
        c = self.config
        cmd = [
            sys.executable, "export.py",
            "--vocab",          c.vocab_file,
            "--out_dir",        c.export_out_dir,
            "--tiktoken_model", c.tiktoken_model,
        ]
        return self._run(cmd)

    def step_tests(self) -> bool:
        self.console.rule("[bold blue]Step 4 — Tests (all 9 batteries)[/bold blue]")
        c = self.config
        cmd = [
            sys.executable, "tests/battle.py",
            "--vocab_file",  c.vocab_file,
            "--test_file",   c.test_file,
            "--full_corpus", c.test_file,
            "--only",
            "complexity", "glitched", "roundtrip", "boundary", "zerobreak",
            "devanagari", "codeswitching", "metavocab",
            "--roundtrip_count", "0",
            "--meta_roundtrip_count", "0",
            "--tiktoken_model", c.tiktoken_model,
        ]
        return self._run(cmd)

    def step_schema_check(self) -> bool:
        self.console.rule("[bold blue]Step 5 — Schema Check[/bold blue]")
        cmd = [sys.executable, "linguis_trie.py"]
        return self._run(cmd)

    def run_full_pipeline(self):
        self.console.rule("[bold green]Full Pipeline[/bold green]")
        steps = [
            ("Train",        self.step_train),
            ("Evaluate",     self.step_evaluate),
            ("Export",       self.step_export),
            ("Tests",        self.step_tests),
            ("Schema Check", self.step_schema_check),
        ]

        for name, fn in steps:
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
                choices=["1", "2", "3", "4", "5", "6", "c", "C", "q", "Q"],
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
                try:
                    subprocess.run([sys.executable, "tests/orchestrator.py"])
                except Exception as e:
                    self.console.print(f"[red]Failed to launch tests: {e}[/red]")
            elif choice == "5":
                self.step_schema_check()
            elif choice == "6":
                self.run_full_pipeline()

            Prompt.ask("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        orch = WWHOOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        rprint("\n[red]Exiting.[/red]")
        sys.exit(0)
