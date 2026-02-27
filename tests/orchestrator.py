"""
WWHO — Battle Test Orchestrator
"""

import os
import sys
import json
import time
from typing import Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
import subprocess

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich import print as rprint

CONFIG_FILE = ".battle_config.json"


@dataclass
class BattleConfig:
    vocab_path:    str = os.path.join(PROJECT_ROOT, "output/vocab.json")
    test_path:     str = os.path.join(PROJECT_ROOT, "dataset/mixed_test.jsonl")
    report_output: str = os.path.join(PROJECT_ROOT, "output/battle_report.json")
    tiktoken_model:str = "o200k_base"


class BattleOrchestrator:

    BATTERY_LABELS = {
        1: "Sinhala Linguistic Complexity",
        2: "Glitched Token Detection",
        3: "Frontier Benchmarking",
        4: "Round-Trip Consistency",
        5: "Boundary & Leading Space",
        6: "Zero-Breakage Extended",
        7: "Devanagari Linguistic Complexity",
        8: "Code-Switching Integrity",
        9: "Meta-Vocab Round-Trip (WWHOMetaEncoder)",
    }

    def __init__(self):
        self.console = Console()
        self.config  = BattleConfig()
        self.loaded = False

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                self.config = BattleConfig(**{
                    k: v for k, v in data.items()
                    if k in BattleConfig.__dataclass_fields__
                })
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
            self.loaded = True
            self.console.print(f"[green]✓ Environment initialized.[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Init failed: {e}[/bold red]")
            self.console.print_exception()

    def edit_config(self):
        self.console.rule("[bold yellow]Configuration[/bold yellow]")
        self.config.vocab_path    = Prompt.ask("Vocab JSON",      default=self.config.vocab_path)
        self.config.test_path     = Prompt.ask("Test JSONL",      default=self.config.test_path)
        self.config.report_output = Prompt.ask("Report Output",   default=self.config.report_output)
        self.config.tiktoken_model= Prompt.ask("Tiktoken model",  default=self.config.tiktoken_model)
        self.save_config()
        self.loaded = False

    def display_menu(self):
        self.console.clear()
        self.console.rule("[bold cyan]WWHO Battle Test Orchestrator[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Battery")
        table.add_column("Script", style="dim")

        sinhala_batteries = [(1, "Sinhala")]
        for i in range(1, 7):
            if i in [2, 3, 4, 5]:
                script_tag = "Multi-Script" if i != 3 else "Multi-Script + Frontier"
            else:
                script_tag = "Sinhala"
            table.add_row(str(i), self.BATTERY_LABELS[i], script_tag)

        table.add_row("─", "─" * 40, "─" * 12)
        table.add_row("7",  self.BATTERY_LABELS[7], "Devanagari")
        table.add_row("8",  self.BATTERY_LABELS[8], "Mixed")
        table.add_row("9",  self.BATTERY_LABELS[9], "Mixed")
        table.add_row("─", "─" * 40, "─" * 12)
        table.add_row("10", "[bold yellow]Run ALL 9 Batteries[/bold yellow]", "All")
        table.add_row("C",  "Configure Paths", "")
        table.add_row("Q",  "Quit", "")

        self.console.print(table)

        if self.loaded:
            self.console.print(f"[dim]Vocab: {self.config.vocab_path}[/dim]")
        else:
            self.console.print("[bold red]⚠ Not initialized[/bold red]")

    def run(self):
        self.load_config()

        while True:
            self.display_menu()
            choices = [str(i) for i in range(1, 11)] + ["c", "C", "q", "Q"]
            choice = Prompt.ask("Select", choices=choices)

            if choice.lower() == "q":
                break
            if choice.lower() == "c":
                self.edit_config()
                continue

            if not self.loaded:
                self.setup()
                if not self.loaded:
                    Prompt.ask("Press Enter...")
                    continue

            if choice == "10":
                self.run_all()
            else:
                self.run_single(int(choice))

            Prompt.ask("\nPress Enter to continue...")

    def run_single(self, battery_id: int):
        self._execute_tests([self._get_battery_flag(battery_id)])

    def run_all(self):
        # Run all 9 batteries 
        flags = [self._get_battery_flag(i) for i in range(1, 10)]
        self._execute_tests(flags)

    def _get_battery_flag(self, battery_id: int) -> str:
        flags = {
            1: "complexity",
            2: "glitched",
            3: "frontier",
            4: "roundtrip",
            5: "boundary",
            6: "zerobreak",
            7: "devanagari",
            8: "codeswitching",
            9: "metavocab"
        }
        return flags.get(battery_id, "")

    def _execute_tests(self, batteries: list[str]):
        valid_batteries = [b for b in batteries if b]
        if not valid_batteries:
            self.console.print("[red]No valid batteries selected.[/red]")
            return

        cmd = [
            sys.executable, "tests/battle.py",
            "--vocab_file", self.config.vocab_path,
            "--test_file", self.config.test_path,
            "--report_output", self.config.report_output,
            "--tiktoken_model", self.config.tiktoken_model,
            "--only", *valid_batteries
        ]
        
        if "roundtrip" in valid_batteries:
           cmd.extend(["--roundtrip_count", "0"]) 
        if "metavocab" in valid_batteries:
           cmd.extend(["--meta_roundtrip_count", "0"])
        if "frontier" in valid_batteries:
           cmd.extend(["--full_eval"])

        try:
            subprocess.run(cmd)
        except Exception as e:
            self.console.print(f"[bold red]Execution failed: {e}[/bold red]")


if __name__ == "__main__":
    try:
        orch = BattleOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        rprint("\n[red]Exiting...[/red]")
        sys.exit(0)
