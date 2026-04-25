"""
Live terminal visualization — rich scoreboard showing the arms race in real time.
"""

from __future__ import annotations
import time
from collections import deque
from typing import List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ArmsFeed:
    """Circular buffer of attack/defense events for the live feed."""
    def __init__(self, maxlen=12):
        self.events: deque = deque(maxlen=maxlen)

    def add_attack(self, attack: str, caught: bool, attack_type: str, episode: int, round: int):
        icon = "🛡️  CAUGHT" if caught else "💀 EVADED"
        short = attack[:60].replace("\n", " ")
        self.events.append({
            "icon": icon,
            "caught": caught,
            "text": short,
            "type": attack_type,
            "ep": episode,
            "rnd": round,
        })

    def add_update(self, agent: str, loss: float, step: int):
        self.events.append({
            "icon": f"⚡ {agent.upper()} UPDATE",
            "caught": None,
            "text": f"loss={loss:.4f}  step={step}",
            "type": "update",
            "ep": 0,
            "rnd": 0,
        })


class ArenaVisualizer:
    """Rich terminal display for the arms race."""

    def __init__(self):
        self.console = Console()
        self.feed = ArmsFeed(maxlen=15)
        self.start_time = time.time()
        self._live: Optional[Live] = None

        # History for sparkline-like display
        self.asr_history: deque = deque(maxlen=30)
        self.def_acc_history: deque = deque(maxlen=30)
        self.atk_loss_history: deque = deque(maxlen=30)
        self.def_loss_history: deque = deque(maxlen=30)

    def _sparkline(self, values: deque, width: int = 20) -> str:
        if not values:
            return "─" * width
        blocks = "▁▂▃▄▅▆▇█"
        vals = list(values)[-width:]
        if max(vals) == min(vals):
            return "─" * len(vals)
        norm = [(v - min(vals)) / (max(vals) - min(vals) + 1e-8) for v in vals]
        return "".join(blocks[int(n * 7)] for n in norm)

    def _make_scoreboard(
        self,
        episode: int,
        round_num: int,
        atk_evasion: float,
        def_accuracy: float,
        difficulty: float,
        hall_of_fame: int,
        atk_updates: int,
        def_updates: int,
        atk_loss: float,
        def_loss: float,
        total_attacks: int,
        total_rounds: int,
    ) -> Table:
        elapsed = time.time() - self.start_time

        t = Table(box=box.DOUBLE_EDGE, expand=True, title="⚔️  ADVERSARIAL ARMS RACE  ⚔️", title_style="bold magenta")
        t.add_column("Metric", style="bold cyan", width=22)
        t.add_column("ATTACKER  🔴", style="bold red", width=28)
        t.add_column("DEFENDER  🔵", style="bold blue", width=28)

        t.add_row("Model", "Gemma 3 1B + LoRA\n(GRPO Policy)", "Gemma 3 4B + LoRA\n(Online Fine-tune)")
        t.add_row("", "", "")
        t.add_row("Score",
            f"Evasion Rate\n[red bold]{atk_evasion:.1%}[/]",
            f"Detection Accuracy\n[blue bold]{def_accuracy:.1%}[/]")
        t.add_row("Updates", f"{atk_updates} steps\nloss={atk_loss:.4f}", f"{def_updates} steps\nloss={def_loss:.4f}")
        t.add_row("Trend",
            f"[red]{self._sparkline(self.asr_history)}[/]",
            f"[blue]{self._sparkline(self.def_acc_history)}[/]")

        t.add_row("", "", "")
        t.add_row("Episode / Round", f"Ep {episode}  Round {round_num}", f"Difficulty  {difficulty:.2f}")
        t.add_row("Total Attacks", f"{total_attacks}", f"Rounds Played: {total_rounds}")
        t.add_row("Hall of Fame", f"🏆 {hall_of_fame} evasions", f"⏱ {elapsed:.0f}s elapsed")

        return t

    def _make_feed(self) -> Panel:
        lines = []
        for ev in reversed(list(self.feed.events)):
            if ev["type"] == "update":
                lines.append(f"[yellow]{ev['icon']}[/] {ev['text']}")
            elif ev["caught"]:
                lines.append(f"[blue]{ev['icon']}[/] [dim][{ev['type']}][/] {ev['text']}")
            else:
                lines.append(f"[red]{ev['icon']}[/] [dim][{ev['type']}][/] {ev['text']}")
        content = "\n".join(lines) if lines else "[dim]Waiting for first round...[/]"
        return Panel(content, title="📡 Live Attack Feed", border_style="yellow", height=18)

    def _make_hall(self, hall_of_fame: list) -> Panel:
        if not hall_of_fame:
            content = "[dim]No evasions yet — defender is winning![/]"
        else:
            recent = hall_of_fame[-5:]
            lines = []
            for i, h in enumerate(reversed(recent)):
                lines.append(f"[red]#{len(hall_of_fame)-i}[/] [dim][{h['type']}][/] {h['attack'][:70]}")
            content = "\n".join(lines)
        return Panel(content, title="🏆 Hall of Fame (Evasions)", border_style="red", height=9)

    def render(
        self,
        episode: int,
        round_num: int,
        atk_evasion: float,
        def_accuracy: float,
        difficulty: float,
        hall_of_fame: list,
        atk_updates: int,
        def_updates: int,
        atk_loss: float = 0.0,
        def_loss: float = 0.0,
        total_attacks: int = 0,
        total_rounds: int = 0,
    ):
        self.asr_history.append(atk_evasion)
        self.def_acc_history.append(def_accuracy)
        if atk_loss: self.atk_loss_history.append(atk_loss)
        if def_loss: self.def_loss_history.append(def_loss)

        scoreboard = self._make_scoreboard(
            episode, round_num, atk_evasion, def_accuracy,
            difficulty, len(hall_of_fame), atk_updates, def_updates,
            atk_loss, def_loss, total_attacks, total_rounds
        )
        feed = self._make_feed()
        hall = self._make_hall(hall_of_fame)

        if self._live:
            layout = Layout()
            layout.split_column(
                Layout(scoreboard, name="score", ratio=2),
                Layout(name="bottom", ratio=3),
            )
            layout["bottom"].split_row(
                Layout(feed, name="feed", ratio=2),
                Layout(hall, name="hall", ratio=1),
            )
            self._live.update(layout)

    def start(self):
        if RICH_AVAILABLE:
            self._live = Live(console=self.console, refresh_per_second=2, screen=False)
            self._live.start()
        else:
            print("[Viz] rich not installed — install with: pip install rich")

    def stop(self):
        if self._live:
            self._live.stop()

    def plain_log(self, msg: str):
        """Fallback logging when rich isn't running."""
        if not self._live:
            print(msg)
