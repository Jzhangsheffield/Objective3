from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


DEMO_ROOT = Path(__file__).resolve().parent
COLORS = {
    "page": "#0B1220",
    "panel": "#111C2E",
    "panel_alt": "#16243A",
    "border": "#2A3B54",
    "text": "#EAF0F8",
    "muted": "#91A3BA",
    "blue": "#46A0FF",
    "green": "#3DDC97",
    "amber": "#F5C451",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DemoLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Choose Task Graph Demo")
        self.root.geometry("980x570")
        self.root.minsize(900, 520)
        self.root.configure(bg=COLORS["page"])
        self.profiles = read_json(DEMO_ROOT / "demo_profiles.json")["profiles"]
        self._build_ui()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=COLORS["page"])
        header.pack(fill="x", padx=34, pady=(28, 18))
        tk.Label(
            header,
            text="TASK GRAPH REAL-TIME DEMOS",
            bg=COLORS["page"],
            fg=COLORS["blue"],
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Choose a participant and run",
            bg=COLORS["page"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 26),
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            header,
            text=(
                "Each demo uses supplied action boundaries and compares "
                "M0, M3 and E2E-Node-Scratch."
            ),
            bg=COLORS["page"],
            fg=COLORS["muted"],
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(7, 0))

        cards = tk.Frame(self.root, bg=COLORS["page"])
        cards.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        for column in range(len(self.profiles)):
            cards.grid_columnconfigure(column, weight=1, uniform="profile")
        cards.grid_rowconfigure(0, weight=1)

        for column, profile in enumerate(self.profiles):
            self._profile_card(cards, column, profile)

        footer = tk.Label(
            self.root,
            text=(
                "Node is the primary output. Tier-3 is shown only as an auxiliary label. "
                "Ground truth is revealed after all three predictions."
            ),
            bg=COLORS["page"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        footer.pack(pady=(0, 18))

    def _profile_card(self, parent: tk.Widget, column: int, profile: dict) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["panel"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(6 if column else 0, 0 if column else 6),
        )
        tk.Frame(panel, bg=COLORS["blue"], height=4).pack(fill="x")
        tk.Label(
            panel,
            text=profile["title"],
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 23),
        ).pack(anchor="w", padx=24, pady=(24, 3))
        tk.Label(
            panel,
            text=profile["subtitle"],
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=24)

        summary_path = DEMO_ROOT / profile["validation_summary"]
        summary = read_json(summary_path) if summary_path.is_file() else None
        if summary is None:
            metrics_text = "Validation results are not available yet."
            metrics_color = COLORS["amber"]
        else:
            actions = int(summary["actions"])
            metrics_text = (
                f"M0   {summary['m0_correct']:>2}/{actions}   "
                f"{100 * summary['m0_accuracy']:5.1f}%\n\n"
                f"M3   {summary['m3_correct']:>2}/{actions}   "
                f"{100 * summary['m3_accuracy']:5.1f}%\n\n"
                f"E2E  {summary['e2e_correct']:>2}/{actions}   "
                f"{100 * summary['e2e_accuracy']:5.1f}%"
            )
            metrics_color = COLORS["green"]
        tk.Label(
            panel,
            text=metrics_text,
            bg=COLORS["panel_alt"],
            fg=metrics_color,
            font=("Consolas", 14),
            justify="left",
            anchor="w",
            padx=18,
            pady=18,
        ).pack(fill="x", padx=24, pady=(28, 22))

        tk.Button(
            panel,
            text=f"Run {profile['title']}  ▶",
            command=lambda selected=profile: self._launch(selected),
            bg=COLORS["blue"],
            fg="#07111E",
            activebackground="#76B8FF",
            activeforeground="#07111E",
            relief="flat",
            font=("Segoe UI Semibold", 12),
            padx=18,
            pady=11,
        ).pack(fill="x", padx=24, pady=(0, 24))

    def _launch(self, profile: dict) -> None:
        config_path = (DEMO_ROOT / profile["config"]).resolve()
        config = read_json(config_path)
        display_video = DEMO_ROOT / config["paths"]["display_video"]
        if not display_video.is_file():
            messagebox.showerror(
                "Display video missing",
                (
                    f"The display video for {profile['title']} is missing.\n\n"
                    "Run build_all_display_videos.bat and try again."
                ),
            )
            return
        subprocess.Popen(
            [sys.executable, str(DEMO_ROOT / "demo.py"), "--config", str(config_path)],
            cwd=str(DEMO_ROOT),
        )
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DemoLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
