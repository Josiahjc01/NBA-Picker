#!/usr/bin/env python3
"""Simple desktop UI: Away vs Home + Date → projections + AI paste."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import date, datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetch import fetch_games_for_date  # noqa: E402
from data.static_arenas import ARENAS, normalize_abbrev  # noqa: E402
from main import format_markdown, resolve_date, run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nba_picker.gui")

# Display labels for comboboxes (abbr + city nickname)
TEAM_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def _team_choices() -> list[str]:
    return [f"{abbr} — {TEAM_NAMES.get(abbr, abbr)}" for abbr in sorted(ARENAS.keys())]


def _abbr_from_choice(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    # "BOS — Boston Celtics" or plain "BOS" / "Boston Celtics"
    if "—" in text:
        text = text.split("—", 1)[0].strip()
    elif "-" in text and text.split("-", 1)[0].strip().isalpha() and len(text.split("-", 1)[0].strip()) <= 3:
        text = text.split("-", 1)[0].strip()
    abbr = normalize_abbrev(text)
    if abbr in ARENAS:
        return abbr
    # Match by full name
    lower = text.lower()
    for a, name in TEAM_NAMES.items():
        if name.lower() == lower or name.lower().endswith(lower):
            return a
    for a, name in TEAM_NAMES.items():
        if lower in name.lower():
            return a
    return normalize_abbrev(text)


class NbaPickerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NBA Picker")
        self.geometry("900x640")
        self.minsize(700, 480)

        self._busy = False
        self._last_md: str = ""
        self._last_md_path: str | None = None
        self._last_ai: str = ""
        self._last_ai_path: str | None = None
        self._schedule_hint: str = ""

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        choices = _team_choices()

        form = ttk.LabelFrame(self, text="Matchup", padding=10)
        form.pack(fill=tk.X, **pad)

        row1 = ttk.Frame(form)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Away").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.away_var = tk.StringVar()
        self.away_box = ttk.Combobox(
            row1, textvariable=self.away_var, values=choices, width=36, state="normal"
        )
        self.away_box.grid(row=0, column=1, sticky=tk.W, padx=(0, 16))

        ttk.Label(row1, text="vs").grid(row=0, column=2, padx=4)

        ttk.Label(row1, text="Home").grid(row=0, column=3, sticky=tk.W, padx=(8, 6))
        self.home_var = tk.StringVar()
        self.home_box = ttk.Combobox(
            row1, textvariable=self.home_var, values=choices, width=36, state="normal"
        )
        self.home_box.grid(row=0, column=4, sticky=tk.W)

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, pady=(10, 2))

        ttk.Label(row2, text="Date").pack(side=tk.LEFT)
        today = datetime.now(timezone.utc).date().isoformat()
        self.date_var = tk.StringVar(value=today)
        self.date_entry = ttk.Entry(row2, textvariable=self.date_var, width=14)
        self.date_entry.pack(side=tk.LEFT, padx=(6, 16))

        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Force refresh", variable=self.force_var).pack(
            side=tk.LEFT, padx=(0, 16)
        )

        self.run_btn = ttk.Button(row2, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=2)

        self.hint_var = tk.StringVar(
            value="Pick Away, Home, and Date — then Run. Only scheduled games that day."
        )
        ttk.Label(form, textvariable=self.hint_var, wraplength=820).pack(
            anchor=tk.W, pady=(8, 0)
        )

        results_frame = ttk.LabelFrame(self, text="Results (AI paste)", padding=8)
        results_frame.pack(fill=tk.BOTH, expand=True, **pad)

        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.results = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scroll_r = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results.yview)
        self.results.configure(yscrollcommand=scroll_r.set)
        self.results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_r.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, **pad)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Open markdown…", command=self.open_markdown).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom, text="Copy AI paste", command=self.copy_ai_paste).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom, text="Save results…", command=self.save_results).pack(
            side=tk.RIGHT, padx=2
        )

    def _parse_date(self) -> date | None:
        raw = self.date_var.get().strip()
        try:
            return resolve_date(raw or None)
        except ValueError:
            messagebox.showerror("Invalid date", f"Use YYYY-MM-DD (got {raw!r})")
            return None

    def _selected_matchup(self) -> tuple[str, str] | None:
        away = _abbr_from_choice(self.away_var.get())
        home = _abbr_from_choice(self.home_var.get())
        if not away or away not in ARENAS:
            messagebox.showerror("Away team", "Pick a valid Away team (name or abbreviation).")
            return None
        if not home or home not in ARENAS:
            messagebox.showerror("Home team", "Pick a valid Home team (name or abbreviation).")
            return None
        if away == home:
            messagebox.showerror("Matchup", "Away and Home must be different teams.")
            return None
        return away, home

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_btn.configure(state=state)
        self.away_box.configure(state=state if busy else "normal")
        self.home_box.configure(state=state if busy else "normal")
        self.date_entry.configure(state=state)
        if msg:
            self.status_var.set(msg)

    def _write_results(self, text: str) -> None:
        self.results.configure(state=tk.NORMAL)
        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, text)
        self.results.see("1.0")

    def on_run(self) -> None:
        if self._busy:
            return
        target = self._parse_date()
        if not target:
            return
        matchup = self._selected_matchup()
        if not matchup:
            return
        away, home = matchup
        self._set_busy(
            True,
            f"Checking schedule for {away} @ {home} on {target.isoformat()}…",
        )

        def work() -> None:
            err: Exception | None = None
            payload: dict[str, Any] | None = None
            not_scheduled_msg: str | None = None
            try:
                games = fetch_games_for_date(target, force=self.force_var.get()) or []
                scheduled = None
                for g in games:
                    ga = normalize_abbrev(str(g.get("away_abbr") or ""))
                    gh = normalize_abbrev(str(g.get("home_abbr") or ""))
                    if ga == away and gh == home:
                        scheduled = g
                        break
                if scheduled is None:
                    slate = ", ".join(
                        f"{normalize_abbrev(str(g.get('away_abbr')))} @ "
                        f"{normalize_abbrev(str(g.get('home_abbr')))}"
                        for g in games
                    ) or "(none)"
                    not_scheduled_msg = (
                        f"{away} @ {home} is not on the NBA schedule for {target.isoformat()}.\n\n"
                        f"Games that day:\n{slate}\n\n"
                        "Pick a scheduled Away @ Home for that date, or change the date."
                    )
                else:
                    payload = run_pipeline(
                        target,
                        force=self.force_var.get(),
                        matchup=(away, home),
                        write_outputs=True,
                    )
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Run failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    self.status_var.set(f"Run failed: {err}")
                    messagebox.showerror("Run", str(err))
                    return
                if not_scheduled_msg:
                    self.hint_var.set(
                        f"{away} @ {home} is not scheduled on {target.isoformat()}."
                    )
                    self._write_results(not_scheduled_msg)
                    self.status_var.set("Not on schedule for that date.")
                    messagebox.showwarning("Not scheduled", not_scheduled_msg)
                    return
                assert payload is not None
                preds = payload.get("predictions") or []
                if not preds:
                    msg = (
                        f"No projection returned for {away} @ {home} on "
                        f"{target.isoformat()}."
                    )
                    self._write_results(msg)
                    self.status_var.set(msg)
                    return

                md = payload.get("markdown") or format_markdown(
                    target, preds, payload.get("meta") or {}
                )
                self._last_md = md
                self._last_md_path = payload.get("md_path")
                self._last_ai = payload.get("ai_paste") or ""
                self._last_ai_path = payload.get("ai_paste_path")
                self._write_results(self._plain_from_payload(payload))
                self.hint_var.set(f"{away} @ {home} · {target.isoformat()}")
                ai = payload.get("ai_paste_path") or ""
                self.status_var.set(
                    f"Done — {away} @ {home}."
                    + (f" AI paste: {ai}" if ai else "")
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _plain_from_payload(payload: dict) -> str:
        ai = (payload.get("ai_paste") or "").strip()
        if ai:
            return ai
        lines: list[str] = []
        d = payload.get("date", "")
        lines.append(f"NBA PICKER  ·  {d}")
        lines.append("=" * 60)
        for p in payload.get("predictions") or []:
            lines.append("")
            lines.append(
                f"{p['away_abbr']} @ {p['home_abbr']}   "
                f"{p['away_pts']:.1f} – {p['home_pts']:.1f}   "
                f"P(home)={p['home_win_prob']:.1%}  margin={p['margin']:+.1f}"
            )
        lines.append("")
        return "\n".join(lines)

    def copy_ai_paste(self) -> None:
        text = (self._last_ai or "").strip()
        if not text:
            messagebox.showinfo("Copy AI paste", "Run the model first.")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            path = self._last_ai_path or ""
            self.status_var.set(
                "AI paste copied to clipboard"
                + (f" (also {path})" if path else "")
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Copy AI paste", str(exc))

    def open_markdown(self) -> None:
        path = self._last_md_path
        if path and Path(path).exists():
            try:
                import os
                import subprocess

                if sys.platform.startswith("win"):
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not open %s: %s", path, exc)
        if self._last_md:
            self._write_results(self._last_md)
            self.status_var.set("Showing markdown in results pane.")
        else:
            messagebox.showinfo("Open markdown", "Run the model first.")

    def save_results(self) -> None:
        text = self.results.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text", "*.txt"),
                ("Markdown", "*.md"),
                ("All files", "*.*"),
            ],
            initialfile="nba_picker_ai_paste.txt",
        )
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
            self.status_var.set(f"Saved {path}")


def launch_gui() -> int:
    try:
        app = NbaPickerApp()
    except tk.TclError as exc:
        print(
            "tkinter/GUI unavailable in this environment:\n"
            f"  {exc}\n"
            "On Windows/macOS use a normal Python install with Tcl/Tk.\n"
            "On Debian/Ubuntu: sudo apt install python3-tk",
            file=sys.stderr,
        )
        return 1
    app.mainloop()
    return 0


def main() -> int:
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
