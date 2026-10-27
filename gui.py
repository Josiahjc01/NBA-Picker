#!/usr/bin/env python3
"""Rough desktop UI for NBA Picker (stdlib tkinter — no extra GUI deps)."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import date, datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

# Ensure project root is on path when frozen (PyInstaller) or run as script
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import format_markdown, resolve_date, run_pipeline  # noqa: E402
from data.fetch import fetch_games_for_date  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nba_picker.gui")


class NbaPickerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NBA Picker")
        self.geometry("920x680")
        self.minsize(720, 520)

        self._games: list[dict[str, Any]] = []
        self._busy = False
        self._last_md: str = ""
        self._last_md_path: str | None = None

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        ttk.Label(top, text="Date (YYYY-MM-DD)").pack(side=tk.LEFT)
        today = datetime.now(timezone.utc).date().isoformat()
        self.date_var = tk.StringVar(value=today)
        self.date_entry = ttk.Entry(top, textvariable=self.date_var, width=14)
        self.date_entry.pack(side=tk.LEFT, padx=(6, 12))

        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Force refresh", variable=self.force_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )

        self.load_btn = ttk.Button(top, text="Load slate", command=self.on_load_slate)
        self.load_btn.pack(side=tk.LEFT, padx=2)
        self.run_btn = ttk.Button(top, text="Run model", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=2)

        mid = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, **pad)

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=1)
        mid.add(right, weight=3)

        ttk.Label(left, text="Games (multi-select; empty = all)").pack(anchor=tk.W)
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.game_list = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False, height=16
        )
        scroll_l = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.game_list.yview)
        self.game_list.configure(yscrollcommand=scroll_l.set)
        self.game_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_l.pack(side=tk.RIGHT, fill=tk.Y)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Select all", command=self.select_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="Clear", command=self.clear_selection).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(right, text="Results").pack(anchor=tk.W)
        text_frame = ttk.Frame(right)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.results = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scroll_r = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results.yview)
        self.results.configure(yscrollcommand=scroll_r.set)
        self.results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_r.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, **pad)
        self.status_var = tk.StringVar(value="Ready — load a slate or run with today’s date.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Open markdown…", command=self.open_markdown).pack(
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

    def select_all(self) -> None:
        self.game_list.selection_set(0, tk.END)

    def clear_selection(self) -> None:
        self.game_list.selection_clear(0, tk.END)

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.load_btn.configure(state=state)
        self.run_btn.configure(state=state)
        if msg:
            self.status_var.set(msg)

    def _write_results(self, text: str) -> None:
        self.results.configure(state=tk.NORMAL)
        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, text)
        self.results.see("1.0")

    def on_load_slate(self) -> None:
        if self._busy:
            return
        target = self._parse_date()
        if not target:
            return
        self._set_busy(True, f"Loading slate for {target.isoformat()}…")

        def work() -> None:
            err = None
            games: list[dict[str, Any]] = []
            try:
                games = fetch_games_for_date(target, force=self.force_var.get())
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Load slate failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    self.status_var.set(f"Load failed: {err}")
                    messagebox.showerror("Load slate", str(err))
                    return
                self._games = games or []
                self.game_list.delete(0, tk.END)
                if not self._games:
                    self.status_var.set(f"No games on {target.isoformat()}")
                    self._write_results(
                        f"No NBA games found for {target.isoformat()}.\n"
                        "Try another date."
                    )
                    return
                for g in self._games:
                    label = f"{g.get('away_abbr', '?')} @ {g.get('home_abbr', '?')}"
                    self.game_list.insert(tk.END, label)
                self.select_all()
                self.status_var.set(
                    f"Loaded {len(self._games)} game(s) for {target.isoformat()}"
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _selected_teams(self) -> set[str] | None:
        idxs = self.game_list.curselection()
        if not idxs:
            return None  # all games
        if self._games and len(idxs) == len(self._games):
            return None
        teams: set[str] = set()
        for i in idxs:
            if i < len(self._games):
                g = self._games[i]
                if g.get("home_abbr"):
                    teams.add(str(g["home_abbr"]).upper())
                if g.get("away_abbr"):
                    teams.add(str(g["away_abbr"]).upper())
            else:
                # Fallback parse "AWAY @ HOME"
                label = self.game_list.get(i)
                parts = label.split("@")
                if len(parts) == 2:
                    teams.add(parts[0].strip().upper())
                    teams.add(parts[1].strip().upper())
        return teams or None

    def on_run(self) -> None:
        if self._busy:
            return
        target = self._parse_date()
        if not target:
            return
        teams = self._selected_teams()
        self._set_busy(True, f"Running model for {target.isoformat()}…")

        def work() -> None:
            err = None
            payload = None
            try:
                payload = run_pipeline(
                    target,
                    force=self.force_var.get(),
                    team_filter=teams,
                    write_outputs=True,
                )
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Run failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    self.status_var.set(f"Run failed: {err}")
                    messagebox.showerror("Run model", str(err))
                    return
                assert payload is not None
                # Refresh list if we hadn't loaded yet
                if not self._games and payload.get("predictions"):
                    self.game_list.delete(0, tk.END)
                    self._games = []
                    for p in payload["predictions"]:
                        label = f"{p['away_abbr']} @ {p['home_abbr']}"
                        self.game_list.insert(tk.END, label)
                        self._games.append(
                            {
                                "away_abbr": p["away_abbr"],
                                "home_abbr": p["home_abbr"],
                            }
                        )
                    self.select_all()

                md = payload.get("markdown") or format_markdown(
                    target,
                    payload.get("predictions") or [],
                    payload.get("wagers") or [],
                    payload.get("meta") or {},
                )
                self._last_md = md
                self._last_md_path = payload.get("md_path")
                self._write_results(self._plain_from_payload(payload))
                n = len(payload.get("predictions") or [])
                w = len(payload.get("wagers") or [])
                path = payload.get("md_path") or ""
                self.status_var.set(
                    f"Done — {n} game(s), {w} wager(s)."
                    + (f" Saved {path}" if path else "")
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _plain_from_payload(payload: dict) -> str:
        lines: list[str] = []
        d = payload.get("date", "")
        lines.append(f"NBA PICKER  ·  {d}")
        lines.append("=" * 60)
        preds = payload.get("predictions") or []
        if not preds:
            lines.append("")
            lines.append(f"No NBA games for {d}.")
            lines.append("")
            return "\n".join(lines)
        for p in preds:
            lines.append("")
            lines.append(
                f"{p['away_abbr']} @ {p['home_abbr']}   "
                f"{p['away_pts']:.1f} – {p['home_pts']:.1f}   "
                f"P(home)={p['home_win_prob']:.1%}  margin={p['margin']:+.1f}"
            )
            notes = p.get("notes") or {}
            if notes.get("injuries"):
                lines.append("  Injuries: " + "; ".join(notes["injuries"]))
            if notes.get("rest_travel"):
                lines.append("  Rest/travel: " + ", ".join(notes["rest_travel"]))
            if notes.get("play_types"):
                lines.append("  Play-types: " + "; ".join(notes["play_types"]))
        lines.append("")
        lines.append("-" * 60)
        lines.append("TOP +EV WAGERS")
        lines.append("-" * 60)
        wagers = payload.get("wagers") or []
        if not wagers:
            lines.append("(none — set ODDS_API_KEY or THERUNDOWN_API_KEY for book lines)")
        else:
            for i, w in enumerate(wagers, 1):
                lines.append(
                    f"{i}. {w['game']}  {w['side']}  {w['book']}  "
                    f"{w['odds']:+d}  edge {w['edge_pct']:.1f}%  "
                    f"stake {w['suggested_stake']:.2f}u"
                )
        lines.append("")
        return "\n".join(lines)

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
            initialfile="nba_picker_results.txt",
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
