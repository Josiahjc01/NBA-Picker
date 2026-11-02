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

import config  # noqa: E402
from data.fetch import fetch_games_for_date  # noqa: E402
from data.static_arenas import ARENAS, normalize_abbrev  # noqa: E402
from main import format_markdown, resolve_date, run_pipeline  # noqa: E402
from model.brief import HIGHLIGHT_TITLES  # noqa: E402
from data.learning_store import (  # noqa: E402
    apply_learning_knobs,
    auto_update_knobs,
    format_learning_summary,
    import_paste,
    reset_knobs_to_defaults,
    summarize_learning_grades,
)
from model.calibrate import (  # noqa: E402
    effective_score_sigma,
    format_calibration_report,
    grade_predictions,
    load_calibration_state,
)
from safe_text import ascii_safe  # noqa: E402

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


def _bind_typeahead(box: ttk.Combobox, all_choices: list[str]) -> None:
    """Filter combobox values as the user types (case-insensitive substring)."""

    def _filter(_event: tk.Event | None = None) -> None:
        typed = (box.get() or "").strip()
        if not typed:
            box.configure(values=all_choices)
            return
        low = typed.lower()
        filtered = [c for c in all_choices if low in c.lower()]
        # Keep current typed text; only shrink the dropdown list
        box.configure(values=filtered if filtered else all_choices)

    def _reset_list(_event: tk.Event | None = None) -> None:
        # After a selection, restore full list for next open
        box.configure(values=all_choices)

    box.bind("<KeyRelease>", _filter)
    box.bind("<<ComboboxSelected>>", _reset_list)
    box.bind("<FocusOut>", _reset_list)


class NbaPickerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NBA Picker")
        self.geometry("920x700")
        self.minsize(720, 520)

        self._busy = False
        self._last_md: str = ""
        self._last_md_path: str | None = None
        self._last_ai: str = ""
        self._last_ai_path: str | None = None
        self._last_payload: dict[str, Any] | None = None
        self._schedule_hint: str = ""
        self._team_choices = _team_choices()

        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        choices = self._team_choices

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
        _bind_typeahead(self.away_box, choices)

        ttk.Label(row1, text="vs").grid(row=0, column=2, padx=4)

        ttk.Label(row1, text="Home").grid(row=0, column=3, sticky=tk.W, padx=(8, 6))
        self.home_var = tk.StringVar()
        self.home_box = ttk.Combobox(
            row1, textvariable=self.home_var, values=choices, width=36, state="normal"
        )
        self.home_box.grid(row=0, column=4, sticky=tk.W)
        _bind_typeahead(self.home_box, choices)

        row2 = ttk.Frame(form)
        row2.pack(fill=tk.X, pady=(10, 2))

        ttk.Label(row2, text="Date").pack(side=tk.LEFT)
        today = datetime.now(timezone.utc).date().isoformat()
        self.date_var = tk.StringVar(value=today)
        self.date_entry = ttk.Entry(row2, textvariable=self.date_var, width=14)
        self.date_entry.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(row2, text="Sims").pack(side=tk.LEFT)
        sim_labels = [f"{n:,}" for n in config.SIM_COUNT_CHOICES]
        default_sim = f"{int(config.SIM_COUNT):,}"
        if default_sim not in sim_labels:
            default_sim = "10,000"
        self.sims_var = tk.StringVar(value=default_sim)
        self.sims_box = ttk.Combobox(
            row2,
            textvariable=self.sims_var,
            values=sim_labels,
            width=10,
            state="readonly",
        )
        self.sims_box.pack(side=tk.LEFT, padx=(6, 16))

        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Force refresh", variable=self.force_var).pack(
            side=tk.LEFT, padx=(0, 16)
        )

        self.run_btn = ttk.Button(row2, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=2)

        self.hint_var = tk.StringVar(
            value="Type to search Away/Home, pick Date + Sims, then Run. "
            "Only scheduled games that day."
        )
        ttk.Label(form, textvariable=self.hint_var, wraplength=860).pack(
            anchor=tk.W, pady=(8, 0)
        )

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, **pad)

        results_tab = ttk.Frame(self.notebook, padding=4)
        quarters_tab = ttk.Frame(self.notebook, padding=4)
        calib_tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(results_tab, text="Results")
        self.notebook.add(quarters_tab, text="Quarters")
        self.notebook.add(calib_tab, text="Calibration")

        self.results = self._make_text(results_tab)
        self.quarters_view = self._make_text(quarters_tab)
        self._build_calibration_tab(calib_tab)
        self._refresh_calibration_pane(silent=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, **pad)
        self.status_var = tk.StringVar(
            value=f"Ready. Effective SCORE_SIGMA={effective_score_sigma():.2f}"
        )
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Grade finals", command=self.on_grade).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom, text="Open markdown...", command=self.open_markdown).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom, text="Copy AI paste", command=self.copy_ai_paste).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(bottom, text="Save results...", command=self.save_results).pack(
            side=tk.RIGHT, padx=2
        )

    @staticmethod
    def _make_text(parent: ttk.Frame) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 10), state=tk.NORMAL)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        return text

    def _build_calibration_tab(self, parent: ttk.Frame) -> None:
        """Calibration + offline Learning: report, paste import, knobs."""
        apply_learning_knobs()
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=(0, 4))
        self.learning_summary_var = tk.StringVar(value="Learning: n=0")
        ttk.Label(top, textvariable=self.learning_summary_var).pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh", command=self._refresh_calibration_pane).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(top, text="Reset knobs", command=self.on_reset_knobs).pack(
            side=tk.RIGHT, padx=2
        )
        ttk.Button(top, text="Apply knobs", command=self.on_apply_knobs).pack(
            side=tk.RIGHT, padx=2
        )

        # Report (calibration + learning)
        report_frame = ttk.LabelFrame(parent, text="Report (MAE / Brier / knobs)")
        report_frame.pack(fill=tk.BOTH, expand=True)
        self.calib_view = self._make_text(report_frame)

        # Paste area
        paste_frame = ttk.LabelFrame(
            parent,
            text="Paste real game / AI comparison (JSON or ASCII DATE/ACT/PRED)",
        )
        paste_frame.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
        paste_wrap = ttk.Frame(paste_frame)
        paste_wrap.pack(fill=tk.BOTH, expand=True)
        self.paste_view = tk.Text(
            paste_wrap, wrap=tk.WORD, font=("Consolas", 9), height=8
        )
        paste_scroll = ttk.Scrollbar(
            paste_wrap, orient=tk.VERTICAL, command=self.paste_view.yview
        )
        self.paste_view.configure(yscrollcommand=paste_scroll.set)
        self.paste_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        paste_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        paste_btns = ttk.Frame(paste_frame)
        paste_btns.pack(fill=tk.X, pady=4)
        ttk.Button(
            paste_btns, text="Apply paste", command=self.on_apply_paste
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            paste_btns, text="Clear paste", command=lambda: self.paste_view.delete("1.0", tk.END)
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            paste_btns,
            text="Means/win% only — never Highlights sim-maxes. Code themes -> inbox.",
        ).pack(side=tk.LEFT, padx=8)

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

    def _selected_sims(self) -> int:
        raw = (self.sims_var.get() or "").replace(",", "").strip()
        try:
            n = int(raw)
        except ValueError:
            n = int(config.SIM_COUNT)
        if n not in config.SIM_COUNT_CHOICES:
            n = int(config.SIM_COUNT)
        return n

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_btn.configure(state=state)
        self.away_box.configure(state=state if busy else "normal")
        self.home_box.configure(state=state if busy else "normal")
        self.date_entry.configure(state=state)
        self.sims_box.configure(state=tk.DISABLED if busy else "readonly")
        if msg:
            self.status_var.set(msg)

    def _write_pane(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, ascii_safe(text))
        widget.see("1.0")

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
        sim_count = self._selected_sims()
        big = sim_count >= config.SIM_PROGRESS_THRESHOLD
        status = (
            f"Running {away} @ {home} on {target.isoformat()} "
            f"({sim_count:,} sims"
            + (" - large run, please wait..." if big else "")
            + ")..."
        )
        self._set_busy(True, status)

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
                        sim_count=sim_count,
                    )
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Run failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    msg = ascii_safe(err)
                    self.status_var.set(f"Run failed: {msg}")
                    messagebox.showerror("Run", msg)
                    return
                if not_scheduled_msg:
                    self.hint_var.set(
                        f"{away} @ {home} is not scheduled on {target.isoformat()}."
                    )
                    self._write_pane(self.results, ascii_safe(not_scheduled_msg))
                    self._write_pane(self.quarters_view, "")
                    self.status_var.set("Not on schedule for that date.")
                    messagebox.showwarning("Not scheduled", ascii_safe(not_scheduled_msg))
                    return
                assert payload is not None
                preds = payload.get("predictions") or []
                if not preds:
                    msg = (
                        f"No projection returned for {away} @ {home} on "
                        f"{target.isoformat()}."
                    )
                    self._write_pane(self.results, msg)
                    self._write_pane(self.quarters_view, "")
                    self.status_var.set(msg)
                    return

                md = payload.get("markdown") or format_markdown(
                    target, preds, payload.get("meta") or {}
                )
                self._last_md = md
                self._last_md_path = payload.get("md_path")
                self._last_ai = payload.get("ai_paste") or ""
                self._last_ai_path = payload.get("ai_paste_path")
                self._last_payload = payload
                self._write_pane(self.results, self._format_results(payload))
                self._write_pane(self.quarters_view, self._format_quarters(payload))
                self.notebook.select(0)
                self.hint_var.set(f"{away} @ {home} | {target.isoformat()}")
                elapsed = None
                if preds:
                    elapsed = preds[0].get("sim_elapsed_sec")
                ai = payload.get("ai_paste_path") or ""
                status_bits = [f"Done - {away} @ {home} ({sim_count:,} sims)"]
                if elapsed is not None:
                    status_bits.append(f"MC {elapsed:.2f}s")
                if ai:
                    status_bits.append(f"AI paste: {ai}")
                self.status_var.set(". ".join(status_bits))

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _fmt_leader(side: dict[str, Any] | None, fallback_team: str) -> str:
        if not side:
            return f"{fallback_team} (n/a)"
        name = side.get("player") or "?"
        val = side.get("value")
        team = side.get("team") or fallback_team
        if val is None:
            return f"{team} {name} (n/a)"
        return f"{team} {name} {val}"

    def _format_results(self, payload: dict[str, Any]) -> str:
        """Readable Results pane: Projection / Highlights / Notes / AI paste."""
        lines: list[str] = []
        d = payload.get("date", "")
        preds = payload.get("predictions") or []
        lines.append(f"NBA PICKER  |  {d}")
        lines.append("=" * 64)
        lines.append("")

        if not preds:
            lines.append("No projection for this matchup.")
            lines.append("")
            return "\n".join(lines)

        for pred in preds:
            away = pred.get("away_abbr") or "?"
            home = pred.get("home_abbr") or "?"
            nsim = pred.get("sim_count") or config.SIM_COUNT

            lines.append("PROJECTION")
            lines.append("-" * 64)
            lines.append(f"  {away} @ {home}")
            lines.append(
                f"  Mean score (sim n={nsim:,}):  "
                f"{away} {pred.get('away_pts')}  -  {home} {pred.get('home_pts')}"
            )
            lines.append(
                f"  Total {pred.get('total')}   Margin {pred.get('margin'):+.1f} home   "
                f"Pace {pred.get('pace')}"
            )
            lines.append(
                f"  Win %:  {home} {float(pred.get('home_win_prob') or 0):.1%}  /  "
                f"{away} {float(pred.get('away_win_prob') or 0):.1%}  "
                f"(confidence {pred.get('confidence')})"
            )
            sims = pred.get("sims") or {}
            if sims:
                lines.append(
                    f"  Score band p10-p90:  {away} {sims.get('away_pts_p10')}-"
                    f"{sims.get('away_pts_p90')}  /  {home} {sims.get('home_pts_p10')}-"
                    f"{sims.get('home_pts_p90')}  "
                    f"(total {sims.get('total_p10')}-{sims.get('total_p90')})"
                )
            tags = pred.get("form_tags") or []
            if tags:
                lines.append("  Form tags: " + ", ".join(tags))
            if pred.get("clutch_q4"):
                lines.append("  Clutch: Q4 weighting ON (projected close)")
            elapsed = pred.get("sim_elapsed_sec")
            if elapsed is not None and int(nsim) >= config.SIM_PROGRESS_THRESHOLD:
                lines.append(f"  Sims runtime: {elapsed:.2f}s")
            lines.append("")

            n_hl = int(pred.get("sim_count") or config.SIM_COUNT)
            lines.append(f"HIGHLIGHTS  (sim max over {n_hl:,} runs)")
            lines.append("-" * 64)
            highlights = pred.get("highlights") or []
            if not highlights:
                lines.append("  (no sim leaders)")
            else:
                for h in highlights:
                    title = h.get("title") or HIGHLIGHT_TITLES.get(h.get("stat"), h.get("stat"))
                    away_s = self._fmt_leader(h.get("away"), away)
                    home_s = self._fmt_leader(h.get("home"), home)
                    lines.append(f"  {title:10}  {away_s}")
                    lines.append(f"  {'':10}  {home_s}")
            lines.append("")

            lines.append("NOTES")
            lines.append("-" * 64)
            notes = pred.get("notes") or {}
            any_note = False
            for label, key in (
                ("Injuries", "injuries"),
                ("Hot/cold", "form"),
                ("Quarter edges", "quarters"),
                ("Rest / travel", "rest_travel"),
                ("Play-types", "play_types"),
            ):
                vals = notes.get(key) or []
                if not vals:
                    continue
                any_note = True
                joined = "; ".join(vals) if isinstance(vals[0], str) else str(vals)
                lines.append(f"  {label}: {joined}")
            if not any_note:
                lines.append("  (none)")
            lines.append("")

        lines.append("AI PASTE  (copyable block - use Copy AI paste button)")
        lines.append("-" * 64)
        ai = (payload.get("ai_paste") or "").strip()
        if ai:
            lines.append(ai)
        else:
            lines.append("(run produced no AI paste)")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_quarters(payload: dict[str, Any]) -> str:
        """Quarters tab: Q1-Q4 projected scores, edges, tendency + efficiency."""
        lines: list[str] = []
        d = payload.get("date", "")
        preds = payload.get("predictions") or []
        lines.append(f"QUARTER BY QUARTER  |  {d}")
        lines.append("=" * 64)
        lines.append("")
        if not preds:
            lines.append("No quarter breakdown available.")
            lines.append("")
            return "\n".join(lines)

        for pred in preds:
            away = pred.get("away_abbr") or "?"
            home = pred.get("home_abbr") or "?"
            qblock = pred.get("quarters") or {}
            lines.append(f"{away} @ {home}")
            lines.append("-" * 64)
            lines.append(
                f"  Game totals (sim mean):  {away} {pred.get('away_pts')}  -  "
                f"{home} {pred.get('home_pts')}  (total {pred.get('total')})"
            )
            if pred.get("clutch_q4"):
                lines.append("  Clutch Q4 weighting: ON")
            lines.append("")

            quarters = qblock.get("quarters") or []
            if not quarters:
                lines.append("  (no quarter profiles - schedule lookback empty)")
                lines.append("")
                continue

            lines.append(
                f"  {'Qtr':<4}  {away:>8}  {home:>8}  {'Edge(H)':>8}  "
                f"{'Tend '+away:>10}  {'Tend '+home:>10}"
            )
            lines.append("  " + "-" * 58)
            for q in quarters:
                label = q.get("label") or f"Q{q.get('quarter')}"
                a_pts = q.get("away_pts")
                h_pts = q.get("home_pts")
                edge = q.get("edge_home")
                a_tend = q.get("away_tendency")
                h_tend = q.get("home_tendency")
                lines.append(
                    f"  {label:<4}  {a_pts!s:>8}  {h_pts!s:>8}  "
                    f"{edge!s:>8}  {a_tend!s:>10}  {h_tend!s:>10}"
                )
            lines.append("")

            lines.append("  Tendency (recent lookback avg pts / allow)")
            for q in quarters:
                label = q.get("label") or f"Q{q.get('quarter')}"
                a_off = q.get("away_tendency")
                h_off = q.get("home_tendency")
                a_all = q.get("away_allow")
                h_all = q.get("home_allow")
                te = q.get("tendency_edge_home")
                te_s = f"{te:+.2f}" if te is not None else "n/a"
                lines.append(
                    f"  {label}: {away} off {a_off} allow {a_all}  |  "
                    f"{home} off {h_off} allow {h_all}  |  edge {te_s}"
                )
            lines.append("")

            eff = qblock.get("efficiency") or {}
            lines.append("  Efficiency / pace summary")
            lines.append(
                f"  Pace {eff.get('pace')}   "
                f"poss/Q proxy {eff.get('poss_per_quarter_proxy')}   "
                f"pts/poss {away} {eff.get('away_pts_per_poss_proxy')}  "
                f"{home} {eff.get('home_pts_per_poss_proxy')}"
            )
            lines.append(
                f"  Sample games: {away} {eff.get('away_games_sample')}  "
                f"{home} {eff.get('home_games_sample')}   "
                f"basis: {qblock.get('basis')}"
            )
            notes = (pred.get("notes") or {}).get("quarters") or []
            if notes:
                lines.append("  Matchup notes: " + "; ".join(notes))
            lines.append("")

        lines.append(
            "Projected quarter scores scale each team's recent Q1-Q4 scoring "
            "share to the sim-mean team total."
        )
        lines.append("")
        return "\n".join(lines)

    def _refresh_calibration_pane(self, silent: bool = False) -> None:
        apply_learning_knobs()
        state = load_calibration_state()
        calib_text = format_calibration_report(state if state else None)
        learn_text = format_learning_summary()
        text = learn_text + "\n" + calib_text
        self._write_pane(self.calib_view, text)
        summary = summarize_learning_grades()
        n = int(summary.get("n_games") or 0)
        mae_t = summary.get("mae_total")
        mae_m = summary.get("mae_margin")
        brier = summary.get("brier")
        self.learning_summary_var.set(
            f"Learning: n={n}  MAE total={mae_t}  MAE margin={mae_m}  "
            f"Brier={brier}  SCORE_SIGMA={effective_score_sigma():.2f}"
        )
        if not silent:
            self.status_var.set(
                f"Calibration/learning loaded ({n} graded). "
                f"Effective SCORE_SIGMA={effective_score_sigma():.2f}"
            )

    def on_apply_paste(self) -> None:
        if self._busy:
            return
        raw = self.paste_view.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showinfo(
                "Paste",
                "Paste real_box / ai_comparison JSON or ASCII DATE/ACT/PRED block first.",
            )
            return
        self._set_busy(True, "Applying paste into offline learning...")

        def work() -> None:
            err: Exception | None = None
            result: dict[str, Any] | None = None
            try:
                result = import_paste(raw, auto_knobs=True)
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Paste import failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    msg = ascii_safe(err)
                    self.status_var.set(f"Paste failed: {msg}")
                    messagebox.showerror("Paste", msg)
                    return
                assert result is not None
                self._refresh_calibration_pane(silent=True)
                self.notebook.select(2)
                g = result.get("grade") or {}
                self.status_var.set(
                    f"Paste graded {g.get('away_abbr')}@{g.get('home_abbr')} "
                    f"err_total={g.get('abs_err_total')}  "
                    f"SCORE_SIGMA={effective_score_sigma():.2f}"
                )
                themes = result.get("code_themes_inbox")
                extra = (
                    f"\nCode themes queued (not auto-applied):\n{themes}"
                    if themes
                    else ""
                )
                knobs = result.get("knobs") or {}
                if knobs.get("applied"):
                    extra += f"\nKnobs updated: {knobs.get('changes')}"
                elif knobs.get("skipped_reason"):
                    extra += f"\nKnobs: {knobs.get('skipped_reason')}"
                messagebox.showinfo(
                    "Paste applied",
                    ascii_safe(
                        f"Graded means/win% for {g.get('away_abbr')}@{g.get('home_abbr')}.\n"
                        f"abs_err_total={g.get('abs_err_total')}  "
                        f"abs_err_margin={g.get('abs_err_margin')}  "
                        f"brier={g.get('brier')}"
                        f"{extra}"
                    ),
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def on_apply_knobs(self) -> None:
        if self._busy:
            return
        self._set_busy(True, "Applying capped learning knobs...")

        def work() -> None:
            err: Exception | None = None
            result: dict[str, Any] | None = None
            try:
                result = auto_update_knobs(reason="gui_apply", force=True)
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Apply knobs failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    msg = ascii_safe(err)
                    self.status_var.set(f"Apply knobs failed: {msg}")
                    messagebox.showerror("Apply knobs", msg)
                    return
                assert result is not None
                self._refresh_calibration_pane(silent=True)
                self.notebook.select(2)
                if result.get("applied"):
                    self.status_var.set(
                        f"Knobs applied. SCORE_SIGMA={effective_score_sigma():.2f}"
                    )
                    messagebox.showinfo(
                        "Apply knobs",
                        ascii_safe(f"Updated: {result.get('changes')}"),
                    )
                else:
                    reason = result.get("skipped_reason") or "No changes"
                    self.status_var.set(ascii_safe(str(reason)))
                    messagebox.showinfo("Apply knobs", ascii_safe(str(reason)))

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def on_reset_knobs(self) -> None:
        if self._busy:
            return
        if not messagebox.askyesno(
            "Reset knobs",
            "Reset knobs_current.json to config defaults?",
        ):
            return
        try:
            reset_knobs_to_defaults(reason="gui_reset")
            self._refresh_calibration_pane(silent=True)
            self.status_var.set(
                f"Knobs reset. SCORE_SIGMA={effective_score_sigma():.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Reset knobs", ascii_safe(exc))

    def on_grade(self) -> None:
        if self._busy:
            return
        self._set_busy(True, "Grading stored predictions vs finals...")

        def work() -> None:
            err: Exception | None = None
            result: dict[str, Any] | None = None
            try:
                result = grade_predictions()
            except Exception as exc:  # noqa: BLE001
                err = exc
                logger.exception("Grade failed")

            def done() -> None:
                self._set_busy(False)
                if err:
                    msg = ascii_safe(err)
                    self.status_var.set(f"Grade failed: {msg}")
                    messagebox.showerror("Grade", msg)
                    return
                assert result is not None
                self._refresh_calibration_pane(silent=True)
                self.notebook.select(2)
                n = int((result.get("report") or {}).get("n_games") or 0)
                self.status_var.set(
                    f"Graded {n} games. Effective SCORE_SIGMA={effective_score_sigma():.2f}"
                )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

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
            self._write_pane(self.results, self._last_md)
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
