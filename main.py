#!/usr/bin/env python3
"""NBA Picker — free on-demand game predictor (local personal use)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import config
from data.fetch import fetch_all
from data.process import build_features, verify_matchup_personnel
from model.brief import (
    enrich_predictions_with_brief,
    format_ai_paste,
    format_ai_paste_markdown,
)
from model.predict import predict_slate
from safe_text import ascii_safe, safe_print, write_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nba_picker")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NBA Picker — on-demand predictions")
    p.add_argument(
        "--date",
        dest="game_date",
        default=None,
        help="Slate date YYYY-MM-DD (default: today UTC)",
    )
    p.add_argument("--force-refresh", action="store_true", help="Ignore cache freshness")
    p.add_argument("-q", "--quiet", action="store_true", help="Less logging")
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop UI (same as python gui.py)",
    )
    p.add_argument("--away", default=None, help="Away team abbr (with --home filters one matchup)")
    p.add_argument("--home", default=None, help="Home team abbr (with --away filters one matchup)")
    return p.parse_args(argv)


def resolve_date(raw: str | None) -> date:
    if not raw:
        return datetime.now(timezone.utc).date()
    return date.fromisoformat(raw)


def format_markdown(
    target: date,
    predictions: list[dict],
    meta: dict,
) -> str:
    lines = [
        f"# NBA Picker — {target.isoformat()}",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"Games: **{len(predictions)}** · Injuries cached: {meta.get('injuries_count', '?')} · "
        f"Impact rows: {meta.get('impact_count', '?')}",
        "",
    ]
    if not predictions:
        lines += [
            "## No games",
            "",
            f"No NBA games found for **{target.isoformat()}**. "
            "Try another `--date YYYY-MM-DD` (season slate / schedule gap).",
            "",
        ]
        return "\n".join(lines)

    lines.append("## Game projections")
    lines.append("")
    for pred in predictions:
        lines.append(
            f"### {pred['away_abbr']} @ {pred['home_abbr']}"
        )
        lines.append("")
        nsim = pred.get("sim_count") or config.SIM_COUNT
        lines.append(
            f"- **Mean score ({nsim} sims):** {pred['away_abbr']} {pred['away_pts']} - "
            f"{pred['home_abbr']} {pred['home_pts']} "
            f"(total {pred['total']}, margin {pred['margin']:+.1f} home)"
        )
        lines.append(
            f"- **Win probability (sims):** {pred['home_abbr']} {pred['home_win_prob']:.1%} / "
            f"{pred['away_abbr']} {pred['away_win_prob']:.1%} "
            f"(confidence {pred['confidence']:.2f})"
        )
        sims = pred.get("sims") or {}
        if sims:
            lines.append(
                f"- **Score band p10-p90:** {pred['away_abbr']} "
                f"{sims.get('away_pts_p10')}-{sims.get('away_pts_p90')} / "
                f"{pred['home_abbr']} {sims.get('home_pts_p10')}-{sims.get('home_pts_p90')}"
            )
        notes = pred.get("notes") or {}
        tags = pred.get("form_tags") or []
        if tags:
            lines.append("- **Form tags:** " + ", ".join(tags))
        if notes.get("form"):
            lines.append("- **Hot/cold:** " + "; ".join(notes["form"]))
        if notes.get("quarters"):
            lines.append("- **Quarter edges:** " + "; ".join(notes["quarters"]))
        if pred.get("clutch_q4"):
            lines.append("- **Clutch Q4 weighting:** on (projected close)")
        if notes.get("injuries"):
            lines.append("- **Injuries:** " + "; ".join(notes["injuries"]))
        if notes.get("rest_travel"):
            lines.append("- **Rest / travel:** " + ", ".join(notes["rest_travel"]))
        if notes.get("play_types"):
            lines.append("- **Play-type edges:** " + "; ".join(notes["play_types"]))
        lines.append("")

    lines.append(format_ai_paste_markdown(target.isoformat(), predictions))
    return "\n".join(lines)


def _print_team_season(season: dict) -> None:
    abbr = season.get("team") or "?"
    win_pct = season.get("win_pct")
    avg_total = season.get("avg_total")
    w, l = season.get("wins"), season.get("losses")
    win_s = f"{win_pct:.1f}% ({w}-{l})" if win_pct is not None else "n/a"
    tot_s = f"{avg_total:.1f}" if avg_total is not None else "n/a"
    print(f"      {abbr}: win {win_s}  avg total {tot_s}")
    tops = season.get("top_players") or {}
    for stat, title in (
        ("PTS", "PTS"),
        ("REB", "REB"),
        ("AST", "AST"),
        ("STL", "STL"),
        ("BLK", "BLK"),
    ):
        leaders = tops.get(stat) or []
        if not leaders:
            continue
        joined = ", ".join(f"{x['player']} {x['value']}" for x in leaders)
        print(f"        {title}: {joined}")


def _print_personnel(personnel: dict | None) -> None:
    if not personnel:
        return
    print()
    print("-" * 64)
    print(" PERSONNEL CHECK (roster / injuries / minutes)")
    print("-" * 64)
    for side in ("away", "home"):
        s = personnel.get(side) or {}
        if not s or s.get("error"):
            continue
        print(f"  {s.get('team')}: roster {s.get('roster_size')}")
        for p in (s.get("key_players") or [])[:5]:
            flag = f"  [{p['injury']}]" if p.get("injury") else ""
            safe_print(
                f"    {p['player']:22}  {p['proj_min']:5.1f} -> {p['adj_min']:5.1f} min{flag}"
            )
        for p in (s.get("injury_flags") or [])[:4]:
            if p.get("player") in {k["player"] for k in (s.get("key_players") or [])[:5]}:
                continue
            safe_print(
                f"    ! {p['player']:20}  {p.get('status')}  "
                f"{p['proj_min']:.0f}->{p['adj_min']:.0f}"
            )


def print_terminal(
    target: date,
    predictions: list[dict],
    ai_paste: str | None = None,
    personnel: dict | None = None,
) -> None:
    print()
    print("=" * 64)
    print(f" NBA PICKER  ·  {target.isoformat()}")
    print("=" * 64)
    if not predictions:
        print()
        print(f"No NBA games scheduled for {target.isoformat()}.")
        print("Try:  python main.py --date YYYY-MM-DD")
        print()
        return

    for pred in predictions:
        print()
        nsim = pred.get("sim_count") or config.SIM_COUNT
        print(f"  {pred['away_abbr']:3} @ {pred['home_abbr']:3}   "
              f"{pred['away_pts']:5.1f} - {pred['home_pts']:5.1f}   "
              f"P(home)={pred['home_win_prob']:.1%}  margin={pred['margin']:+.1f}  "
              f"[{nsim} sims]")
        notes = pred.get("notes") or {}
        tags = pred.get("form_tags") or []
        if tags:
            safe_print(f"      [form-tag] {ascii_safe(', '.join(tags))}")
        if pred.get("clutch_q4"):
            safe_print("      [clutch] Q4 weighting ON")
        for label, key in (
            ("form", "form"),
            ("qtr", "quarters"),
            ("inj", "injuries"),
            ("ctx", "rest_travel"),
            ("pt", "play_types"),
        ):
            vals = notes.get(key) or []
            if vals:
                joined = ", ".join(vals) if isinstance(vals[0], str) else str(vals)
                safe_print(f"      [{label}] {ascii_safe(joined)}")
        season = pred.get("season") or {}
        for side in ("away", "home"):
            if season.get(side):
                _print_team_season(season[side])

    _print_personnel(personnel)
    if ai_paste:
        print()
        print("-" * 64)
        print(" AI PASTE (also written to output/ai_paste_*.txt)")
        print("-" * 64)
        print("  Copy the AI paste file into another AI for wager ideas.")
    print()


def save_outputs(
    target: date,
    payload: dict,
    md: str,
    ai_paste: str,
) -> tuple[Path, Path, Path]:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dated_md = config.OUTPUT_DIR / f"predictions_{target.isoformat()}.md"
    dated_json = config.OUTPUT_DIR / f"predictions_{target.isoformat()}.json"
    dated_ai = config.OUTPUT_DIR / f"ai_paste_{target.isoformat()}.txt"
    write_text(dated_md, md)
    write_text(dated_json, json.dumps(payload, indent=2, default=str))
    write_text(dated_ai, ai_paste)
    today = datetime.now(timezone.utc).date()
    if target == today:
        md_path = config.OUTPUT_DIR / "today_predictions.md"
        json_path = config.OUTPUT_DIR / "today_predictions.json"
        ai_path = config.OUTPUT_DIR / "today_ai_paste.txt"
        write_text(md_path, md)
        write_text(json_path, json.dumps(payload, indent=2, default=str))
        write_text(ai_path, ai_paste)
        return md_path, json_path, ai_path
    return dated_md, dated_json, dated_ai


def run_pipeline(
    target: date,
    force: bool = False,
    *,
    team_filter: set[str] | None = None,
    matchup: tuple[str, str] | None = None,
    write_outputs: bool = True,
) -> dict:
    """
    Core pipeline used by CLI and GUI.

    Returns payload with predictions (incl. season/top players), ai_paste,
    markdown, and output paths.

    ``matchup`` is ``(away_abbr, home_abbr)`` and takes precedence over
    ``team_filter`` when set.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = fetch_all(target, force=force)
    features = build_features(raw, target)
    predictions = enrich_predictions_with_brief(predict_slate(features), features)

    personnel = None
    if matchup:
        away_m = matchup[0].upper()
        home_m = matchup[1].upper()
        personnel = verify_matchup_personnel(features, away_m, home_m)
        predictions = [
            p
            for p in predictions
            if p.get("away_abbr", "").upper() == away_m
            and p.get("home_abbr", "").upper() == home_m
        ]
    elif team_filter:
        wanted = {t.upper() for t in team_filter}
        predictions = [
            p
            for p in predictions
            if p.get("home_abbr", "").upper() in wanted
            or p.get("away_abbr", "").upper() in wanted
        ]

    meta = raw.get("meta") or {}
    if personnel:
        meta = {**meta, "personnel_verify": personnel}
    ai_paste = format_ai_paste(target.isoformat(), predictions)
    payload = {
        "date": target.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "predictions": predictions,
        "ai_paste": ai_paste,
        "personnel_verify": personnel,
    }
    md = format_markdown(target, predictions, meta)
    payload["markdown"] = md
    md_path = json_path = ai_path = None
    if write_outputs:
        md_path, json_path, ai_path = save_outputs(target, payload, md, ai_paste)
    payload["md_path"] = str(md_path) if md_path else None
    payload["json_path"] = str(json_path) if json_path else None
    payload["ai_paste_path"] = str(ai_path) if ai_path else None
    return payload


def run(
    target: date,
    force: bool = False,
    *,
    matchup: tuple[str, str] | None = None,
) -> int:
    payload = run_pipeline(target, force=force, matchup=matchup)
    predictions = payload["predictions"]
    print_terminal(
        target,
        predictions,
        payload.get("ai_paste"),
        payload.get("personnel_verify"),
    )
    if payload.get("md_path"):
        print(f"Wrote {payload['md_path']}")
    if payload.get("json_path"):
        print(f"Wrote {payload['json_path']}")
    if payload.get("ai_paste_path"):
        print(f"Wrote {payload['ai_paste_path']}  <- paste this into an AI for wager ideas")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gui:
        from gui import launch_gui

        return launch_gui()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    try:
        target = resolve_date(args.game_date)
    except ValueError:
        print(f"Invalid --date: {args.game_date!r} (use YYYY-MM-DD)", file=sys.stderr)
        return 2
    matchup = None
    if args.away or args.home:
        if not (args.away and args.home):
            print("Provide both --away and --home", file=sys.stderr)
            return 2
        matchup = (args.away.upper(), args.home.upper())
    try:
        return run(target, force=args.force_refresh, matchup=matchup)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
