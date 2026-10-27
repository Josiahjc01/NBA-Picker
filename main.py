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
from data.process import build_features
from model.predict import predict_slate

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
        lines.append(
            f"- **Projected score:** {pred['away_abbr']} {pred['away_pts']} – "
            f"{pred['home_abbr']} {pred['home_pts']} "
            f"(total {pred['total']}, margin {pred['margin']:+.1f} home)"
        )
        lines.append(
            f"- **Win probability:** {pred['home_abbr']} {pred['home_win_prob']:.1%} · "
            f"{pred['away_abbr']} {pred['away_win_prob']:.1%} "
            f"(confidence {pred['confidence']:.2f})"
        )
        notes = pred.get("notes") or {}
        if notes.get("injuries"):
            lines.append("- **Injuries:** " + "; ".join(notes["injuries"]))
        if notes.get("rest_travel"):
            lines.append("- **Rest / travel:** " + ", ".join(notes["rest_travel"]))
        if notes.get("play_types"):
            lines.append("- **Play-type edges:** " + "; ".join(notes["play_types"]))
        lines.append("")

    return "\n".join(lines)


def print_terminal(target: date, predictions: list[dict]) -> None:
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
        print(f"  {pred['away_abbr']:3} @ {pred['home_abbr']:3}   "
              f"{pred['away_pts']:5.1f} – {pred['home_pts']:5.1f}   "
              f"P(home)={pred['home_win_prob']:.1%}  margin={pred['margin']:+.1f}")
        notes = pred.get("notes") or {}
        for label, key in (
            ("inj", "injuries"),
            ("ctx", "rest_travel"),
            ("pt", "play_types"),
        ):
            vals = notes.get(key) or []
            if vals:
                joined = ", ".join(vals) if isinstance(vals[0], str) else str(vals)
                print(f"      [{label}] {joined}")

    print()


def save_outputs(target: date, payload: dict, md: str) -> tuple[Path, Path]:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dated_md = config.OUTPUT_DIR / f"predictions_{target.isoformat()}.md"
    dated_json = config.OUTPUT_DIR / f"predictions_{target.isoformat()}.json"
    dated_md.write_text(md)
    dated_json.write_text(json.dumps(payload, indent=2, default=str))
    # Alias "today_*" only when targeting today (UTC)
    today = datetime.now(timezone.utc).date()
    if target == today:
        md_path = config.OUTPUT_DIR / "today_predictions.md"
        json_path = config.OUTPUT_DIR / "today_predictions.json"
        md_path.write_text(md)
        json_path.write_text(json.dumps(payload, indent=2, default=str))
        return md_path, json_path
    return dated_md, dated_json


def run_pipeline(
    target: date,
    force: bool = False,
    *,
    team_filter: set[str] | None = None,
    write_outputs: bool = True,
) -> dict:
    """
    Core pipeline used by CLI and GUI.

    Returns payload dict with keys: date, generated_at, meta, predictions,
    markdown, md_path, json_path.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = fetch_all(target, force=force)
    features = build_features(raw, target)
    predictions = predict_slate(features)

    if team_filter:
        wanted = {t.upper() for t in team_filter}
        predictions = [
            p
            for p in predictions
            if p.get("home_abbr", "").upper() in wanted
            or p.get("away_abbr", "").upper() in wanted
        ]

    meta = raw.get("meta") or {}
    payload = {
        "date": target.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "predictions": predictions,
    }
    md = format_markdown(target, predictions, meta)
    payload["markdown"] = md
    md_path = json_path = None
    if write_outputs:
        md_path, json_path = save_outputs(target, payload, md)
    payload["md_path"] = str(md_path) if md_path else None
    payload["json_path"] = str(json_path) if json_path else None
    return payload


def run(target: date, force: bool = False) -> int:
    payload = run_pipeline(target, force=force)
    predictions = payload["predictions"]
    print_terminal(target, predictions)
    if payload.get("md_path"):
        print(f"Wrote {payload['md_path']}")
    if payload.get("json_path"):
        print(f"Wrote {payload['json_path']}")
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
    try:
        return run(target, force=args.force_refresh)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
