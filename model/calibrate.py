"""Grade finalized games vs stored predictions; calibrate MC sigmas."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config
from data.static_arenas import normalize_abbrev
from safe_text import ascii_safe, write_text

logger = logging.getLogger(__name__)


def _grades_dir() -> Path:
    d = Path(config.GRADES_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def calibration_state_path() -> Path:
    return config.CACHE_DIR / "calibration_state.json"


def load_calibration_state() -> dict[str, Any]:
    path = calibration_state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read calibration state: %s", exc)
        return {}


def save_calibration_state(state: dict[str, Any]) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = calibration_state_path()
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def effective_score_sigma() -> float:
    """SCORE_SIGMA from learning knobs when present, else capped calibration blend."""
    try:
        from data.learning_store import apply_learning_knobs, load_knobs

        apply_learning_knobs()
        knobs = load_knobs()
        if knobs.get("SCORE_SIGMA") is not None:
            return float(config.SCORE_SIGMA)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Learning knobs unavailable for SCORE_SIGMA: %s", exc)

    base = float(config.SCORE_SIGMA)
    if not config.CALIBRATION_AUTO_APPLY:
        return base
    state = load_calibration_state()
    n = int(state.get("n_games") or 0)
    if n < config.CALIBRATION_MIN_GAMES:
        return base
    suggested = state.get("suggested_score_sigma")
    try:
        sig = float(suggested)
    except (TypeError, ValueError):
        return base
    if not np.isfinite(sig):
        return base
    # Gentle blend toward suggested (40%) so we don't overfit
    blended = 0.60 * base + 0.40 * sig
    return float(
        np.clip(blended, config.CALIBRATION_SIGMA_MIN, config.CALIBRATION_SIGMA_MAX)
    )


def effective_margin_sigma() -> float:
    """MARGIN_SIGMA from learning knobs when present, else config default."""
    try:
        from data.learning_store import apply_learning_knobs, load_knobs

        apply_learning_knobs()
        knobs = load_knobs()
        if knobs.get("MARGIN_SIGMA") is not None:
            return float(config.MARGIN_SIGMA)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Learning knobs unavailable for MARGIN_SIGMA: %s", exc)
    return float(config.MARGIN_SIGMA)


def _list_prediction_files() -> list[Path]:
    out_dir = Path(config.OUTPUT_DIR)
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("predictions_*.json"))
    today = out_dir / "today_predictions.json"
    if today.exists() and today not in files:
        files.append(today)
    return files


def _fetch_final_score(game_date: date, away: str, home: str) -> dict[str, Any] | None:
    """Pull final scores from ESPN scoreboard for Away @ Home on date."""
    try:
        from data.espn import scoreboard
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN import failed: %s", exc)
        return None
    try:
        games = scoreboard(game_date) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scoreboard %s failed: %s", game_date, exc)
        return None
    away_u = normalize_abbrev(away)
    home_u = normalize_abbrev(home)
    for g in games:
        ga = normalize_abbrev(str(g.get("away_abbr") or ""))
        gh = normalize_abbrev(str(g.get("home_abbr") or ""))
        if ga != away_u or gh != home_u:
            continue
        status = str(g.get("status") or "").lower()
        # Prefer completed
        try:
            away_pts = float(g.get("away_score") if g.get("away_score") is not None else g.get("away_pts"))
        except (TypeError, ValueError):
            away_pts = None
        try:
            home_pts = float(g.get("home_score") if g.get("home_score") is not None else g.get("home_pts"))
        except (TypeError, ValueError):
            home_pts = None
        # scoreboard() may not include scores — try summary via competitors if present
        if away_pts is None or home_pts is None:
            continue
        if not np.isfinite(away_pts) or not np.isfinite(home_pts):
            continue
        return {
            "away_pts": away_pts,
            "home_pts": home_pts,
            "total": away_pts + home_pts,
            "margin": home_pts - away_pts,
            "status": status,
            "game_id": g.get("game_id"),
        }
    return None


def _enrich_scoreboard_scores(game_date: date) -> list[dict[str, Any]]:
    """Fetch scoreboard JSON and parse competitor scores (finals only)."""
    from data import espn

    stamp = game_date.strftime("%Y%m%d")
    try:
        data = espn._get(f"{espn.SITE}/scoreboard", params={"dates": stamp})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scoreboard raw %s failed: %s", game_date, exc)
        return []
    rows: list[dict[str, Any]] = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [None])[0]
        if not comp:
            continue
        status_name = ((comp.get("status") or {}).get("type") or {}).get("name") or ""
        if status_name != "STATUS_FINAL":
            continue
        home = away = None
        for c in comp.get("competitors") or []:
            team = c.get("team") or {}
            abbr = normalize_abbrev(team.get("abbreviation"))
            sc = c.get("score")
            try:
                pts = float(sc.get("value") if isinstance(sc, dict) else sc)
            except (TypeError, ValueError):
                pts = None
            entry = {"abbr": abbr, "pts": pts}
            if c.get("homeAway") == "home":
                home = entry
            else:
                away = entry
        if not home or not away or home["pts"] is None or away["pts"] is None:
            continue
        rows.append(
            {
                "game_id": str(ev.get("id") or ""),
                "away_abbr": away["abbr"],
                "home_abbr": home["abbr"],
                "away_pts": float(away["pts"]),
                "home_pts": float(home["pts"]),
                "total": float(away["pts"]) + float(home["pts"]),
                "margin": float(home["pts"]) - float(away["pts"]),
            }
        )
    return rows


def grade_predictions(
    *,
    force: bool = False,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """
    Compare stored prediction means/win% to finalized ESPN scores.
    Writes grade rows under output/grades/ and refreshes calibration_state.json.
    """
    from datetime import timedelta

    pred_files = _list_prediction_files()
    graded: list[dict[str, Any]] = []
    seen: set[str] = set()
    cutoff = None
    if lookback_days is not None:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=int(lookback_days))

    finals_cache: dict[str, list[dict[str, Any]]] = {}

    for path in pred_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip %s: %s", path, exc)
            continue
        d_raw = payload.get("date") or ""
        try:
            d = date.fromisoformat(str(d_raw)[:10])
        except ValueError:
            continue
        if cutoff and d < cutoff:
            continue
        # Only grade past dates (or today if finals exist)
        today = datetime.now(timezone.utc).date()
        if d > today:
            continue
        key_d = d.isoformat()
        if key_d not in finals_cache:
            finals_cache[key_d] = _enrich_scoreboard_scores(d)
        finals = {
            (normalize_abbrev(r["away_abbr"]), normalize_abbrev(r["home_abbr"])): r
            for r in finals_cache[key_d]
        }
        for pred in payload.get("predictions") or []:
            away = normalize_abbrev(pred.get("away_abbr"))
            home = normalize_abbrev(pred.get("home_abbr"))
            ukey = f"{d.isoformat()}:{away}@{home}"
            if ukey in seen:
                continue
            actual = finals.get((away, home))
            if not actual:
                continue
            seen.add(ukey)
            try:
                pred_a = float(pred.get("away_pts"))
                pred_h = float(pred.get("home_pts"))
                p_home = float(pred.get("home_win_prob") or 0.5)
            except (TypeError, ValueError):
                continue
            act_a = float(actual["away_pts"])
            act_h = float(actual["home_pts"])
            home_won = 1.0 if act_h > act_a else (0.0 if act_h < act_a else 0.5)
            row = {
                "date": d.isoformat(),
                "away_abbr": away,
                "home_abbr": home,
                "pred_away_pts": round(pred_a, 1),
                "pred_home_pts": round(pred_h, 1),
                "pred_total": round(pred_a + pred_h, 1),
                "pred_margin": round(pred_h - pred_a, 1),
                "pred_home_win_prob": round(p_home, 4),
                "act_away_pts": act_a,
                "act_home_pts": act_h,
                "act_total": act_a + act_h,
                "act_margin": act_h - act_a,
                "abs_err_away": round(abs(pred_a - act_a), 2),
                "abs_err_home": round(abs(pred_h - act_h), 2),
                "abs_err_total": round(abs((pred_a + pred_h) - (act_a + act_h)), 2),
                "abs_err_margin": round(abs((pred_h - pred_a) - (act_h - act_a)), 2),
                "sq_err_margin": round(((pred_h - pred_a) - (act_h - act_a)) ** 2, 4),
                "brier": round((p_home - home_won) ** 2, 6),
                "home_won": home_won,
                "sim_count": pred.get("sim_count"),
                "source_file": str(path.name),
            }
            graded.append(row)

    report = summarize_grades(graded)
    # Persist
    grades_path = _grades_dir() / "graded_games.json"
    write_text(grades_path, json.dumps(graded, indent=2, default=str))
    report_path = _grades_dir() / "calibration_report.json"
    write_text(report_path, json.dumps(report, indent=2, default=str))
    report_txt = format_calibration_report(report)
    write_text(_grades_dir() / "calibration_report.txt", report_txt)

    state = {
        **report,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": report.get("n_games", 0),
        "suggested_score_sigma": report.get("suggested_score_sigma"),
        "suggested_margin_sigma": report.get("suggested_margin_sigma"),
        "grades_path": str(grades_path),
    }
    save_calibration_state(state)

    # Sync into offline learning store + capped auto-knobs (paste path shares this).
    learning_info: dict[str, Any] = {}
    try:
        from data.learning_store import (
            auto_update_knobs,
            format_learning_summary,
            sync_grades_from_list,
        )

        added = sync_grades_from_list(graded, source="espn_grade")
        knob_result = auto_update_knobs(reason="espn_grade")
        learning_info = {
            "synced": added,
            "knobs": knob_result,
            "summary_text": format_learning_summary(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Learning store sync failed: %s", exc)

    logger.info(
        "Graded %s games; MAE total=%.2f margin=%.2f Brier=%.4f suggested sigma=%.2f",
        report.get("n_games"),
        report.get("mae_total") or 0,
        report.get("mae_margin") or 0,
        report.get("brier") or 0,
        report.get("suggested_score_sigma") or config.SCORE_SIGMA,
    )
    return {
        "grades": graded,
        "report": report,
        "text": report_txt,
        "state": state,
        "learning": learning_info,
    }


def summarize_grades(graded: list[dict[str, Any]]) -> dict[str, Any]:
    if not graded:
        return {
            "n_games": 0,
            "mae_total": None,
            "mae_margin": None,
            "mae_team_score": None,
            "rmse_margin": None,
            "brier": None,
            "suggested_score_sigma": float(config.SCORE_SIGMA),
            "suggested_margin_sigma": float(config.MARGIN_SIGMA),
            "note": "No graded finals yet. Run projections, wait for games to finish, then grade.",
        }
    n = len(graded)
    mae_total = float(np.mean([g["abs_err_total"] for g in graded]))
    mae_margin = float(np.mean([g["abs_err_margin"] for g in graded]))
    mae_team = float(
        np.mean(
            [0.5 * (g["abs_err_away"] + g["abs_err_home"]) for g in graded]
        )
    )
    rmse_margin = float(math.sqrt(np.mean([g["sq_err_margin"] for g in graded])))
    brier = float(np.mean([g["brier"] for g in graded]))
    # Empirically, team score SD ≈ RMSE of team scores; use mae*1.25 as soft proxy
    # Prefer RMSE of (pred - act) team scores
    team_errs = []
    for g in graded:
        team_errs.append(g["pred_away_pts"] - g["act_away_pts"])
        team_errs.append(g["pred_home_pts"] - g["act_home_pts"])
    score_rmse = float(np.sqrt(np.mean(np.square(team_errs)))) if team_errs else float(config.SCORE_SIGMA)
    # Suggest sigma between clamps; don't leap from tiny samples
    suggested = float(
        np.clip(score_rmse, config.CALIBRATION_SIGMA_MIN, config.CALIBRATION_SIGMA_MAX)
    )
    suggested_margin = float(
        np.clip(rmse_margin, config.CALIBRATION_SIGMA_MIN, config.CALIBRATION_SIGMA_MAX + 2)
    )
    # Reliability buckets for win%
    buckets = []
    for lo, hi in ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.75), (0.75, 1.01)):
        rows = [g for g in graded if lo <= float(g["pred_home_win_prob"]) < hi]
        if not rows:
            continue
        buckets.append(
            {
                "p_home_range": f"{lo:.2f}-{hi:.2f}",
                "n": len(rows),
                "avg_pred": round(float(np.mean([g["pred_home_win_prob"] for g in rows])), 3),
                "avg_actual": round(float(np.mean([g["home_won"] for g in rows])), 3),
            }
        )
    return {
        "n_games": n,
        "mae_total": round(mae_total, 2),
        "mae_margin": round(mae_margin, 2),
        "mae_team_score": round(mae_team, 2),
        "rmse_margin": round(rmse_margin, 2),
        "rmse_team_score": round(score_rmse, 2),
        "brier": round(brier, 4),
        "suggested_score_sigma": round(suggested, 2),
        "suggested_margin_sigma": round(suggested_margin, 2),
        "effective_score_sigma": round(effective_score_sigma(), 2),
        "auto_apply": bool(config.CALIBRATION_AUTO_APPLY),
        "min_games_for_auto": config.CALIBRATION_MIN_GAMES,
        "win_prob_buckets": buckets,
        "base_score_sigma": float(config.SCORE_SIGMA),
    }


def format_calibration_report(report: dict[str, Any] | None = None) -> str:
    report = report or load_calibration_state() or summarize_grades([])
    lines = [
        "NBA PICKER — CALIBRATION REPORT",
        "=" * 64,
        "",
        f"Graded games: {report.get('n_games', 0)}",
        f"MAE total: {report.get('mae_total')}",
        f"MAE margin: {report.get('mae_margin')}",
        f"MAE team score: {report.get('mae_team_score')}",
        f"RMSE margin: {report.get('rmse_margin')}",
        f"RMSE team score: {report.get('rmse_team_score')}",
        f"Brier (home win%): {report.get('brier')}",
        "",
        f"Base SCORE_SIGMA: {report.get('base_score_sigma', config.SCORE_SIGMA)}",
        f"Suggested SCORE_SIGMA: {report.get('suggested_score_sigma')}",
        f"Effective SCORE_SIGMA: {report.get('effective_score_sigma', effective_score_sigma())}",
        f"Auto-apply: {report.get('auto_apply', config.CALIBRATION_AUTO_APPLY)} "
        f"(needs >= {report.get('min_games_for_auto', config.CALIBRATION_MIN_GAMES)} games)",
        "",
    ]
    note = report.get("note")
    if note:
        lines.append(ascii_safe(str(note)))
        lines.append("")
    buckets = report.get("win_prob_buckets") or []
    if buckets:
        lines.append("Win% calibration buckets (pred home win -> actual):")
        for b in buckets:
            lines.append(
                f"  {b['p_home_range']}: n={b['n']}  "
                f"pred={b['avg_pred']:.3f}  actual={b['avg_actual']:.3f}"
            )
        lines.append("")
    lines.append(
        "Grading uses projection MEANS and win% — not Highlights sim-maxes."
    )
    lines.append("Run: python main.py --grade   or use the Calibration tab in the GUI.")
    lines.append("")
    return "\n".join(lines)
