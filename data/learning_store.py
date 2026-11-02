"""Offline learning store: paste grades, capped knobs, audit trail.

Layout under cache/learning/ (gitignored via cache/*):
  manifest.json, graded_games.jsonl, feature_residuals.jsonl,
  knob_history.jsonl, knobs_current.json, paste_inbox/, paste_applied.jsonl,
  suggestions_inbox/, applied_suggestions.jsonl

Projections only — no odds. Code themes never auto-apply.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config
from data.static_arenas import normalize_abbrev
from safe_text import ascii_safe, write_text

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SAFE_KNOB_KEYS = (
    "SCORE_SIGMA",
    "MARGIN_SIGMA",
    "HOME_COURT_POINTS",
    "REST_B2B_POINTS",
    "REST_3IN4_POINTS",
    "REST_4IN6_POINTS",
    "ALTITUDE_HOME_POINTS",
)

# Snapshot of config defaults at import (env overrides included) for Reset.
_CONFIG_DEFAULTS: dict[str, float] = {
    k: float(getattr(config, k)) for k in SAFE_KNOB_KEYS
}


def learning_dir() -> Path:
    d = Path(config.LEARNING_DIR)
    d.mkdir(parents=True, exist_ok=True)
    (d / "paste_inbox").mkdir(parents=True, exist_ok=True)
    (d / "suggestions_inbox").mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_for_file() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def manifest_path() -> Path:
    return learning_dir() / "manifest.json"


def knobs_path() -> Path:
    return learning_dir() / "knobs_current.json"


def graded_path() -> Path:
    return learning_dir() / "graded_games.jsonl"


def residuals_path() -> Path:
    return learning_dir() / "feature_residuals.jsonl"


def knob_history_path() -> Path:
    return learning_dir() / "knob_history.jsonl"


def paste_applied_path() -> Path:
    return learning_dir() / "paste_applied.jsonl"


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": None,
            "n_graded": 0,
            "n_pastes": 0,
            "n_suggestions_pending": 0,
            "knobs_revision": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read learning manifest: %s", exc)
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": None,
            "n_graded": 0,
            "n_pastes": 0,
            "n_suggestions_pending": 0,
            "knobs_revision": 0,
        }


def save_manifest(mani: dict[str, Any]) -> Path:
    mani = {**mani, "schema_version": SCHEMA_VERSION, "updated_at": _now_iso()}
    path = manifest_path()
    write_text(path, json.dumps(mani, indent=2, default=str) + "\n")
    return path


def refresh_manifest_counts() -> dict[str, Any]:
    mani = load_manifest()
    graded = _read_jsonl(graded_path())
    pastes = _read_jsonl(paste_applied_path())
    inbox = learning_dir() / "suggestions_inbox"
    pending = len(list(inbox.glob("*.json"))) if inbox.exists() else 0
    mani["n_graded"] = len(graded)
    mani["n_pastes"] = len(pastes)
    mani["n_suggestions_pending"] = pending
    save_manifest(mani)
    return mani


def config_defaults() -> dict[str, float]:
    return dict(_CONFIG_DEFAULTS)


def load_knobs() -> dict[str, Any]:
    path = knobs_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read knobs_current: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _clip_knob(key: str, value: float) -> float:
    caps = config.LEARNING_KNOB_CAPS.get(key) or getattr(config, "LEARNING_KNOB_CAPS", {}).get(key)
    if caps is None and key.startswith("REST_"):
        caps = config.LEARNING_KNOB_CAPS.get("REST_B2B_POINTS")
    if caps is None:
        return float(value)
    lo, hi = caps
    return float(np.clip(float(value), lo, hi))


def save_knobs(
    knobs: dict[str, Any],
    *,
    n_games_basis: int | None = None,
    reason: str = "update",
) -> Path:
    """Write knobs_current.json (safe keys only) and bump manifest revision."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "n_games_basis": int(
            n_games_basis
            if n_games_basis is not None
            else knobs.get("n_games_basis") or 0
        ),
    }
    for key in SAFE_KNOB_KEYS:
        if key in knobs and knobs[key] is not None:
            try:
                out[key] = round(_clip_knob(key, float(knobs[key])), 4)
            except (TypeError, ValueError):
                continue
    team_biases = knobs.get("team_biases") or {}
    if isinstance(team_biases, dict) and team_biases:
        cleaned: dict[str, float] = {}
        for abbr, bias in team_biases.items():
            try:
                b = float(bias)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(b):
                continue
            cleaned[normalize_abbrev(str(abbr))] = round(
                float(np.clip(b, -config.LEARNING_TEAM_BIAS_CAP, config.LEARNING_TEAM_BIAS_CAP)),
                4,
            )
        if cleaned:
            out["team_biases"] = cleaned
    caps_out = {k: list(v) for k, v in config.LEARNING_KNOB_CAPS.items()}
    out["caps"] = caps_out
    path = knobs_path()
    write_text(path, json.dumps(out, indent=2, default=str) + "\n")
    mani = load_manifest()
    mani["knobs_revision"] = int(mani.get("knobs_revision") or 0) + 1
    save_manifest(mani)
    logger.info("Saved learning knobs (%s) revision=%s", reason, mani["knobs_revision"])
    return path


def apply_learning_knobs() -> dict[str, Any]:
    """
    Overlay knobs_current.json onto config module attributes.
    Always starts from import-time defaults so Reset + re-apply is idempotent.
    """
    for key, val in _CONFIG_DEFAULTS.items():
        setattr(config, key, float(val))
    knobs = load_knobs()
    if not knobs:
        return {}
    applied: dict[str, float] = {}
    for key in SAFE_KNOB_KEYS:
        if key not in knobs:
            continue
        try:
            val = _clip_knob(key, float(knobs[key]))
        except (TypeError, ValueError):
            continue
        setattr(config, key, val)
        applied[key] = val
    if applied:
        logger.debug("Applied learning knobs: %s", applied)
    return knobs


def team_bias_for(abbr: str) -> float:
    knobs = load_knobs()
    biases = knobs.get("team_biases") or {}
    if not isinstance(biases, dict):
        return 0.0
    try:
        return float(biases.get(normalize_abbrev(abbr)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def reset_knobs_to_defaults(*, reason: str = "user_reset") -> dict[str, Any]:
    """Restore knobs_current.json to config defaults; audit the change."""
    before = load_knobs()
    defaults = config_defaults()
    payload = {
        **defaults,
        "n_games_basis": 0,
        "team_biases": {},
    }
    save_knobs(payload, n_games_basis=0, reason=reason)
    _append_jsonl(
        knob_history_path(),
        {
            "at": _now_iso(),
            "action": "reset",
            "reason": reason,
            "before": {k: before.get(k) for k in SAFE_KNOB_KEYS if k in before},
            "after": defaults,
        },
    )
    apply_learning_knobs()
    return load_knobs()


def step_toward(current: float, target: float, frac: float | None = None) -> float:
    frac = float(config.LEARNING_STEP_FRAC if frac is None else frac)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(current + frac * (target - current))


def _record_knob_change(
    *,
    action: str,
    changes: dict[str, dict[str, float]],
    n_games: int,
    reason: str,
) -> None:
    if not changes:
        return
    _append_jsonl(
        knob_history_path(),
        {
            "at": _now_iso(),
            "action": action,
            "reason": reason,
            "n_games": n_games,
            "changes": changes,
        },
    )


def build_grade_row(
    *,
    game_date: str,
    away_abbr: str,
    home_abbr: str,
    pred_away_pts: float,
    pred_home_pts: float,
    pred_home_win_prob: float,
    act_away_pts: float,
    act_home_pts: float,
    sim_count: Any = None,
    source: str = "user_paste",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    away = normalize_abbrev(away_abbr)
    home = normalize_abbrev(home_abbr)
    pred_a = float(pred_away_pts)
    pred_h = float(pred_home_pts)
    act_a = float(act_away_pts)
    act_h = float(act_home_pts)
    p_home = float(pred_home_win_prob)
    home_won = 1.0 if act_h > act_a else (0.0 if act_h < act_a else 0.5)
    return {
        "date": str(game_date)[:10],
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
        "sim_count": sim_count,
        "source": source,
        "notes": notes or [],
    }


def build_residual_row(
    grade: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feats = {
        "home_court_points": float(config.HOME_COURT_POINTS),
        "b2b_home": False,
        "b2b_away": False,
        "rest_flag_3in4_home": False,
        "rest_flag_3in4_away": False,
        "rest_flag_4in6_home": False,
        "rest_flag_4in6_away": False,
        "altitude": normalize_abbrev(grade.get("home_abbr")) in config.ALTITUDE_TEAMS,
        "injury_delta_home": 0.0,
        "injury_delta_away": 0.0,
        "form_edge_home": 0.0,
        "form_edge_away": 0.0,
        "sample_weight": 1.0,
        "effective_score_sigma": float(config.SCORE_SIGMA),
    }
    if features:
        for k, v in features.items():
            feats[k] = v
    return {
        "date": grade.get("date"),
        "away_abbr": grade.get("away_abbr"),
        "home_abbr": grade.get("home_abbr"),
        "residual_margin_home": round(
            float(grade["act_margin"]) - float(grade["pred_margin"]), 2
        ),
        "residual_total": round(
            float(grade["act_total"]) - float(grade["pred_total"]), 2
        ),
        "features": feats,
    }


def append_graded_game(
    grade: dict[str, Any],
    *,
    features: dict[str, Any] | None = None,
) -> None:
    learning_dir()
    _append_jsonl(graded_path(), grade)
    _append_jsonl(residuals_path(), build_residual_row(grade, features))
    refresh_manifest_counts()


def sync_grades_from_list(
    grades: list[dict[str, Any]],
    *,
    source: str = "espn_grade",
) -> int:
    """Append ESPN/calibration grades that are not already in learning jsonl."""
    existing = {
        f"{g.get('date')}:{g.get('away_abbr')}@{g.get('home_abbr')}"
        for g in _read_jsonl(graded_path())
    }
    added = 0
    for g in grades:
        key = f"{g.get('date')}:{g.get('away_abbr')}@{g.get('home_abbr')}"
        if key in existing:
            continue
        row = dict(g)
        row.setdefault("source", source)
        row.setdefault("notes", [])
        append_graded_game(row)
        existing.add(key)
        added += 1
    return added


def lookup_paired_prediction(
    game_date: str,
    away_abbr: str,
    home_abbr: str,
) -> dict[str, Any] | None:
    """Find pred means/win% from output/predictions_*.json for a matchup."""
    away = normalize_abbrev(away_abbr)
    home = normalize_abbrev(home_abbr)
    d = str(game_date)[:10]
    candidates = [
        Path(config.OUTPUT_DIR) / f"predictions_{d}.json",
        Path(config.OUTPUT_DIR) / "today_predictions.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if str(payload.get("date") or "")[:10] not in ("", d) and path.name.startswith(
            "predictions_"
        ):
            # dated file must match
            if str(payload.get("date") or "")[:10] != d:
                continue
        for pred in payload.get("predictions") or []:
            if normalize_abbrev(pred.get("away_abbr")) != away:
                continue
            if normalize_abbrev(pred.get("home_abbr")) != home:
                continue
            try:
                return {
                    "pred_away_pts": float(pred.get("away_pts")),
                    "pred_home_pts": float(pred.get("home_pts")),
                    "pred_home_win_prob": float(pred.get("home_win_prob") or 0.5),
                    "sim_count": pred.get("sim_count") or payload.get("sim_count"),
                }
            except (TypeError, ValueError):
                continue
    return None


def parse_ascii_paste(text: str) -> dict[str, Any] | None:
    """Parse minimal ASCII block into a real_box-like dict, or None."""
    raw = ascii_safe(text or "").strip()
    if not raw:
        return None
    # Prefer JSON if the whole blob is JSON
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("kind") in (
                "real_box",
                "ai_comparison",
            ):
                return obj
        except json.JSONDecodeError:
            pass
    date_m = re.search(r"DATE\s+(\d{4}-\d{2}-\d{2})", raw, re.I)
    match_m = re.search(r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b", raw)
    act_m = re.search(r"ACT\s+([\d.]+)\s*[-–]\s*([\d.]+)", raw, re.I)
    pred_m = re.search(r"PRED\s+([\d.]+)\s*[-–]\s*([\d.]+)", raw, re.I)
    ph_m = re.search(r"P\(home\)\s*=\s*([\d.]+)", raw, re.I)
    if not (date_m and match_m and act_m):
        return None
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "real_box",
        "date": date_m.group(1),
        "away_abbr": normalize_abbrev(match_m.group(1)),
        "home_abbr": normalize_abbrev(match_m.group(2)),
        "act_away_pts": float(act_m.group(1)),
        "act_home_pts": float(act_m.group(2)),
        "notes": "ascii paste",
    }
    if pred_m:
        paired: dict[str, Any] = {
            "pred_away_pts": float(pred_m.group(1)),
            "pred_home_pts": float(pred_m.group(2)),
            "pred_home_win_prob": float(ph_m.group(1)) if ph_m else 0.5,
        }
        out["paired_prediction"] = paired
    return out


def _archive_paste(raw_obj: dict[str, Any] | str, kind: str) -> Path:
    inbox = learning_dir() / "paste_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"{_stamp_for_file()}_{kind}.json"
    if isinstance(raw_obj, str):
        payload = {"raw_text": raw_obj, "kind": kind, "archived_at": _now_iso()}
    else:
        payload = {**raw_obj, "archived_at": _now_iso()}
    write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    return path


def _queue_code_themes(
    themes: list[Any],
    *,
    date_s: str,
    matchup: str,
    source: str,
) -> Path | None:
    clean = [ascii_safe(str(t)).strip() for t in (themes or []) if str(t).strip()]
    if not clean:
        return None
    inbox = learning_dir() / "suggestions_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"{_stamp_for_file()}_code_themes.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "code_change",
        "date": date_s,
        "matchup": matchup,
        "source": source,
        "suggested_code_themes": clean,
        "status": "pending",
        "note": "Never auto-applied as code. Review and open a PR if useful.",
        "disclaimer": "Projections only. No betting automation.",
    }
    write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    refresh_manifest_counts()
    return path


def summarize_learning_grades(grades: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    grades = grades if grades is not None else _read_jsonl(graded_path())
    if not grades:
        return {
            "n_games": 0,
            "mae_total": None,
            "mae_margin": None,
            "brier": None,
            "rmse_team_score": None,
            "rmse_margin": None,
        }
    n = len(grades)
    mae_total = float(np.mean([g["abs_err_total"] for g in grades]))
    mae_margin = float(np.mean([g["abs_err_margin"] for g in grades]))
    brier = float(np.mean([g["brier"] for g in grades]))
    team_errs = []
    for g in grades:
        team_errs.append(float(g["pred_away_pts"]) - float(g["act_away_pts"]))
        team_errs.append(float(g["pred_home_pts"]) - float(g["act_home_pts"]))
    score_rmse = float(np.sqrt(np.mean(np.square(team_errs)))) if team_errs else float(
        config.SCORE_SIGMA
    )
    rmse_margin = float(
        math.sqrt(np.mean([float(g.get("sq_err_margin") or g["abs_err_margin"] ** 2) for g in grades]))
    )
    return {
        "n_games": n,
        "mae_total": round(mae_total, 2),
        "mae_margin": round(mae_margin, 2),
        "brier": round(brier, 4),
        "rmse_team_score": round(score_rmse, 2),
        "rmse_margin": round(rmse_margin, 2),
    }


def propose_knob_targets(grades: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Derive safe knob targets from graded history + residuals."""
    grades = grades if grades is not None else _read_jsonl(graded_path())
    residuals = _read_jsonl(residuals_path())
    summary = summarize_learning_grades(grades)
    knobs = load_knobs()
    current = dict(_CONFIG_DEFAULTS)
    for k in SAFE_KNOB_KEYS:
        if k in knobs:
            try:
                current[k] = float(knobs[k])
            except (TypeError, ValueError):
                pass

    targets: dict[str, float] = {}
    n = int(summary.get("n_games") or 0)
    if n <= 0:
        return targets

    if summary.get("rmse_team_score") is not None:
        targets["SCORE_SIGMA"] = _clip_knob("SCORE_SIGMA", float(summary["rmse_team_score"]))
    if summary.get("rmse_margin") is not None:
        targets["MARGIN_SIGMA"] = _clip_knob("MARGIN_SIGMA", float(summary["rmse_margin"]))

    # Home court: mean residual margin (home - away) suggests home court miss
    if residuals:
        mean_res = float(np.mean([float(r.get("residual_margin_home") or 0.0) for r in residuals]))
        targets["HOME_COURT_POINTS"] = _clip_knob(
            "HOME_COURT_POINTS",
            float(current["HOME_COURT_POINTS"]) + mean_res,
        )

        def _adj_for_flag(flag_key: str, knob: str, sign: float = 1.0) -> None:
            rows = [
                r
                for r in residuals
                if (r.get("features") or {}).get(flag_key)
            ]
            if len(rows) < max(5, config.LEARNING_MIN_GAMES // 4):
                return
            # When flag is on the home side, residual_margin_home includes that effect
            mean_r = float(np.mean([float(r.get("residual_margin_home") or 0.0) for r in rows]))
            # If home was B2B and residual is negative (home underperformed), make B2B more negative
            targets[knob] = _clip_knob(knob, float(current[knob]) + sign * mean_r)

        _adj_for_flag("b2b_home", "REST_B2B_POINTS", sign=1.0)
        _adj_for_flag("rest_flag_3in4_home", "REST_3IN4_POINTS", sign=1.0)
        _adj_for_flag("rest_flag_4in6_home", "REST_4IN6_POINTS", sign=1.0)

        alt_rows = [
            r for r in residuals if (r.get("features") or {}).get("altitude")
        ]
        if len(alt_rows) >= max(5, config.LEARNING_MIN_GAMES // 4):
            mean_r = float(
                np.mean([float(r.get("residual_margin_home") or 0.0) for r in alt_rows])
            )
            targets["ALTITUDE_HOME_POINTS"] = _clip_knob(
                "ALTITUDE_HOME_POINTS",
                float(current["ALTITUDE_HOME_POINTS"]) + mean_r,
            )

    return targets


def propose_team_biases(grades: list[dict[str, Any]] | None = None) -> dict[str, float]:
    grades = grades if grades is not None else _read_jsonl(graded_path())
    by_team: dict[str, list[float]] = {}
    for g in grades:
        # residual from home perspective for home team; negate for away
        res_h = float(g["act_margin"]) - float(g["pred_margin"])
        home = normalize_abbrev(g.get("home_abbr"))
        away = normalize_abbrev(g.get("away_abbr"))
        by_team.setdefault(home, []).append(res_h)
        by_team.setdefault(away, []).append(-res_h)
    out: dict[str, float] = {}
    for abbr, vals in by_team.items():
        if len(vals) < config.LEARNING_TEAM_BIAS_MIN_GAMES:
            continue
        bias = float(np.mean(vals))
        bias = float(np.clip(bias, -config.LEARNING_TEAM_BIAS_CAP, config.LEARNING_TEAM_BIAS_CAP))
        if abs(bias) < 0.05:
            continue
        out[abbr] = round(bias, 4)
    return out


def auto_update_knobs(
    *,
    suggested: dict[str, float] | None = None,
    reason: str = "auto_calibrate",
    force: bool = False,
) -> dict[str, Any]:
    """
    Step safe knobs <=10% toward targets when n_games >= LEARNING_MIN_GAMES.
    Returns {applied: bool, changes, knobs, n_games, skipped_reason?}.
    """
    grades = _read_jsonl(graded_path())
    n = len(grades)
    # Hard rule: never auto-apply below min-N (force only bypasses LEARNING_AUTO_APPLY flag)
    if n < config.LEARNING_MIN_GAMES:
        return {
            "applied": False,
            "n_games": n,
            "min_games": config.LEARNING_MIN_GAMES,
            "skipped_reason": f"Need >= {config.LEARNING_MIN_GAMES} graded games (have {n})",
            "changes": {},
            "knobs": load_knobs(),
        }
    if not config.LEARNING_AUTO_APPLY and not force:
        return {
            "applied": False,
            "n_games": n,
            "skipped_reason": "LEARNING_AUTO_APPLY disabled",
            "changes": {},
            "knobs": load_knobs(),
        }

    targets = suggested if suggested is not None else propose_knob_targets(grades)
    knobs = load_knobs()
    current = dict(_CONFIG_DEFAULTS)
    for k in SAFE_KNOB_KEYS:
        if k in knobs:
            try:
                current[k] = float(knobs[k])
            except (TypeError, ValueError):
                pass

    changes: dict[str, dict[str, float]] = {}
    new_vals = dict(current)
    for key, target in (targets or {}).items():
        if key not in SAFE_KNOB_KEYS:
            continue
        try:
            tgt = _clip_knob(key, float(target))
        except (TypeError, ValueError):
            continue
        cur = float(current.get(key, _CONFIG_DEFAULTS[key]))
        nxt = _clip_knob(key, step_toward(cur, tgt))
        if abs(nxt - cur) < 1e-6:
            continue
        changes[key] = {"from": round(cur, 4), "to": round(nxt, 4), "target": round(tgt, 4)}
        new_vals[key] = nxt

    team_biases = propose_team_biases(grades)
    if team_biases:
        new_vals["team_biases"] = team_biases
        changes["team_biases"] = {"to_count": float(len(team_biases))}

    if not changes:
        return {
            "applied": False,
            "n_games": n,
            "skipped_reason": "No knob movement after caps/step",
            "changes": {},
            "knobs": knobs,
        }

    save_knobs(new_vals, n_games_basis=n, reason=reason)
    _record_knob_change(action="auto_update", changes=changes, n_games=n, reason=reason)
    apply_learning_knobs()
    return {
        "applied": True,
        "n_games": n,
        "changes": changes,
        "knobs": load_knobs(),
    }


def import_paste(
    raw: str | dict[str, Any],
    *,
    auto_knobs: bool = True,
) -> dict[str, Any]:
    """
    Archive + normalize a real_box / ai_comparison / ASCII paste into learning.
    Grades means/win% only — never Highlights sim-maxes.
    suggested_code_themes → suggestions_inbox only.
    """
    learning_dir()
    obj: dict[str, Any] | None
    raw_text: str | None = None
    if isinstance(raw, dict):
        obj = raw
    else:
        raw_text = str(raw)
        text = raw_text.strip()
        obj = None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    obj = parsed
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            obj = parse_ascii_paste(text)
        if obj is None:
            raise ValueError(
                "Could not parse paste. Provide real_box / ai_comparison JSON "
                "or ASCII block (DATE / AWAY @ HOME / ACT / PRED)."
            )

    kind = str(obj.get("kind") or "").strip()
    if kind not in ("real_box", "ai_comparison"):
        raise ValueError(f"Unsupported paste kind: {kind!r} (need real_box or ai_comparison)")

    archive_path = _archive_paste(obj if raw_text is None else {"raw_text": raw_text, **obj}, kind)

    date_s = str(obj.get("date") or "")[:10]
    away = normalize_abbrev(obj.get("away_abbr"))
    home = normalize_abbrev(obj.get("home_abbr"))
    if not date_s or not away or not home:
        raise ValueError("Paste needs date, away_abbr, home_abbr")

    try:
        act_a = float(obj.get("act_away_pts"))
        act_h = float(obj.get("act_home_pts"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Paste needs act_away_pts and act_home_pts") from exc

    # Predictions: from paste fields / paired_prediction / output lookup
    pred_a = pred_h = p_home = None
    sim_count = None
    if kind == "ai_comparison":
        try:
            pred_a = float(obj.get("pred_away_pts"))
            pred_h = float(obj.get("pred_home_pts"))
            p_home = float(obj.get("pred_home_win_prob") or 0.5)
        except (TypeError, ValueError):
            pred_a = pred_h = None
        sim_count = obj.get("sim_count")
    paired = obj.get("paired_prediction") if kind == "real_box" else None
    if isinstance(paired, dict) and pred_a is None:
        try:
            pred_a = float(paired.get("pred_away_pts"))
            pred_h = float(paired.get("pred_home_pts"))
            p_home = float(paired.get("pred_home_win_prob") or 0.5)
            sim_count = paired.get("sim_count")
        except (TypeError, ValueError):
            pred_a = pred_h = None
    if pred_a is None:
        looked = lookup_paired_prediction(date_s, away, home)
        if not looked:
            raise ValueError(
                f"No paired prediction for {away}@{home} on {date_s}. "
                "Include paired_prediction / pred_* fields or run the sim first."
            )
        pred_a = float(looked["pred_away_pts"])
        pred_h = float(looked["pred_home_pts"])
        p_home = float(looked["pred_home_win_prob"])
        sim_count = looked.get("sim_count")
    if p_home is None:
        p_home = 0.5

    notes: list[str] = []
    if obj.get("notes"):
        notes.append(ascii_safe(obj.get("notes")))
    if obj.get("ai_summary"):
        notes.append(ascii_safe(obj.get("ai_summary")))

    grade = build_grade_row(
        game_date=date_s,
        away_abbr=away,
        home_abbr=home,
        pred_away_pts=float(pred_a),
        pred_home_pts=float(pred_h),
        pred_home_win_prob=float(p_home),
        act_away_pts=act_a,
        act_home_pts=act_h,
        sim_count=sim_count,
        source="user_paste" if kind == "real_box" else "ai_comparison",
        notes=notes,
    )
    # Never grade Highlights / sim-maxes — only means/win% above.
    append_graded_game(grade)

    applied_rec = {
        "at": _now_iso(),
        "kind": kind,
        "archive": str(archive_path.name),
        "grade_key": f"{date_s}:{away}@{home}",
        "abs_err_total": grade["abs_err_total"],
        "abs_err_margin": grade["abs_err_margin"],
        "brier": grade["brier"],
    }
    _append_jsonl(paste_applied_path(), applied_rec)

    themes_path = _queue_code_themes(
        list(obj.get("suggested_code_themes") or []),
        date_s=date_s,
        matchup=f"{away}@{home}",
        source=kind,
    )

    knob_result: dict[str, Any] = {"applied": False, "changes": {}}
    suggested_knobs = obj.get("suggested_knobs") if isinstance(obj.get("suggested_knobs"), dict) else None
    conf = None
    try:
        conf = float(obj.get("confidence")) if obj.get("confidence") is not None else None
    except (TypeError, ValueError):
        conf = None

    n_after = len(_read_jsonl(graded_path()))
    if auto_knobs and suggested_knobs:
        safe_suggested = {
            k: float(v)
            for k, v in suggested_knobs.items()
            if k in SAFE_KNOB_KEYS
        }
        conf_ok = conf is None or conf >= config.LEARNING_SUGGEST_CONFIDENCE_MIN
        if safe_suggested and conf_ok and n_after >= config.LEARNING_MIN_GAMES:
            knob_result = auto_update_knobs(
                suggested=safe_suggested,
                reason=f"paste_{kind}_suggested",
            )
        elif safe_suggested:
            knob_result = {
                "applied": False,
                "proposed": safe_suggested,
                "skipped_reason": (
                    f"Suggested knobs held (need n>={config.LEARNING_MIN_GAMES} "
                    f"and confidence>={config.LEARNING_SUGGEST_CONFIDENCE_MIN}; "
                    f"n={n_after}, confidence={conf})"
                ),
            }
    elif auto_knobs:
        knob_result = auto_update_knobs(reason=f"paste_{kind}")

    refresh_manifest_counts()
    summary = summarize_learning_grades()
    return {
        "ok": True,
        "kind": kind,
        "archive": str(archive_path),
        "grade": grade,
        "summary": summary,
        "knobs": knob_result,
        "code_themes_inbox": str(themes_path) if themes_path else None,
        "note": "Graded means/win% only — Highlights sim-maxes ignored.",
    }


def format_learning_summary() -> str:
    """ASCII-safe summary for Calibration / Learning pane."""
    apply_learning_knobs()
    summary = summarize_learning_grades()
    knobs = load_knobs()
    mani = load_manifest()
    defaults = config_defaults()
    lines = [
        "NBA PICKER — OFFLINE LEARNING",
        "=" * 64,
        "",
        f"Graded games (learning store): {summary.get('n_games', 0)}",
        f"Pastes applied: {mani.get('n_pastes', 0)}",
        f"Suggestions pending: {mani.get('n_suggestions_pending', 0)}",
        f"Knobs revision: {mani.get('knobs_revision', 0)}",
        "",
        f"MAE total: {summary.get('mae_total')}",
        f"MAE margin: {summary.get('mae_margin')}",
        f"Brier (home win%): {summary.get('brier')}",
        f"RMSE team score: {summary.get('rmse_team_score')}",
        f"RMSE margin: {summary.get('rmse_margin')}",
        "",
        "Active knobs (knobs_current.json over config defaults):",
    ]
    for key in SAFE_KNOB_KEYS:
        cur = knobs.get(key)
        base = defaults[key]
        if cur is None:
            lines.append(f"  {key}: {base} (default)")
        else:
            mark = "" if abs(float(cur) - float(base)) < 1e-9 else " *"
            lines.append(f"  {key}: {cur}  (default {base}){mark}")
    biases = knobs.get("team_biases") or {}
    if biases:
        lines.append("")
        lines.append(
            f"Per-team biases ({len(biases)}; min {config.LEARNING_TEAM_BIAS_MIN_GAMES} "
            f"games, cap +/-{config.LEARNING_TEAM_BIAS_CAP}):"
        )
        for abbr in sorted(biases):
            lines.append(f"  {abbr}: {biases[abbr]:+.2f}")
    lines.append("")
    lines.append(f"Effective SCORE_SIGMA (runtime): {float(config.SCORE_SIGMA):.2f}")
    lines.append(f"Effective MARGIN_SIGMA (runtime): {float(config.MARGIN_SIGMA):.2f}")
    lines.append(f"HOME_COURT_POINTS (runtime): {float(config.HOME_COURT_POINTS):.2f}")
    lines.append("")
    lines.append("Paste real_box / ai_comparison JSON (or ASCII DATE/ACT/PRED).")
    lines.append("suggested_code_themes go to suggestions_inbox/ — never auto code.")
    lines.append("Grading uses projection MEANS and win% — not Highlights sim-maxes.")
    lines.append("Projections only. No betting automation.")
    lines.append("")
    return "\n".join(lines)
