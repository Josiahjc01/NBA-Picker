"""Generate team ratings, projected scores, and win probabilities."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

import config
from data.process import playtype_matchup_residual
from data.static_arenas import normalize_abbrev

logger = logging.getLogger(__name__)

# Logistic scale: ~ margin std of NBA games ≈ 12–13 pts
MARGIN_SIGMA = 12.5


def _team_row(ratings: pd.DataFrame, abbr: str) -> dict[str, float]:
    abbr = normalize_abbrev(abbr)
    defaults = {
        "off_rtg": config.DEFAULT_OFF_RTG,
        "def_rtg": config.DEFAULT_DEF_RTG,
        "net_rtg": 0.0,
        "pace": config.DEFAULT_PACE,
        "efg": 0.53,
        "tov_pct": 0.13,
        "orb_pct": 0.28,
        "ft_rate": 0.25,
        "games": 0,
    }
    if ratings is None or ratings.empty:
        return defaults
    hit = ratings[ratings["team"] == abbr]
    if hit.empty:
        return defaults
    r = hit.iloc[0].to_dict()
    for k, v in defaults.items():
        if k not in r or r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])):
            r[k] = v
    return r


def margin_to_win_prob(margin: float, sigma: float = MARGIN_SIGMA) -> float:
    """P(home win) from expected margin (home − away) via normal CDF approx."""
    # Φ(margin / sigma)
    z = margin / max(sigma, 1e-6)
    # logistic approximation to normal CDF
    return float(1.0 / (1.0 + math.exp(-1.7 * z)))


def project_game(game: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    ratings = features.get("ratings")
    home = normalize_abbrev(game.get("home_abbr"))
    away = normalize_abbrev(game.get("away_abbr"))
    hr = _team_row(ratings, home)
    ar = _team_row(ratings, away)

    # Pace: average team possessions / game
    pace = 0.5 * (float(hr["pace"]) + float(ar["pace"]))
    pace = float(np.clip(pace, 90.0, 110.0))

    # Base ratings with player-impact adjustment (EPM-like points per 100 poss)
    impact = features.get("team_impact") or {}
    hi = impact.get(home) or {}
    ai = impact.get(away) or {}
    def _finite(val, default=0.0):
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    h_pi = float(np.clip(_finite(hi.get("player_impact")), -12.0, 12.0))
    a_pi = float(np.clip(_finite(ai.get("player_impact")), -12.0, 12.0))
    h_off = float(np.clip(hr["off_rtg"], 95.0, 130.0)) + 0.5 * h_pi
    h_def = float(np.clip(hr["def_rtg"], 95.0, 130.0)) - 0.25 * h_pi
    a_off = float(np.clip(ar["off_rtg"], 95.0, 130.0)) + 0.5 * a_pi
    a_def = float(np.clip(ar["def_rtg"], 95.0, 130.0)) - 0.25 * a_pi

    # Four-factors soft nudge
    h_off += 12.0 * (float(hr["efg"]) - 0.53) - 25.0 * (float(hr["tov_pct"]) - 0.13)
    a_off += 12.0 * (float(ar["efg"]) - 0.53) - 25.0 * (float(ar["tov_pct"]) - 0.13)

    league = config.DEFAULT_OFF_RTG
    # Opp-adjusted points: own offense vs opponent defense around league average
    home_pts = pace * (h_off - (a_def - league) * 0.5) / 100.0
    away_pts = pace * (a_off - (h_def - league) * 0.5) / 100.0

    ctx = features.get("context") or {}
    h_ctx = ctx.get(f"{game.get('game_id')}:{home}") or {}
    a_ctx = ctx.get(f"{game.get('game_id')}:{away}") or {}
    # Split context so home-court lands mostly on home margin
    home_pts += float(h_ctx.get("context_points") or (config.HOME_COURT_POINTS if not h_ctx else 0.0)) * 0.6
    away_pts += float(a_ctx.get("context_points") or 0.0) * 0.6

    matchup_edge, matchup_notes = playtype_matchup_residual(
        features.get("play_types"), home, away
    )
    home_pts += matchup_edge / 2.0
    away_pts -= matchup_edge / 2.0

    # Sanity clamp — NBA team scores rarely outside this band in projections
    home_pts = float(np.clip(home_pts, 85.0, 145.0))
    away_pts = float(np.clip(away_pts, 85.0, 145.0))

    # Injury priority already in player_impact; surface notes
    injury_notes = list(hi.get("injury_notes") or []) + list(ai.get("injury_notes") or [])
    # Biggest injury deltas
    if hi.get("injury_delta"):
        injury_notes.insert(0, f"{home} injury dNet {hi['injury_delta']:+.2f}")
    if ai.get("injury_delta"):
        injury_notes.insert(0, f"{away} injury dNet {ai['injury_delta']:+.2f}")

    margin = home_pts - away_pts
    win_p = margin_to_win_prob(margin)

    # Confidence: more games + fewer missing layers → higher
    games_n = min(int(hr.get("games") or 0), int(ar.get("games") or 0))
    confidence = float(np.clip(0.35 + 0.05 * games_n, 0.35, 0.9))
    if not (features.get("team_impact")):
        confidence *= 0.85
    if injury_notes:
        confidence = min(confidence, 0.75)

    flags = list(dict.fromkeys((h_ctx.get("flags") or []) + (a_ctx.get("flags") or [])))

    return {
        "game_id": game.get("game_id"),
        "game_date": game.get("game_date"),
        "home_abbr": home,
        "away_abbr": away,
        "home_name": game.get("home_name") or home,
        "away_name": game.get("away_name") or away,
        "home_pts": round(home_pts, 1),
        "away_pts": round(away_pts, 1),
        "total": round(home_pts + away_pts, 1),
        "margin": round(margin, 1),
        "home_win_prob": round(win_p, 4),
        "away_win_prob": round(1.0 - win_p, 4),
        "confidence": round(confidence, 3),
        "pace": round(pace, 1),
        "notes": {
            "injuries": injury_notes[:6],
            "rest_travel": flags,
            "play_types": matchup_notes,
        },
        "components": {
            "home_off": round(h_off, 2),
            "home_def": round(h_def, 2),
            "away_off": round(a_off, 2),
            "away_def": round(a_def, 2),
            "matchup_residual": round(matchup_edge, 2),
            "home_player_impact": round(float(hi.get("player_impact") or 0.0), 3),
            "away_player_impact": round(float(ai.get("player_impact") or 0.0), 3),
        },
    }


def predict_slate(features: dict[str, Any]) -> list[dict[str, Any]]:
    games = features.get("games") or []
    preds = []
    for g in games:
        try:
            preds.append(project_game(g, features))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prediction failed for %s: %s", g.get("game_id"), exc)
    return preds
