"""Monte Carlo player-level counting stats consistent with team score sims."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

import config
from data.static_arenas import normalize_abbrev
from safe_text import normalize_player_name

logger = logging.getLogger(__name__)

# Stats drawn each sim (Highlights use PTS/REB/AST/FG3M; AI paste also STL/BLK)
PLAYER_SIM_STATS = ("PTS", "REB", "AST", "FG3M", "STL", "BLK")
STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "PTS": ("PTS", "pts", "avgPoints"),
    "REB": ("REB", "reb", "TREB", "avgRebounds", "REB_TOTAL"),
    "AST": ("AST", "ast", "avgAssists"),
    "STL": ("STL", "stl", "avgSteals"),
    "BLK": ("BLK", "blk", "avgBlocks"),
    "FG3M": (
        "FG3M",
        "fg3m",
        "AVG_FG3M",
        "avgThreePointFieldGoalsMade",
        "threePointFieldGoalsMade",
        "FG3",
    ),
}
MIN_ADJ_MINUTES = 1.0
# Multiplicative usage noise on PTS share allocation (log-space)
PTS_SHARE_SIGMA = 0.28
# Chunk sims so 1M x roster stays memory-friendly
DEFAULT_CHUNK = 25_000


def _col(df: pd.DataFrame, *names: str) -> str | None:
    upper = {c.upper(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.upper() in upper:
            return upper[n.upper()]
    return None


def _norm_name(name: str) -> str:
    return normalize_player_name(name)


def _season_rates_by_name(players: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """player-name-norm -> season per-game counting stats + MIN."""
    out: dict[str, dict[str, float]] = {}
    if players is None or (isinstance(players, pd.DataFrame) and players.empty):
        return out
    df = players
    name_c = _col(df, "PLAYER_NAME", "player", "PLAYER")
    if not name_c:
        return out
    min_c = _col(df, "MIN", "minutes")
    for _, row in df.iterrows():
        key = normalize_player_name(row.get(name_c))
        if not key:
            continue
        entry: dict[str, float] = {}
        if min_c:
            try:
                entry["MIN"] = float(pd.to_numeric(row.get(min_c), errors="coerce") or 0.0)
            except (TypeError, ValueError):
                entry["MIN"] = 0.0
        else:
            entry["MIN"] = 0.0
        for stat in PLAYER_SIM_STATS:
            val = None
            for a in STAT_ALIASES.get(stat, (stat,)):
                c = _col(df, a)
                if not c:
                    continue
                try:
                    v = float(pd.to_numeric(row.get(c), errors="coerce"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    val = v
                    break
            if val is not None:
                entry[stat] = float(val)
        if key not in out or (entry.get("MIN") or 0) > (out[key].get("MIN") or 0):
            out[key] = entry
    return out


def build_active_roster(
    players: pd.DataFrame | None,
    minutes: pd.DataFrame | None,
    team: str,
    *,
    min_adj_minutes: float = MIN_ADJ_MINUTES,
    player_rates: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Active players with generative means = blended recent L5/L10 + season
    rates x adj_minutes. Out / 0-minute players are excluded.
    """
    abbr = normalize_abbrev(team)
    if minutes is None or (isinstance(minutes, pd.DataFrame) and minutes.empty):
        return []
    if "team" not in minutes.columns or "player" not in minutes.columns:
        return []

    team_min = minutes[minutes["team"].map(lambda x: normalize_abbrev(str(x))) == abbr].copy()
    if team_min.empty:
        return []
    team_min["_adj"] = pd.to_numeric(team_min.get("adj_minutes"), errors="coerce").fillna(0.0)
    team_min = team_min[team_min["_adj"] >= float(min_adj_minutes)]
    if team_min.empty:
        return []

    season_by = _season_rates_by_name(players)
    rates_by = player_rates or {}
    w5 = float(config.PLAYER_RATE_W_L5)
    w10 = float(config.PLAYER_RATE_W_L10)
    w_s = float(config.PLAYER_RATE_W_SEASON)
    w_sum = max(w5 + w10 + w_s, 1e-9)
    w5, w10, w_s = w5 / w_sum, w10 / w_sum, w_s / w_sum

    roster: list[dict[str, Any]] = []
    for _, r in team_min.iterrows():
        name = str(r.get("player") or "")
        key = normalize_player_name(name)
        adj = float(r["_adj"])
        season = season_by.get(key) or {}
        recent = rates_by.get(key) or {}
        season_min = float(season.get("MIN") or 0.0)
        if season_min < 1.0 and not any(s in season for s in PLAYER_SIM_STATS) and not recent:
            continue
        denom = season_min if season_min >= 5.0 else max(season_min, 30.0)
        means: dict[str, float] = {}
        for stat in PLAYER_SIM_STATS:
            season_pg = season.get(stat)
            # Recent per-game averages from gamelog profiles
            l5 = recent.get(f"l5_{stat}")
            l10 = recent.get(f"l10_{stat}")
            parts = []
            weights = []
            if l5 is not None and np.isfinite(float(l5)):
                parts.append(float(l5))
                weights.append(w5)
            if l10 is not None and np.isfinite(float(l10)):
                parts.append(float(l10))
                weights.append(w10)
            if season_pg is not None and np.isfinite(float(season_pg)) and float(season_pg) >= 0:
                parts.append(float(season_pg))
                weights.append(w_s)
            if not parts:
                continue
            wtot = sum(weights) or 1.0
            blended_pg = sum(p * w for p, w in zip(parts, weights)) / wtot
            # Scale per-game blend to this-game minutes
            if season_min >= 5.0:
                means[stat] = float(blended_pg) * (adj / denom)
            else:
                means[stat] = float(blended_pg) * (adj / max(denom, 1.0))
        if "PTS" not in means and not means:
            continue
        if "PTS" not in means:
            means["PTS"] = max(adj / 36.0 * 8.0, 1.0)
        usage_share = None
        try:
            usage_share = float(r.get("usage_share")) if r.get("usage_share") is not None else None
        except (TypeError, ValueError):
            usage_share = None
        roster.append(
            {
                "player": name,
                "team": abbr,
                "adj_minutes": round(adj, 1),
                "means": means,
                "usage_share": usage_share,
            }
        )
    return roster


def _simulate_chunk_pts(
    mu_pts: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    usage_shares: np.ndarray | None = None,
) -> np.ndarray:
    """
    Allocate team score to players each sim via noisy usage shares.
    Returns shape (C, P) floats summing to scores along axis=1.
    """
    c = scores.shape[0]
    p = mu_pts.shape[0]
    if p == 0:
        return np.zeros((c, 0), dtype=np.float64)
    mu = np.maximum(mu_pts.astype(np.float64), 1e-3)
    if usage_shares is not None and usage_shares.shape[0] == p and float(np.nansum(usage_shares)) > 0:
        us = np.maximum(np.nan_to_num(usage_shares.astype(np.float64), nan=0.0), 0.0)
        if float(us.sum()) > 0:
            us = us / us.sum()
            mu_n = mu / mu.sum()
            # Blend possession share with scoring-rate share (avoid double-counting)
            base = 0.55 * us + 0.45 * mu_n
        else:
            base = mu
    else:
        base = mu
    # Multiplicative log-noise around usage
    z = rng.standard_normal((c, p))
    weights = base[None, :] * np.exp(PTS_SHARE_SIGMA * z)
    shares = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    return shares * scores[:, None]


def _simulate_chunk_count(
    mu: np.ndarray,
    scores: np.ndarray,
    score_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Counting-stat draws scaled by team-score intensity (pace/scoring context).
    Poisson around mu * (team_score / E[team_score]).
    """
    c = scores.shape[0]
    p = mu.shape[0]
    if p == 0:
        return np.zeros((c, 0), dtype=np.float64)
    intensity = scores / max(float(score_mean), 1e-6)
    lam = np.maximum(mu[None, :].astype(np.float64) * intensity[:, None], 0.0)
    # Cap lambda to keep Poisson stable for wild outliers
    lam = np.clip(lam, 0.0, 80.0)
    return rng.poisson(lam).astype(np.float64)


def simulate_team_players(
    roster: list[dict[str, Any]],
    team_scores: np.ndarray,
    rng: np.random.Generator,
    *,
    chunk_size: int = DEFAULT_CHUNK,
) -> dict[str, Any]:
    """
    Run player-level MC for one team given that team's score path.

    Returns per-player sim means and sim maxes for PLAYER_SIM_STATS.
    """
    scores = np.asarray(team_scores, dtype=np.float64).reshape(-1)
    n = int(scores.shape[0])
    if not roster or n <= 0:
        return {
            "sim_count": n,
            "players": [],
            "basis": "monte_carlo_player",
        }

    names = [r["player"] for r in roster]
    adj = [float(r["adj_minutes"]) for r in roster]
    p = len(roster)

    mu: dict[str, np.ndarray] = {}
    for stat in PLAYER_SIM_STATS:
        mu[stat] = np.array(
            [float((r.get("means") or {}).get(stat) or 0.0) for r in roster],
            dtype=np.float64,
        )
    usage = np.array(
        [
            float(r["usage_share"])
            if r.get("usage_share") is not None and np.isfinite(float(r.get("usage_share") or np.nan))
            else np.nan
            for r in roster
        ],
        dtype=np.float64,
    )
    if not np.isfinite(usage).any():
        usage = None

    sum_acc = {s: np.zeros(p, dtype=np.float64) for s in PLAYER_SIM_STATS}
    max_acc = {s: np.full(p, -np.inf, dtype=np.float64) for s in PLAYER_SIM_STATS}
    score_mean = float(scores.mean()) if n else 1.0

    chunk = max(1_000, int(chunk_size))
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sc = scores[start:end]
        # PTS: allocate team score
        pts = _simulate_chunk_pts(mu["PTS"], sc, rng, usage_shares=usage)
        sum_acc["PTS"] += pts.sum(axis=0)
        max_acc["PTS"] = np.maximum(max_acc["PTS"], pts.max(axis=0))
        # Other counting stats: intensity-scaled Poisson
        for stat in PLAYER_SIM_STATS:
            if stat == "PTS":
                continue
            if not np.any(mu[stat] > 0):
                continue
            draws = _simulate_chunk_count(mu[stat], sc, score_mean, rng)
            sum_acc[stat] += draws.sum(axis=0)
            max_acc[stat] = np.maximum(max_acc[stat], draws.max(axis=0))

    players_out: list[dict[str, Any]] = []
    for i in range(p):
        means_out: dict[str, float] = {}
        maxes_out: dict[str, float] = {}
        for stat in PLAYER_SIM_STATS:
            if not np.any(mu[stat] > 0) and stat != "PTS":
                continue
            if stat != "PTS" and mu[stat][i] <= 0 and max_acc[stat][i] < 0:
                continue
            means_out[stat] = round(float(sum_acc[stat][i] / n), 1)
            mx = max_acc[stat][i]
            if np.isfinite(mx) and mx >= 0:
                # PTS stays 1-decimal; counting stats are integer draws
                maxes_out[stat] = (
                    round(float(mx), 1) if stat == "PTS" else int(round(float(mx)))
                )
        players_out.append(
            {
                "player": names[i],
                "adj_minutes": adj[i],
                "mean": means_out,
                "max": maxes_out,
            }
        )

    return {
        "sim_count": n,
        "players": players_out,
        "basis": "monte_carlo_player",
    }


def leaders_from_means(
    team_sim: dict[str, Any],
    *,
    n: int = 3,
    stats: tuple[str, ...] = PLAYER_SIM_STATS,
) -> dict[str, list[dict[str, Any]]]:
    """Top-n by sim mean for each stat (AI paste)."""
    players = team_sim.get("players") or []
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in stats}
    for stat in stats:
        ranked = sorted(
            (
                pl
                for pl in players
                if (pl.get("mean") or {}).get(stat) is not None
                and float(pl["adj_minutes"]) >= MIN_ADJ_MINUTES
            ),
            key=lambda pl: float(pl["mean"][stat]),
            reverse=True,
        )[:n]
        out[stat] = [
            {
                "player": pl["player"],
                "value": pl["mean"][stat],
                "adj_minutes": pl.get("adj_minutes"),
                "basis": "sim_mean",
            }
            for pl in ranked
        ]
    return out


def leaders_from_maxes(
    team_sim: dict[str, Any],
    *,
    stats: tuple[str, ...] = ("PTS", "REB", "AST", "FG3M"),
) -> dict[str, dict[str, Any] | None]:
    """
    One leader per stat = player with the highest sim-max for that stat.
    """
    players = team_sim.get("players") or []
    out: dict[str, dict[str, Any] | None] = {}
    for stat in stats:
        best = None
        best_val = -1.0
        for pl in players:
            if float(pl.get("adj_minutes") or 0) < MIN_ADJ_MINUTES:
                continue
            mx = (pl.get("max") or {}).get(stat)
            if mx is None:
                continue
            v = float(mx)
            if v > best_val:
                best_val = v
                best = {
                    "player": pl["player"],
                    "value": mx if stat != "PTS" else round(float(mx), 1),
                    "adj_minutes": pl.get("adj_minutes"),
                    "basis": "sim_max",
                }
        out[stat] = best
    return out


def run_matchup_player_sims(
    features: dict[str, Any],
    home: str,
    away: str,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Simulate both teams' player stats from the same score paths."""
    players = features.get("players")
    minutes = features.get("minutes")
    player_rates = features.get("player_rates") or {}
    rng = np.random.default_rng(seed)
    home_roster = build_active_roster(
        players, minutes, home, player_rates=player_rates
    )
    away_roster = build_active_roster(
        players, minutes, away, player_rates=player_rates
    )
    home_sim = simulate_team_players(home_roster, home_scores, rng)
    away_sim = simulate_team_players(away_roster, away_scores, rng)
    n = int(np.asarray(home_scores).reshape(-1).shape[0])
    return {
        "sim_count": n,
        "basis": "monte_carlo_player",
        "home": home_sim,
        "away": away_sim,
    }
