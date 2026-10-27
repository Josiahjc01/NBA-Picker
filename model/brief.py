"""Season averages + player leaders for AI paste / wager-assistant context."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from data.static_arenas import normalize_abbrev

logger = logging.getLogger(__name__)

STAT_KEYS = ("PTS", "REB", "AST", "STL", "BLK")
STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "PTS": ("PTS", "pts", "avgPoints"),
    "REB": ("REB", "reb", "TREB", "avgRebounds", "REB_TOTAL"),
    "AST": ("AST", "ast", "avgAssists"),
    "STL": ("STL", "stl", "avgSteals"),
    "BLK": ("BLK", "blk", "avgBlocks"),
}


def _col(df: pd.DataFrame, *names: str) -> str | None:
    upper = {c.upper(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.upper() in upper:
            return upper[n.upper()]
    return None


def team_season_profiles(team_logs: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """
    Per-team season rates from game logs:
      win_pct, avg_pts, avg_opp_pts, avg_total (both teams' combined score).
    """
    out: dict[str, dict[str, Any]] = {}
    if team_logs is None or (isinstance(team_logs, pd.DataFrame) and team_logs.empty):
        return out
    df = team_logs.copy()
    team_c = _col(df, "TEAM_ABBREVIATION", "team_abbreviation")
    pts_c = _col(df, "PTS", "pts")
    pm_c = _col(df, "PLUS_MINUS", "plus_minus")
    wl_c = _col(df, "WL", "wl")
    if not team_c or not pts_c:
        return out

    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_pts"] = pd.to_numeric(df[pts_c], errors="coerce")
    if pm_c:
        df["_pm"] = pd.to_numeric(df[pm_c], errors="coerce")
        df["_opp"] = df["_pts"] - df["_pm"]
    else:
        df["_opp"] = np.nan

    for abbr, g in df.groupby("_abbr"):
        pts = g["_pts"].dropna()
        opp = g["_opp"].dropna()
        wins = losses = 0
        if wl_c:
            wl = g[wl_c].astype(str).str.upper()
            wins = int((wl == "W").sum())
            losses = int((wl == "L").sum())
        elif pm_c and g["_pm"].notna().any():
            wins = int((g["_pm"] > 0).sum())
            losses = int((g["_pm"] < 0).sum())
        games = wins + losses
        win_pct = (100.0 * wins / games) if games else None
        avg_pts = float(pts.mean()) if len(pts) else None
        avg_opp = float(opp.mean()) if len(opp) else None
        avg_total = (
            float(avg_pts + avg_opp)
            if avg_pts is not None and avg_opp is not None
            else None
        )
        out[str(abbr)] = {
            "team": abbr,
            "games": int(len(g)),
            "wins": wins,
            "losses": losses,
            "win_pct": round(win_pct, 1) if win_pct is not None else None,
            "avg_pts": round(avg_pts, 1) if avg_pts is not None else None,
            "avg_opp_pts": round(avg_opp, 1) if avg_opp is not None else None,
            "avg_total": round(avg_total, 1) if avg_total is not None else None,
        }
    return out


def _stat_series(df: pd.DataFrame, stat: str) -> pd.Series:
    aliases = STAT_ALIASES.get(stat, (stat,))
    for a in aliases:
        c = _col(df, a)
        if c:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def top_players_for_team(
    players: pd.DataFrame | None,
    team: str,
    *,
    n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Top-n players per counting-stat category for one team."""
    empty = {s: [] for s in STAT_KEYS}
    if players is None or players.empty:
        return empty
    abbr = normalize_abbrev(team)
    df = players.copy()
    team_c = _col(df, "TEAM_ABBREVIATION", "team_abbreviation", "team")
    name_c = _col(df, "PLAYER_NAME", "player", "PLAYER")
    if not team_c or not name_c:
        return empty
    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    team_df = df[df["_abbr"] == abbr]
    if team_df.empty:
        return empty

    leaders: dict[str, list[dict[str, Any]]] = {}
    for stat in STAT_KEYS:
        s = _stat_series(team_df, stat)
        ranked = team_df.assign(_stat=s).dropna(subset=["_stat"])
        ranked = ranked.sort_values("_stat", ascending=False).head(n)
        leaders[stat] = [
            {
                "player": str(r[name_c]),
                "value": round(float(r["_stat"]), 1),
            }
            for _, r in ranked.iterrows()
        ]
    return leaders


def enrich_predictions_with_brief(
    predictions: list[dict[str, Any]],
    features: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach season win%, avg combined total, and top-3 leaders to each game."""
    profiles = team_season_profiles(features.get("team_logs"))
    players = features.get("players")
    enriched: list[dict[str, Any]] = []
    for pred in predictions:
        p = dict(pred)
        sides = {}
        for side, key in (("home", "home_abbr"), ("away", "away_abbr")):
            abbr = normalize_abbrev(p.get(key))
            season = dict(profiles.get(abbr) or {"team": abbr})
            season["top_players"] = top_players_for_team(players, abbr, n=3)
            sides[side] = season
        p["season"] = sides
        enriched.append(p)
    return enriched


def _fmt_leaders(leaders: list[dict[str, Any]]) -> str:
    if not leaders:
        return "(n/a)"
    return ", ".join(f"{x['player']} {x['value']}" for x in leaders)


def _team_block(label: str, season: dict[str, Any]) -> list[str]:
    abbr = season.get("team") or label
    win_pct = season.get("win_pct")
    avg_total = season.get("avg_total")
    avg_pts = season.get("avg_pts")
    avg_opp = season.get("avg_opp_pts")
    w = season.get("wins")
    l = season.get("losses")
    lines = [
        f"### {abbr}",
        "",
        f"- **Win rate:** "
        + (f"{win_pct:.1f}% ({w}-{l})" if win_pct is not None else "n/a"),
        f"- **Avg combined score (both teams):** "
        + (f"{avg_total:.1f}" if avg_total is not None else "n/a")
        + (
            f" (team {avg_pts:.1f} + opp {avg_opp:.1f})"
            if avg_pts is not None and avg_opp is not None
            else ""
        ),
        "- **Top 3 players (season per-game):**",
    ]
    tops = season.get("top_players") or {}
    for stat, title in (
        ("PTS", "Points"),
        ("REB", "Rebounds"),
        ("AST", "Assists"),
        ("STL", "Steals"),
        ("BLK", "Blocks"),
    ):
        lines.append(f"  - {title}: {_fmt_leaders(tops.get(stat) or [])}")
    lines.append("")
    return lines


def format_ai_paste(target_date: str, predictions: list[dict[str, Any]]) -> str:
    """
    Plain, paste-friendly block for an external AI to suggest wagers from.
    Includes model projections plus season win%, avg totals, and player leaders.
    """
    sim_n = next(
        (p.get("sim_count") for p in predictions if p.get("sim_count")),
        10000,
    )
    lines = [
        "NBA PICKER — AI PASTE CONTEXT",
        f"Slate date: {target_date}",
        f"Monte Carlo sims per game: n={sim_n}",
        "",
        "Use the season rates and player leaders below (with model projections) "
        "to suggest good wagers. This app does not fetch book odds.",
        "Tags: HOT/COLD = recent form; quarter edges = Q1-Q4 tendency matchup; "
        "clutch Q4 = close-game weighting.",
        "",
    ]
    if not predictions:
        lines.append("No games on this slate.")
        lines.append("")
        return "\n".join(lines)

    for pred in predictions:
        away = pred.get("away_abbr")
        home = pred.get("home_abbr")
        lines.append("=" * 56)
        lines.append(f"{away} @ {home}")
        lines.append("=" * 56)
        lines.append("")
        lines.append("Model projection")
        nsim = pred.get("sim_count") or sim_n
        lines.append(
            f"- Mean score (sim n={nsim}): "
            f"{away} {pred.get('away_pts')} - {home} {pred.get('home_pts')} "
            f"(total {pred.get('total')}, margin {pred.get('margin'):+.1f} home)"
        )
        lines.append(
            f"- Win probability (sims n={nsim}): {home} {float(pred.get('home_win_prob') or 0):.1%} / "
            f"{away} {float(pred.get('away_win_prob') or 0):.1%} "
            f"(confidence {pred.get('confidence')})"
        )
        tags = pred.get("form_tags") or []
        if tags:
            lines.append("- Form tags: " + ", ".join(tags))
        if pred.get("clutch_q4"):
            lines.append("- Clutch: Q4 weighting ON (projected close game)")
        sims = pred.get("sims") or {}
        if sims:
            lines.append(
                f"- Score band p10-p90: {away} {sims.get('away_pts_p10')}-{sims.get('away_pts_p90')} / "
                f"{home} {sims.get('home_pts_p10')}-{sims.get('home_pts_p90')} "
                f"(total {sims.get('total_p10')}-{sims.get('total_p90')})"
            )
        notes = pred.get("notes") or {}
        if notes.get("form"):
            lines.append("- Hot/cold: " + "; ".join(notes["form"]))
        if notes.get("quarters"):
            lines.append("- Quarter edges: " + "; ".join(notes["quarters"]))
        if notes.get("injuries"):
            lines.append("- Injuries: " + "; ".join(notes["injuries"]))
        if notes.get("rest_travel"):
            lines.append("- Rest / travel: " + ", ".join(notes["rest_travel"]))
        lines.append("")

        season = pred.get("season") or {}
        for side in ("away", "home"):
            block = season.get(side) or {"team": pred.get(f"{side}_abbr")}
            # Flatten for plain paste (no markdown hashes)
            plain = []
            for raw in _team_block(side, block):
                if raw.startswith("### "):
                    plain.append(raw[4:])
                    plain.append("-" * len(raw[4:]))
                elif raw.startswith("- **"):
                    plain.append(raw.replace("**", ""))
                elif raw.startswith("  - "):
                    plain.append("  " + raw[4:])
                elif raw.startswith("- "):
                    plain.append(raw)
                else:
                    plain.append(raw)
            lines.extend(plain)

    lines.append("END AI PASTE CONTEXT")
    lines.append("")
    return "\n".join(lines)


def format_ai_paste_markdown(target_date: str, predictions: list[dict[str, Any]]) -> str:
    """Markdown section with the same AI-oriented context."""
    lines = [
        "## AI paste context",
        "",
        "_Copy everything under this heading into another AI for wager ideas. "
        "Includes season win rate, average combined score, and top-3 players "
        "in PTS / REB / AST / STL / BLK._",
        "",
        f"Slate date: **{target_date}**",
        "",
    ]
    if not predictions:
        lines += ["No games on this slate.", ""]
        return "\n".join(lines)

    for pred in predictions:
        away = pred.get("away_abbr")
        home = pred.get("home_abbr")
        lines.append(f"### {away} @ {home}")
        lines.append("")
        nsim = pred.get("sim_count") or 10000
        lines.append(
            f"- **Model score (sim n={nsim}):** {away} {pred.get('away_pts')} – "
            f"{home} {pred.get('home_pts')} "
            f"(total {pred.get('total')}, margin {pred.get('margin'):+.1f} home)"
        )
        lines.append(
            f"- **Model win %:** {home} {float(pred.get('home_win_prob') or 0):.1%} · "
            f"{away} {float(pred.get('away_win_prob') or 0):.1%}"
        )
        tags = pred.get("form_tags") or []
        if tags:
            lines.append(f"- **Form tags:** {', '.join(tags)}")
        notes = pred.get("notes") or {}
        if notes.get("form"):
            lines.append("- **Hot/cold:** " + "; ".join(notes["form"]))
        if notes.get("quarters"):
            lines.append("- **Quarter edges:** " + "; ".join(notes["quarters"]))
        if pred.get("clutch_q4"):
            lines.append("- **Clutch Q4 weighting:** on (projected close)")
        lines.append("")
        season = pred.get("season") or {}
        for side in ("away", "home"):
            lines.extend(_team_block(side, season.get(side) or {"team": pred.get(f"{side}_abbr")}))
    return "\n".join(lines)
