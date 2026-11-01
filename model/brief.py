"""Season context + projected-this-game player leaders for AI paste."""

from __future__ import annotations

import logging
import re
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
# Exclude inactive / DNP from projected leaders
MIN_ADJ_MINUTES_FOR_LEADERS = 1.0


def _col(df: pd.DataFrame, *names: str) -> str | None:
    upper = {c.upper(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.upper() in upper:
            return upper[n.upper()]
    return None


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


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


def _season_stat_by_name(players: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """Normalized player name -> season per-game counting stats + season minutes."""
    out: dict[str, dict[str, float]] = {}
    if players is None or (isinstance(players, pd.DataFrame) and players.empty):
        return out
    df = players.copy()
    name_c = _col(df, "PLAYER_NAME", "player", "PLAYER")
    if not name_c:
        return out
    min_c = _col(df, "MIN", "minutes")
    for _, row in df.iterrows():
        key = _norm_name(row.get(name_c))
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
        for stat in STAT_KEYS:
            aliases = STAT_ALIASES.get(stat, (stat,))
            val = None
            for a in aliases:
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
                entry[stat] = val
        # Prefer first row; roster-aligned frames are unique by name
        if key not in out or (entry.get("MIN") or 0) > (out[key].get("MIN") or 0):
            out[key] = entry
    return out


def projected_game_leaders_for_team(
    players: pd.DataFrame | None,
    minutes: pd.DataFrame | None,
    team: str,
    *,
    team_proj_pts: float | None = None,
    n: int = 3,
    min_adj_minutes: float = MIN_ADJ_MINUTES_FOR_LEADERS,
) -> dict[str, list[dict[str, Any]]]:
    """
    Top-n players by **projected this-game** counting stats.

    Uses season per-game rates scaled by adj_minutes / season_minutes (injury /
    inactive minutes already applied). Players with adj_minutes below the floor
    are excluded. Team membership comes from the minutes frame (roster-aligned),
    not a soft cross-team name match.
    """
    empty = {s: [] for s in STAT_KEYS}
    abbr = normalize_abbrev(team)
    if minutes is None or (isinstance(minutes, pd.DataFrame) and minutes.empty):
        return empty
    if "team" not in minutes.columns or "player" not in minutes.columns:
        return empty

    team_min = minutes[minutes["team"].map(lambda x: normalize_abbrev(str(x))) == abbr].copy()
    if team_min.empty:
        return empty
    team_min["_adj"] = pd.to_numeric(team_min.get("adj_minutes"), errors="coerce").fillna(0.0)
    team_min = team_min[team_min["_adj"] >= float(min_adj_minutes)]
    if team_min.empty:
        return empty

    season_by = _season_stat_by_name(players)
    rows: list[dict[str, Any]] = []
    for _, r in team_min.iterrows():
        name = str(r.get("player") or "")
        key = _norm_name(name)
        adj = float(r["_adj"])
        season = season_by.get(key) or {}
        season_min = float(season.get("MIN") or 0.0)
        # Scale factor: this-game minutes vs season average minutes
        if season_min >= 5.0:
            scale = adj / season_min
        elif season_min > 0:
            scale = adj / max(season_min, 1.0)
        else:
            # Roster-only / no season line — skip counting leaders (no rate basis)
            continue
        proj: dict[str, Any] = {
            "player": name,
            "team": abbr,
            "adj_minutes": round(adj, 1),
        }
        has_any = False
        for stat in STAT_KEYS:
            base = season.get(stat)
            if base is None or not np.isfinite(base):
                proj[stat] = None
                continue
            # Per-game season rate × minutes scale → projected this game
            val = float(base) * scale
            # Soft floors for counting stats
            if stat in ("STL", "BLK") and val < 0:
                val = 0.0
            proj[stat] = float(val)
            has_any = True
        if has_any:
            rows.append(proj)

    if not rows:
        return empty

    # Align projected PTS shares to team sim mean when the active roster
    # already explains most of the team total (avoid exploding thin samples).
    if team_proj_pts is not None and np.isfinite(team_proj_pts) and team_proj_pts > 0:
        pts_vals = [float(r["PTS"]) for r in rows if r.get("PTS") is not None]
        pts_sum = sum(pts_vals)
        if len(pts_vals) >= 5 and pts_sum >= 0.55 * float(team_proj_pts):
            factor = float(team_proj_pts) / pts_sum
            factor = float(np.clip(factor, 0.70, 1.35))
            for r in rows:
                if r.get("PTS") is not None:
                    r["PTS"] = float(r["PTS"]) * factor

    leaders: dict[str, list[dict[str, Any]]] = {}
    for stat in STAT_KEYS:
        ranked = sorted(
            (r for r in rows if r.get(stat) is not None and np.isfinite(r[stat])),
            key=lambda r: float(r[stat]),
            reverse=True,
        )[:n]
        leaders[stat] = [
            {
                "player": str(r["player"]),
                "value": round(float(r[stat]), 1),
                "adj_minutes": r.get("adj_minutes"),
            }
            for r in ranked
        ]
    return leaders


def enrich_predictions_with_brief(
    predictions: list[dict[str, Any]],
    features: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Attach season win% / avg total (context) and top-3 **projected this game**
    leaders (sim-mean team score + minutes-scaled rates) to each game.
    """
    profiles = team_season_profiles(features.get("team_logs"))
    players = features.get("players")
    minutes = features.get("minutes")
    enriched: list[dict[str, Any]] = []
    for pred in predictions:
        p = dict(pred)
        sides = {}
        for side, key, pts_key in (
            ("home", "home_abbr", "home_pts"),
            ("away", "away_abbr", "away_pts"),
        ):
            abbr = normalize_abbrev(p.get(key))
            season = dict(profiles.get(abbr) or {"team": abbr})
            team_pts = p.get(pts_key)
            try:
                team_pts_f = float(team_pts) if team_pts is not None else None
            except (TypeError, ValueError):
                team_pts_f = None
            season["top_players"] = projected_game_leaders_for_team(
                players,
                minutes,
                abbr,
                team_proj_pts=team_pts_f,
                n=3,
            )
            season["leaders_basis"] = "projected_this_game"
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
        f"- **Season win rate:** "
        + (f"{win_pct:.1f}% ({w}-{l})" if win_pct is not None else "n/a"),
        f"- **Season avg combined score (both teams):** "
        + (f"{avg_total:.1f}" if avg_total is not None else "n/a")
        + (
            f" (team {avg_pts:.1f} + opp {avg_opp:.1f})"
            if avg_pts is not None and avg_opp is not None
            else ""
        ),
        "- **Top 3 players (projected this game, sim mean):**",
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
    Includes model projections, season context, and projected-this-game leaders.
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
        "Use model projections and projected-this-game player leaders below "
        "(plus season win rate / avg totals as context) to suggest good wagers. "
        "This app does not fetch book odds.",
        "Player leaders are projected for THIS game (minutes x rates, scaled to "
        "sim mean team score) — not season per-game averages.",
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
        "Includes season win rate / avg combined score (context) and top-3 "
        "players by **projected this game** PTS / REB / AST / STL / BLK "
        "(sim mean — not season per-game)._",
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
