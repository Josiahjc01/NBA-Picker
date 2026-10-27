"""Build features: rolling team stats, injury minutes, rest/travel context."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

import config
from data.static_arenas import normalize_abbrev, travel_miles, timezone_shift_hours

logger = logging.getLogger(__name__)

OUT_STATUSES = re.compile(r"\b(out|ofs|inactive)\b", re.I)
DOUBTFUL = re.compile(r"\bdoubtful\b", re.I)
QUESTIONABLE = re.compile(r"\b(questionable|gtd|day[\s\-]?to[\s\-]?day|dtd)\b", re.I)
PROBABLE = re.compile(r"\b(probable|available)\b", re.I)


def _parse_game_date(val: Any) -> date | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    s = str(val)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:19].replace("Z", ""), fmt if "T" not in fmt else "%Y-%m-%dT%H:%M:%S").date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:  # noqa: BLE001
        return None


def minutes_factor_from_status(status: str) -> float:
    """Map injury status → expected minutes multiplier (1 = full)."""
    if not status:
        return 1.0
    if OUT_STATUSES.search(status):
        return 0.0
    if DOUBTFUL.search(status):
        return 0.15
    if QUESTIONABLE.search(status):
        return 0.55
    if PROBABLE.search(status):
        return 0.9
    return 1.0


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


def index_injuries(injuries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """player-name-lower → injury record with minutes factor + team abbr when known."""
    out: dict[str, dict[str, Any]] = {}
    for inj in injuries or []:
        name = str(inj.get("player") or "").strip()
        if not name:
            continue
        status = str(inj.get("status") or "")
        # Prefer dedicated status; only fall back to comment if status empty
        if not status.strip():
            status = str(inj.get("comment") or "")
        key = _norm_name(name)
        factor = minutes_factor_from_status(status)
        team_abbr = normalize_abbrev(str(inj.get("team_abbr") or ""))
        prev = out.get(key)
        if prev is None or factor < prev["minutes_factor"]:
            out[key] = {
                "player": name,
                "team": inj.get("team"),
                "team_abbr": team_abbr,
                "status": status,
                "minutes_factor": factor,
                "source": inj.get("source"),
            }
    return out


def align_players_to_rosters(
    players: pd.DataFrame | None,
    rosters: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Remap season stats onto current roster membership.

    - Players on a roster keep/get that team's abbreviation (fixes post-trade tags).
    - Roster players missing from the stats feed are added with empty counting stats
      so injury / depth charts still see them.
    - Stats rows for players not on any current roster are dropped (stale).
    """
    roster_rows = rosters or []
    if not roster_rows:
        return players.copy() if players is not None else pd.DataFrame()

    roster_by_name: dict[str, dict[str, Any]] = {}
    for r in roster_rows:
        key = _norm_name(r.get("PLAYER_NAME") or "")
        if not key:
            continue
        # Prefer active over inactive if duplicates
        prev = roster_by_name.get(key)
        status = str(r.get("STATUS") or "active").lower()
        if prev is None or (status == "active" and str(prev.get("STATUS") or "").lower() != "active"):
            roster_by_name[key] = r

    if players is None or (isinstance(players, pd.DataFrame) and players.empty):
        # Roster-only skeleton
        return pd.DataFrame(
            [
                {
                    "PLAYER_NAME": r["PLAYER_NAME"],
                    "TEAM_ABBREVIATION": normalize_abbrev(r["TEAM_ABBREVIATION"]),
                    "MIN": 0.0,
                    "PTS": None,
                    "REB": None,
                    "AST": None,
                    "STL": None,
                    "BLK": None,
                    "GP": None,
                    "ON_ROSTER": True,
                    "ROSTER_STATUS": r.get("STATUS"),
                    "SOURCE": "roster_only",
                }
                for r in roster_by_name.values()
            ]
        )

    df = players.copy()
    name_c = next((c for c in ("PLAYER_NAME", "player", "PLAYER") if c in df.columns), None)
    team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in df.columns), None)
    if not name_c:
        return df

    aligned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        name = str(row.get(name_c) or "")
        key = _norm_name(name)
        rost = roster_by_name.get(key)
        if rost is None:
            # Soft last-name+first-initial match against roster
            parts = [p for p in key.replace(".", "").split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
            if len(parts) >= 2:
                last, first = parts[-1], parts[0]
                for rk, rv in roster_by_name.items():
                    rp = [p for p in rk.replace(".", "").split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
                    if len(rp) >= 2 and rp[-1] == last and rp[0][:1] == first[:1]:
                        rost = rv
                        key = rk
                        break
        if rost is None:
            continue  # not on a current roster
        if key in seen:
            continue
        seen.add(key)
        d = row.to_dict()
        d["PLAYER_NAME"] = rost.get("PLAYER_NAME") or name
        d["TEAM_ABBREVIATION"] = normalize_abbrev(rost["TEAM_ABBREVIATION"])
        d["ON_ROSTER"] = True
        d["ROSTER_STATUS"] = rost.get("STATUS")
        aligned.append(d)

    # Add roster players with no season-stats row yet
    for key, rost in roster_by_name.items():
        if key in seen:
            continue
        aligned.append(
            {
                "PLAYER_NAME": rost["PLAYER_NAME"],
                "TEAM_ABBREVIATION": normalize_abbrev(rost["TEAM_ABBREVIATION"]),
                "MIN": 0.0,
                "PTS": None,
                "REB": None,
                "AST": None,
                "STL": None,
                "BLK": None,
                "GP": None,
                "ON_ROSTER": True,
                "ROSTER_STATUS": rost.get("STATUS"),
                "SOURCE": "roster_only",
            }
        )

    out = pd.DataFrame(aligned)
    logger.info(
        "Aligned players to rosters: %d stats→roster matches, %d roster-only, %d total",
        len(seen),
        len(out) - len(seen),
        len(out),
    )
    return out



def build_team_rolling(team_logs: pd.DataFrame, span: int | None = None) -> pd.DataFrame:
    """EWM Net Rating / Pace / Four Factors proxies from team game logs."""
    span = span or config.EWM_SPAN
    if team_logs is None or team_logs.empty:
        return pd.DataFrame()

    df = team_logs.copy()
    # Normalize columns
    colmap = {c.upper(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.upper() in colmap:
                return colmap[n.upper()]
        return None

    team_c = col("TEAM_ABBREVIATION", "team_abbreviation")
    date_c = col("GAME_DATE", "game_date")
    pts_c = col("PTS", "pts")
    plus_c = col("PLUS_MINUS", "plus_minus")
    if not team_c or not date_c:
        logger.warning("Team logs missing TEAM_ABBREVIATION/GAME_DATE")
        return pd.DataFrame()

    df["_date"] = df[date_c].apply(_parse_game_date)
    df = df.dropna(subset=["_date"]).sort_values(["_date"])
    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))

    # Approximate possessions / pace & ratings from box score fields when available
    fga = col("FGA")
    fta = col("FTA")
    tov = col("TOV", "TO")
    oreb = col("OREB")
    fg3m = col("FG3M")
    fgm = col("FGM")

    rows = []
    for abbr, g in df.groupby("_abbr"):
        g = g.sort_values("_date").copy()
        pts = pd.to_numeric(g[pts_c], errors="coerce") if pts_c else pd.Series(0, index=g.index)
        pm = pd.to_numeric(g[plus_c], errors="coerce") if plus_c else pd.Series(0, index=g.index)
        opp = pts - pm

        def _num(cname):
            if not cname:
                return None
            s = pd.to_numeric(g[cname], errors="coerce")
            return s if s.notna().any() else None

        fga_s, fta_s, tov_s, oreb_s = _num(fga), _num(fta), _num(tov), _num(oreb)
        fgm_s, fg3m_s = _num(fgm), _num(fg3m)

        # Possessions ≈ FGA - OREB + TOV + 0.44*FTA (only when box score present)
        if fga_s is not None and tov_s is not None and oreb_s is not None and fta_s is not None:
            # Team possessions per game (typically ~95–105)
            poss = (fga_s.fillna(0) - oreb_s.fillna(0) + tov_s.fillna(0) + 0.44 * fta_s.fillna(0)).clip(
                lower=80.0, upper=120.0
            )
        else:
            # Score-only logs (e.g. ESPN schedules): league-average team possessions
            poss = pd.Series(float(config.DEFAULT_PACE), index=g.index)

        off_rtg = 100.0 * pts / poss
        def_rtg = 100.0 * opp / poss
        pace = poss  # store as team possessions / game
        net = off_rtg - def_rtg

        # Four factors
        if fgm_s is not None and fga_s is not None and fg3m_s is not None:
            efg = (fgm_s + 0.5 * fg3m_s) / fga_s.replace(0, np.nan)
        else:
            efg = pd.Series(np.nan, index=g.index)
        if tov_s is not None and fga_s is not None and fta_s is not None:
            tov_pct = tov_s / (fga_s + 0.44 * fta_s + tov_s).replace(0, np.nan)
        else:
            tov_pct = pd.Series(np.nan, index=g.index)
        if oreb_s is not None:
            orb_pct = oreb_s / (oreb_s + 25.0)
        else:
            orb_pct = pd.Series(np.nan, index=g.index)
        if fta_s is not None and fga_s is not None:
            ft_rate = fta_s / fga_s.replace(0, np.nan)
        else:
            ft_rate = pd.Series(np.nan, index=g.index)

        ewm = lambda s: s.ewm(span=span, min_periods=1).mean()  # noqa: E731
        last = {
            "team": abbr,
            "games": int(len(g)),
            "last_game": g["_date"].iloc[-1].isoformat(),
            "off_rtg": float(ewm(off_rtg).iloc[-1]),
            "def_rtg": float(ewm(def_rtg).iloc[-1]),
            "net_rtg": float(ewm(net).iloc[-1]),
            "pace": float(ewm(pace).iloc[-1]),
            "efg": float(ewm(efg.fillna(efg.mean())).iloc[-1]) if efg.notna().any() else 0.53,
            "tov_pct": float(ewm(tov_pct.fillna(0.13)).iloc[-1]) if tov_pct.notna().any() else 0.13,
            "orb_pct": float(ewm(orb_pct.fillna(0.28)).iloc[-1]) if orb_pct.notna().any() else 0.28,
            "ft_rate": float(ewm(ft_rate.fillna(0.25)).iloc[-1]) if ft_rate.notna().any() else 0.25,
            "recent_dates": [d.isoformat() for d in g["_date"].tolist()[-5:]],
        }
        rows.append(last)
    return pd.DataFrame(rows)


def blend_with_prior(
    current: pd.DataFrame,
    prior: pd.DataFrame,
    n_blend: int | None = None,
) -> pd.DataFrame:
    """Blend prior-season ratings until ~n_blend games played."""
    n_blend = n_blend or config.PRIOR_BLEND_GAMES
    if current is None or current.empty:
        return prior.copy() if prior is not None else pd.DataFrame()
    if prior is None or prior.empty:
        return current.copy()

    prior_idx = prior.set_index("team")
    early = float(max(config.EARLY_SEASON_GAMES, 1))
    out = []
    for _, row in current.iterrows():
        team = row["team"]
        games = int(row.get("games") or 0)
        # Stronger prior trust below EARLY_SEASON_GAMES (quadratic soft ramp)
        if games < early:
            w = (games / early) ** 2 * (early / float(n_blend))
        else:
            w = min(1.0, games / float(n_blend))
        w = float(np.clip(w, 0.0, 1.0))
        if team in prior_idx.index:
            p = prior_idx.loc[team]
            blended = row.to_dict()
            for metric in ("off_rtg", "def_rtg", "net_rtg", "pace", "efg", "tov_pct", "orb_pct", "ft_rate"):
                if metric in row and metric in p:
                    blended[metric] = w * float(row[metric]) + (1 - w) * float(p[metric])
            blended["blend_weight_current"] = w
            out.append(blended)
        else:
            d = row.to_dict()
            d["blend_weight_current"] = w
            out.append(d)
    # Teams only in prior
    cur_teams = set(current["team"])
    for team, p in prior_idx.iterrows():
        if team not in cur_teams:
            d = p.to_dict()
            d["team"] = team
            d["games"] = 0
            d["blend_weight_current"] = 0.0
            out.append(d)
    return pd.DataFrame(out)


def build_team_form(
    team_logs: pd.DataFrame,
    *,
    last_n: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Hot/cold from last-N net margin / scoring vs season averages.
    Returns per-team form_edge (points, capped) + notes when material.
    """
    last_n = last_n or config.FORM_LAST_N
    out: dict[str, dict[str, Any]] = {}
    if team_logs is None or (isinstance(team_logs, pd.DataFrame) and team_logs.empty):
        return out

    df = team_logs.copy()
    colmap = {c.upper(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
            if n.upper() in colmap:
                return colmap[n.upper()]
        return None

    team_c = col("TEAM_ABBREVIATION", "team_abbreviation")
    date_c = col("GAME_DATE", "game_date")
    pts_c = col("PTS", "pts")
    plus_c = col("PLUS_MINUS", "plus_minus")
    if not team_c or not date_c or not pts_c:
        return out

    df["_date"] = df[date_c].apply(_parse_game_date)
    df = df.dropna(subset=["_date"]).sort_values("_date")
    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_pts"] = pd.to_numeric(df[pts_c], errors="coerce")
    if plus_c:
        df["_pm"] = pd.to_numeric(df[plus_c], errors="coerce")
    else:
        df["_pm"] = np.nan

    for abbr, g in df.groupby("_abbr"):
        g = g.dropna(subset=["_pts"]).sort_values("_date")
        if len(g) < 3:
            continue
        season_margin = float(g["_pm"].mean()) if g["_pm"].notna().any() else 0.0
        season_pts = float(g["_pts"].mean())
        recent = g.tail(last_n)
        recent_margin = float(recent["_pm"].mean()) if recent["_pm"].notna().any() else season_margin
        recent_pts = float(recent["_pts"].mean())
        # Form signal: margin gap + mild scoring gap (offense heat)
        margin_gap = recent_margin - season_margin
        pts_gap = recent_pts - season_pts
        raw = 0.55 * margin_gap + 0.20 * pts_gap
        edge = float(np.clip(raw, -config.FORM_EDGE_CAP, config.FORM_EDGE_CAP))
        # Streak: consecutive W/L at end of log
        streak = 0
        if plus_c and recent["_pm"].notna().any():
            signs = list(recent["_pm"].dropna().map(lambda x: 1 if x > 0 else -1))
            if signs:
                streak = signs[-1]
                for s in reversed(signs[:-1]):
                    if s == signs[-1]:
                        streak += signs[-1]
                    else:
                        break
        tag = "NEUTRAL"
        if edge >= 1.25 or streak >= 3:
            tag = "HOT"
            # Streak-only heat still moves the needle a little
            if edge < 1.25:
                edge = max(edge, min(1.5, 0.45 * abs(streak)))
        elif edge <= -1.25 or streak <= -3:
            tag = "COLD"
            if edge > -1.25:
                edge = min(edge, -min(1.5, 0.45 * abs(streak)))
        edge = float(np.clip(edge, -config.FORM_EDGE_CAP, config.FORM_EDGE_CAP))
        note = None
        if tag != "NEUTRAL":
            note = (
                f"{abbr} {tag}: last {len(recent)}g margin {recent_margin:+.1f} "
                f"(szn {season_margin:+.1f}), pts {recent_pts:.1f}"
                + (f", streak {streak:+d}" if abs(streak) >= 3 else "")
            )
        out[str(abbr)] = {
            "team": abbr,
            "tag": tag,
            "form_edge": round(edge, 2),
            "recent_margin": round(recent_margin, 2),
            "season_margin": round(season_margin, 2),
            "recent_pts": round(recent_pts, 1),
            "season_pts": round(season_pts, 1),
            "streak": int(streak),
            "n": int(len(recent)),
            "note": note,
        }
    return out


def build_home_road_splits(team_logs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Home vs road offensive/defensive scoring splits (points vs own average)."""
    out: dict[str, dict[str, Any]] = {}
    if team_logs is None or (isinstance(team_logs, pd.DataFrame) and team_logs.empty):
        return out

    df = team_logs.copy()
    colmap = {c.upper(): c for c in df.columns}
    team_c = colmap.get("TEAM_ABBREVIATION") or ("TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None)
    match_c = colmap.get("MATCHUP") or ("MATCHUP" if "MATCHUP" in df.columns else None)
    pts_c = colmap.get("PTS") or ("PTS" if "PTS" in df.columns else None)
    plus_c = colmap.get("PLUS_MINUS") or ("PLUS_MINUS" if "PLUS_MINUS" in df.columns else None)
    if not team_c or not pts_c:
        return out

    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_pts"] = pd.to_numeric(df[pts_c], errors="coerce")
    if plus_c:
        df["_pm"] = pd.to_numeric(df[plus_c], errors="coerce")
        df["_opp"] = df["_pts"] - df["_pm"]
    else:
        df["_opp"] = np.nan

    def _is_home(row: pd.Series) -> bool | None:
        if match_c and isinstance(row.get(match_c), str):
            m = str(row[match_c])
            if "@" in m:
                return False
            if "vs" in m.lower():
                return True
        return None

    df["_home"] = df.apply(_is_home, axis=1)
    df = df.dropna(subset=["_pts", "_home"])
    for abbr, g in df.groupby("_abbr"):
        home = g[g["_home"] == True]  # noqa: E712
        road = g[g["_home"] == False]  # noqa: E712
        if len(home) < 2 or len(road) < 2:
            continue
        h_pts, r_pts = float(home["_pts"].mean()), float(road["_pts"].mean())
        h_opp = float(home["_opp"].mean()) if home["_opp"].notna().any() else None
        r_opp = float(road["_opp"].mean()) if road["_opp"].notna().any() else None
        # Offense: home scoring boost vs road; Defense: points allowed (lower better)
        off_split = h_pts - r_pts
        def_split = 0.0
        if h_opp is not None and r_opp is not None:
            def_split = r_opp - h_opp  # positive => better home defense
        out[str(abbr)] = {
            "team": abbr,
            "home_pts": round(h_pts, 1),
            "road_pts": round(r_pts, 1),
            "home_opp": round(h_opp, 1) if h_opp is not None else None,
            "road_opp": round(r_opp, 1) if r_opp is not None else None,
            "off_split": round(off_split, 2),
            "def_split": round(def_split, 2),
            "home_n": int(len(home)),
            "road_n": int(len(road)),
        }
    return out


def home_road_matchup_residual(
    splits: dict[str, dict[str, Any]],
    home: str,
    away: str,
) -> tuple[float, list[str]]:
    """Small residual from home/road offense-defense splits for this venue."""
    if not splits:
        return 0.0, []
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)
    hs = splits.get(home) or {}
    as_ = splits.get(away) or {}
    notes: list[str] = []
    edge = 0.0
    # Home team plays at home: use half of their home-road off/def split
    if hs:
        off = 0.25 * float(hs.get("off_split") or 0.0)
        deff = 0.20 * float(hs.get("def_split") or 0.0)
        edge += off + deff
        if abs(hs.get("off_split") or 0) >= 4.0:
            notes.append(
                f"{home} home off {hs.get('home_pts')} vs road {hs.get('road_pts')}"
            )
    # Away team on road: suppress by their home-road gap
    if as_:
        off = -0.25 * float(as_.get("off_split") or 0.0)
        deff = -0.15 * float(as_.get("def_split") or 0.0)
        edge += off + deff
        if abs(as_.get("off_split") or 0) >= 4.0:
            notes.append(
                f"{away} road off {as_.get('road_pts')} vs home {as_.get('home_pts')}"
            )
    edge = float(np.clip(edge, -config.HOME_ROAD_EDGE_CAP, config.HOME_ROAD_EDGE_CAP))
    return edge, notes[:2]


def build_quarter_profiles(quarter_logs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-team average Q1-Q4 scoring and opponent allowed."""
    out: dict[str, dict[str, Any]] = {}
    if quarter_logs is None or (isinstance(quarter_logs, pd.DataFrame) and quarter_logs.empty):
        return out
    df = quarter_logs.copy()
    team_c = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None
    if not team_c:
        return out
    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    for abbr, g in df.groupby("_abbr"):
        prof: dict[str, Any] = {"team": abbr, "games": int(len(g))}
        for q in (1, 2, 3, 4):
            col = f"Q{q}"
            opp = f"OPP_Q{q}"
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                prof[f"q{q}_pts"] = float(s.mean()) if len(s) else None
            if opp in g.columns:
                s = pd.to_numeric(g[opp], errors="coerce").dropna()
                prof[f"q{q}_opp"] = float(s.mean()) if len(s) else None
        out[str(abbr)] = prof
    return out


def quarter_matchup_residual(
    profiles: dict[str, dict[str, Any]],
    home: str,
    away: str,
    *,
    clutch: bool = False,
) -> tuple[float, list[str]]:
    """
    Matchup residual from quarter scoring tendencies.
    offense_q(home) - defense_q(away) style, heavily shrunk.
    When clutch=True (close game), overweight Q4.
    """
    if not profiles:
        return 0.0, []
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)
    hp = profiles.get(home) or {}
    ap = profiles.get(away) or {}
    if not hp or not ap:
        return 0.0, []

    # League-ish quarter average ~28
    lg = 28.0
    notes: list[str] = []
    edge = 0.0
    weights = {1: 0.22, 2: 0.22, 3: 0.22, 4: 0.34}
    if clutch:
        weights = {1: 0.15, 2: 0.15, 3: 0.20, 4: 0.50}

    for q, w in weights.items():
        h_off = hp.get(f"q{q}_pts")
        a_off = ap.get(f"q{q}_pts")
        h_def = hp.get(f"q{q}_opp")  # points allowed
        a_def = ap.get(f"q{q}_opp")
        if h_off is None or a_def is None or a_off is None or h_def is None:
            continue
        # Points-allowed: higher = weaker defense, so it ADDS to opponent offense edge
        h_edge = (float(h_off) - lg) + (float(a_def) - lg)
        a_edge = (float(a_off) - lg) + (float(h_def) - lg)
        q_edge = 0.10 * (h_edge - a_edge)  # shrink
        if clutch and q == 4:
            q_edge *= config.CLUTCH_Q4_MULT
        edge += w * q_edge
        if abs(h_edge - a_edge) >= 4.0:
            notes.append(
                f"Q{q}: {home} off {float(h_off):.1f} vs {away} allow {float(a_def):.1f}"
                f" / {away} off {float(a_off):.1f} vs {home} allow {float(h_def):.1f}"
            )

    edge = float(np.clip(edge, -config.QUARTER_EDGE_CAP, config.QUARTER_EDGE_CAP))
    if clutch and abs(edge) >= 0.3:
        notes.insert(0, f"Clutch Q4 weighting on (projected close)")
    return edge, notes[:3]


def _injury_team_matches(inj: dict[str, Any], team: str) -> bool:
    """True when injury row is for this team abbr (or team name contains abbr/city)."""
    team_u = normalize_abbrev(team)
    if not team_u:
        return False
    inj_abbr = normalize_abbrev(str(inj.get("team_abbr") or ""))
    if inj_abbr and inj_abbr == team_u:
        return True
    inj_team = str(inj.get("team") or "").lower()
    if not inj_team:
        return False
    # "Boston Celtics" vs BOS — require abbr already set when possible; else name token
    if inj_abbr:
        return False  # different team_abbr already known
    return team_u.lower() in inj_team


def _match_injury(
    name: str,
    team: str,
    injuries: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Exact name match first; soft last-name match only when the injury is on the
    same team (never apply another team's Coby White Out to Derrick White).
    """
    key = _norm_name(name)
    if key in injuries:
        hit = injuries[key]
        # Exact full-name hit is trusted even if team_abbr missing
        return hit
    parts = [p for p in key.replace(".", "").split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
    last = parts[-1] if parts else ""
    first = parts[0] if parts else ""
    if len(last) < 4:
        return None
    team_u = normalize_abbrev(team)
    if not team_u:
        return None
    same_team: list[dict[str, Any]] = []
    for ik, iv in injuries.items():
        if not _injury_team_matches(iv, team_u):
            continue
        ik_parts = [p for p in ik.replace(".", "").split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
        if not ik_parts or ik_parts[-1] != last:
            continue
        # Prefer first-initial agreement when both have a first name
        if first and ik_parts[0] and ik_parts[0][0] != first[0]:
            continue
        same_team.append(iv)
    if not same_team:
        return None
    return min(same_team, key=lambda c: c.get("minutes_factor", 1.0))


def project_player_minutes(
    players: pd.DataFrame,
    injuries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Season / recent minutes × injury factor (injuries win)."""
    if players is None or players.empty:
        return pd.DataFrame(columns=["player", "team", "proj_minutes", "impact", "adj_minutes"])

    df = players.copy()
    name_c = next((c for c in ("PLAYER_NAME", "player") if c in df.columns), None)
    team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in df.columns), None)
    min_c = next((c for c in ("MIN", "minutes") if c in df.columns), None)
    if not name_c:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        name = str(r.get(name_c))
        team = normalize_abbrev(str(r.get(team_c, ""))) if team_c else ""
        mins = float(pd.to_numeric(r.get(min_c), errors="coerce") or 0.0) if min_c else 0.0
        # Inactive / G-League two-ways with no minutes stay at 0
        inj = _match_injury(name, team, injuries)
        factor = inj["minutes_factor"] if inj else 1.0
        rows.append(
            {
                "player": name,
                "team": team,
                "proj_minutes": mins,
                "minutes_factor": factor,
                "adj_minutes": mins * factor,
                "injury_status": (inj or {}).get("status"),
                "injury_source": (inj or {}).get("source"),
            }
        )
    return pd.DataFrame(rows)


def team_impact_from_players(
    minute_df: pd.DataFrame,
    impact_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Minutes-weighted sum of EPM/RAPM → team strength delta + injury notes."""
    impact_by_name: dict[str, float] = {}
    for row in impact_rows or []:
        name = re.sub(r"\s+", " ", str(row.get("player") or "")).lower()
        if not name:
            continue
        try:
            impact_by_name[name] = float(row.get("epm") or 0.0)
        except (TypeError, ValueError):
            continue

    if minute_df is None or minute_df.empty:
        return {}

    out: dict[str, dict[str, float]] = {}
    notes: dict[str, list[str]] = {}
    for team, g in minute_df.groupby("team"):
        if not team:
            continue
        total = 0.0
        healthy = 0.0
        team_notes: list[str] = []
        for _, r in g.iterrows():
            key = re.sub(r"\s+", " ", str(r["player"])).lower()
            epm = impact_by_name.get(key)
            if epm is None:
                # last-name fallback
                last = key.split()[-1]
                for n, v in impact_by_name.items():
                    if last in n:
                        epm = v
                        break
            try:
                epm = float(epm or 0.0)
            except (TypeError, ValueError):
                epm = 0.0
            if not np.isfinite(epm):
                epm = 0.0
            mins = float(r["adj_minutes"] or 0.0)
            base_mins = float(r["proj_minutes"] or 0.0)
            if not np.isfinite(mins):
                mins = 0.0
            if not np.isfinite(base_mins):
                base_mins = 0.0
            # Rotation players are above replacement even when EPM scrapes are noisy
            # or negative early-season z-scores — otherwise Out looks like a "help".
            if base_mins >= config.ROTATION_MIN_MINUTES:
                epm = max(epm, config.ROTATION_OUT_EPM_FLOOR)
            total += epm * (mins / 48.0)
            healthy += epm * (base_mins / 48.0)
            if r["minutes_factor"] < 0.95 and base_mins >= 15:
                lost = epm * ((base_mins - mins) / 48.0)
                team_notes.append(
                    f"{r['player']} ({r.get('injury_status') or 'injured'}): "
                    f"{base_mins:.0f}->{mins:.0f} min, d{lost:+.2f} EPM-w"
                )
        out[team] = {
            "player_impact": total,
            "healthy_impact": healthy,
            "injury_delta": total - healthy,
        }

        def _note_mag(s: str) -> float:
            # Parse trailing "d+1.23 EPM-w" (ASCII; was Greek Delta before)
            try:
                token = s.rsplit("d", 1)[-1].replace(" EPM-w", "").strip()
                return abs(float(token))
            except (TypeError, ValueError, IndexError):
                return 0.0

        notes[team] = sorted(team_notes, key=_note_mag, reverse=True)[:4]
        out[team]["injury_notes"] = notes[team]  # type: ignore[assignment]
    return out


def rest_and_travel_context(
    games: list[dict[str, Any]],
    team_logs: pd.DataFrame,
    target: date,
    ratings: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-team rest days, B2B, travel miles, TZ shift into today's venue.

    Also returns pace_adj: tired teams (B2B / short rest) play slower; well-rested
    slightly faster — applied in the projection pace layer.
    """
    last_game: dict[str, date] = {}
    last_venue: dict[str, str] = {}
    pace_by_team: dict[str, float] = {}
    if ratings is not None and not ratings.empty and "team" in ratings.columns:
        for _, r in ratings.iterrows():
            try:
                pace_by_team[normalize_abbrev(str(r["team"]))] = float(r.get("pace") or config.DEFAULT_PACE)
            except (TypeError, ValueError):
                continue

    if team_logs is not None and not team_logs.empty:
        df = team_logs.copy()
        colmap = {c.upper(): c for c in df.columns}
        team_c = colmap.get("TEAM_ABBREVIATION")
        date_c = colmap.get("GAME_DATE")
        match_c = colmap.get("MATCHUP")
        if team_c and date_c:
            df["_date"] = df[date_c].apply(_parse_game_date)
            df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
            df = df.dropna(subset=["_date"]).sort_values("_date")
            for _, r in df.iterrows():
                if r["_date"] >= target:
                    continue
                abbr = r["_abbr"]
                last_game[abbr] = r["_date"]
                # MATCHUP like "BOS @ MIA" or "BOS vs. MIA"
                venue = abbr
                if match_c and isinstance(r.get(match_c), str):
                    m = str(r[match_c])
                    if "@" in m:
                        parts = m.split("@")
                        venue = normalize_abbrev(parts[-1].strip().split()[-1])
                    else:
                        venue = abbr
                last_venue[abbr] = venue

    ctx: dict[str, dict[str, Any]] = {}
    for g in games:
        home = normalize_abbrev(g.get("home_abbr"))
        away = normalize_abbrev(g.get("away_abbr"))
        for team, is_home in ((home, True), (away, False)):
            if not team:
                continue
            prev = last_game.get(team)
            rest = (target - prev).days if prev else 3
            b2b = rest <= 1
            prev_venue = last_venue.get(team, team)
            miles = travel_miles(prev_venue, home)
            tz = timezone_shift_hours(prev_venue, home)
            # Point residual: B2B / travel / TZ (small)
            adj = 0.0
            pace_adj = 0.0  # possessions delta
            flags = []
            team_pace = pace_by_team.get(team, config.DEFAULT_PACE)
            if b2b:
                adj -= 1.5
                # Tired teams slow down more if they normally push pace
                pace_adj -= 1.2 + 0.04 * max(0.0, team_pace - config.DEFAULT_PACE)
                flags.append("B2B")
                flags.append("tired-pace")
            elif rest <= 1:
                adj -= 0.6
                pace_adj -= 0.6
                flags.append(f"rest={rest}d")
            elif rest >= 3:
                adj += 0.3
                pace_adj += 0.4
                flags.append(f"rest={rest}d")
            else:
                flags.append(f"rest={rest}d")
            if miles >= 1500:
                adj -= 0.8
                pace_adj -= 0.5
                flags.append(f"travel={miles:.0f}mi")
            elif miles >= 800:
                adj -= 0.4
                pace_adj -= 0.25
                flags.append(f"travel={miles:.0f}mi")
            if abs(tz) >= 2:
                adj -= 0.4 * abs(tz) / 2
                flags.append(f"TZ{tz:+.0f}h")
            if is_home:
                adj += config.HOME_COURT_POINTS
                flags.append("home")
            else:
                flags.append("away")
            ctx[f"{g.get('game_id')}:{team}"] = {
                "team": team,
                "game_id": g.get("game_id"),
                "rest_days": rest,
                "b2b": b2b,
                "travel_miles": miles,
                "tz_shift": tz,
                "context_points": adj,
                "pace_adj": float(np.clip(pace_adj, -3.0, 2.0)),
                "flags": flags,
            }
    return ctx


def playtype_matchup_residual(
    play_types: pd.DataFrame,
    home: str,
    away: str,
) -> tuple[float, list[str]]:
    """Small residual from offensive play-type PPP vs opponent (shrunk)."""
    if play_types is None or play_types.empty:
        return 0.0, []
    df = play_types.copy()
    team_c = next((c for c in ("TEAM_ABBREVIATION", "TeamAbbreviation", "team") if c in df.columns), None)
    pt_c = next((c for c in ("PLAY_TYPE", "play_type") if c in df.columns), None)
    ppp_c = next((c for c in ("PPP", "POINTS", "PTS") if c in df.columns), None)
    if not team_c or not pt_c or not ppp_c:
        return 0.0, []

    df["_team"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_ppp"] = pd.to_numeric(df[ppp_c], errors="coerce")
    league = df.groupby(pt_c)["_ppp"].mean()

    notes = []
    edge = 0.0
    for team, opp in ((home, away), (away, home)):
        tg = df[df["_team"] == team]
        if tg.empty:
            continue
        for _, r in tg.iterrows():
            pt = r[pt_c]
            ppp = r["_ppp"]
            if pd.isna(ppp) or pt not in league.index:
                continue
            # Without true defensive ranks, use deviation from league as soft edge
            delta = float(ppp - league.loc[pt])
            # shrink heavily
            contrib = 0.15 * delta * (1 if team == home else -1)
            edge += contrib
            if abs(delta) >= 0.08:
                notes.append(f"{team} {pt} PPP {ppp:.2f} (lg {league.loc[pt]:.2f})")

    edge = float(np.clip(edge, -config.MATCHUP_RESIDUAL_CAP, config.MATCHUP_RESIDUAL_CAP))
    notes = notes[:3]
    return edge, notes


def _ratings_from_espn_season(team_adv: pd.DataFrame) -> pd.DataFrame:
    """When game-log box scores are thin, seed ratings from ESPN season averages."""
    if team_adv is None or team_adv.empty:
        return pd.DataFrame()
    if "TEAM_ABBREVIATION" not in team_adv.columns and "team" in team_adv.columns:
        return team_adv
    try:
        from data.espn import enrich_ratings_from_season_stats

        rows = enrich_ratings_from_season_stats(team_adv.to_dict(orient="records"))
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN season enrichment failed: %s", exc)
        return pd.DataFrame()


def build_features(raw: dict[str, Any], target: date) -> dict[str, Any]:
    """Top-level feature builder consumed by the model."""
    team_roll = build_team_rolling(raw.get("team_logs"))
    prior_roll = build_team_rolling(raw.get("prior_logs"))
    ratings = blend_with_prior(team_roll, prior_roll)

    # Early season / score-only logs: blend in ESPN season averages for FF & pace
    espn_seed = _ratings_from_espn_season(raw.get("team_adv"))
    if not espn_seed.empty:
        if ratings is None or ratings.empty:
            ratings = espn_seed
        else:
            # Keep prior heavier when sample is thin
            ratings = blend_with_prior(
                ratings,
                espn_seed,
                n_blend=max(config.PRIOR_BLEND_GAMES, config.EARLY_SEASON_GAMES * 2),
            )

    players = align_players_to_rosters(raw.get("players"), raw.get("rosters") or [])
    inj_idx = index_injuries(raw.get("injuries") or [])
    minutes = project_player_minutes(players, inj_idx)
    impact = team_impact_from_players(minutes, raw.get("impact") or [])
    context = rest_and_travel_context(
        raw.get("games") or [],
        raw.get("team_logs"),
        target,
        ratings=ratings,
    )
    team_form = build_team_form(raw.get("team_logs"))
    home_road = build_home_road_splits(raw.get("team_logs"))
    quarter_profiles = build_quarter_profiles(raw.get("quarter_logs"))

    return {
        "ratings": ratings,
        "injuries": inj_idx,
        "minutes": minutes,
        "team_impact": impact,
        "context": context,
        "team_form": team_form,
        "player_form": raw.get("player_form") or {},
        "home_road": home_road,
        "quarter_profiles": quarter_profiles,
        "play_types": raw.get("play_types"),
        "games": raw.get("games") or [],
        "team_logs": raw.get("team_logs"),
        "players": players,
        "rosters": raw.get("rosters") or [],
        "target_date": target.isoformat(),
    }


def verify_matchup_personnel(
    features: dict[str, Any],
    away: str,
    home: str,
) -> dict[str, Any]:
    """
    Short verification snapshot for a matchup: roster sizes, key players,
    injury flags, and projected minutes — logged/printed to spot bad data.
    """
    away = normalize_abbrev(away)
    home = normalize_abbrev(home)
    minutes = features.get("minutes")
    if minutes is None or (isinstance(minutes, pd.DataFrame) and minutes.empty):
        return {"away": away, "home": home, "error": "no minutes frame"}

    def _side(abbr: str) -> dict[str, Any]:
        g = minutes[minutes["team"] == abbr].copy()
        g = g.sort_values("proj_minutes", ascending=False)
        top = g.head(8)
        injured = g[g["minutes_factor"] < 0.95].sort_values("proj_minutes", ascending=False)

        def _status(val: Any) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            s = str(val).strip()
            return "" if s.lower() in {"nan", "none"} else s

        return {
            "team": abbr,
            "roster_size": int(len(g)),
            "key_players": [
                {
                    "player": str(r["player"]),
                    "proj_min": round(float(r["proj_minutes"] or 0), 1),
                    "adj_min": round(float(r["adj_minutes"] or 0), 1),
                    "injury": _status(r.get("injury_status")),
                }
                for _, r in top.iterrows()
            ],
            "injury_flags": [
                {
                    "player": str(r["player"]),
                    "status": _status(r.get("injury_status")),
                    "factor": float(r["minutes_factor"]),
                    "proj_min": round(float(r["proj_minutes"] or 0), 1),
                    "adj_min": round(float(r["adj_minutes"] or 0), 1),
                }
                for _, r in injured.head(8).iterrows()
            ],
        }

    report = {"away": _side(away), "home": _side(home)}
    for side in ("away", "home"):
        s = report[side]
        logger.info(
            "VERIFY %s roster=%d | top: %s",
            s["team"],
            s["roster_size"],
            ", ".join(
                f"{p['player']} {p['proj_min']}->{p['adj_min']}m"
                + (f" [{p['injury']}]" if p["injury"] else "")
                for p in s["key_players"][:5]
            ),
        )
        if s["injury_flags"]:
            logger.info(
                "VERIFY %s injuries: %s",
                s["team"],
                "; ".join(
                    f"{p['player']}={p['status']} ({p['proj_min']}->{p['adj_min']})"
                    for p in s["injury_flags"][:5]
                ),
            )
    return report

