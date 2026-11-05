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
from safe_text import normalize_player_name

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


def injury_factor_distribution(status: str) -> tuple[float, float, float]:
    """
    Probabilistic injury model: (p_out, p_limited, p_full) approximating
    status uncertainty. Means match minutes_factor_from_status roughly.
    """
    if not status:
        return (0.0, 0.0, 1.0)
    if OUT_STATUSES.search(status):
        return (0.92, 0.06, 0.02)
    if DOUBTFUL.search(status):
        return (0.70, 0.25, 0.05)
    if QUESTIONABLE.search(status):
        return (0.30, 0.45, 0.25)
    if PROBABLE.search(status):
        return (0.05, 0.20, 0.75)
    return (0.0, 0.0, 1.0)


def sample_injury_factor(status: str, rng: np.random.Generator) -> float:
    """Draw a minutes factor from the injury status distribution."""
    p_out, p_lim, p_full = injury_factor_distribution(status)
    u = float(rng.random())
    if u < p_out:
        return 0.0
    if u < p_out + p_lim:
        # Limited: around 0.45–0.70 of projected minutes
        return float(0.45 + 0.25 * rng.random())
    return 1.0


def _norm_name(name: str) -> str:
    """Accent-folding name key (Dončić ≡ Doncic). Prefer normalize_player_name."""
    return normalize_player_name(name)


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
        key = normalize_player_name(name)
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
        key = normalize_player_name(r.get("PLAYER_NAME") or "")
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
                    "PLAYER_ID": r.get("PLAYER_ID"),
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
        key = normalize_player_name(name)
        rost = roster_by_name.get(key)
        if rost is None:
            # Soft last-name+first-initial match against roster (folded names)
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
        # Keep season counting stats; prefer roster display name + ESPN id
        d["PLAYER_NAME"] = rost.get("PLAYER_NAME") or name
        d["TEAM_ABBREVIATION"] = normalize_abbrev(rost["TEAM_ABBREVIATION"])
        d["ON_ROSTER"] = True
        d["ROSTER_STATUS"] = rost.get("STATUS")
        if rost.get("PLAYER_ID") is not None:
            d["PLAYER_ID"] = rost.get("PLAYER_ID")
        d["SOURCE"] = "season+roster"
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
                "PLAYER_ID": rost.get("PLAYER_ID"),
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
    Multi-horizon form: blend L5 / L10 / season margin gaps into form_edge.
    Also tags HOT/COLD when material. Returns per-team form_edge (capped).
    """
    last_n = last_n or config.FORM_LAST_N
    l10_n = config.FORM_L10_N
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

    def _avg_pm(frame: pd.DataFrame) -> float:
        if frame.empty or not frame["_pm"].notna().any():
            return 0.0
        return float(frame["_pm"].mean())

    for abbr, g in df.groupby("_abbr"):
        g = g.dropna(subset=["_pts"]).sort_values("_date")
        if len(g) < 3:
            continue
        season_margin = _avg_pm(g)
        season_pts = float(g["_pts"].mean())
        l5 = g.tail(last_n)
        l10 = g.tail(l10_n)
        l5_margin = _avg_pm(l5)
        l10_margin = _avg_pm(l10)
        l5_pts = float(l5["_pts"].mean())
        l10_pts = float(l10["_pts"].mean())

        # Residuals vs season; season term keeps mild regression toward 0
        r5 = 0.55 * (l5_margin - season_margin) + 0.20 * (l5_pts - season_pts)
        r10 = 0.55 * (l10_margin - season_margin) + 0.15 * (l10_pts - season_pts)
        r_szn = 0.15 * season_margin  # weak season-level net pull into form
        w5, w10, wsz = config.FORM_W_L5, config.FORM_W_L10, config.FORM_W_SEASON
        wsum = max(w5 + w10 + wsz, 1e-6)
        raw = (w5 * r5 + w10 * r10 + wsz * r_szn) / wsum
        edge = float(np.clip(raw, -config.FORM_EDGE_CAP, config.FORM_EDGE_CAP))

        streak = 0
        if plus_c and l5["_pm"].notna().any():
            signs = list(l5["_pm"].dropna().map(lambda x: 1 if x > 0 else -1))
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
            if edge < 1.25:
                edge = max(edge, min(1.5, 0.45 * abs(streak)))
        elif edge <= -1.25 or streak <= -3:
            tag = "COLD"
            if edge > -1.25:
                edge = min(edge, -min(1.5, 0.45 * abs(streak)))
        edge = float(np.clip(edge, -config.FORM_EDGE_CAP, config.FORM_EDGE_CAP))
        note = None
        if tag != "NEUTRAL" or abs(edge) >= 1.0:
            note = (
                f"{abbr} {tag if tag != 'NEUTRAL' else 'FORM'}: "
                f"L{len(l5)} {l5_margin:+.1f} / L{len(l10)} {l10_margin:+.1f} / "
                f"szn {season_margin:+.1f} -> edge {edge:+.1f}"
                + (f", streak {streak:+d}" if abs(streak) >= 3 else "")
            )
        out[str(abbr)] = {
            "team": abbr,
            "tag": tag,
            "form_edge": round(edge, 2),
            "l5_margin": round(l5_margin, 2),
            "l10_margin": round(l10_margin, 2),
            "recent_margin": round(l5_margin, 2),
            "season_margin": round(season_margin, 2),
            "recent_pts": round(l5_pts, 1),
            "season_pts": round(season_pts, 1),
            "streak": int(streak),
            "n": int(len(l5)),
            "n10": int(len(l10)),
            "note": note,
        }
    return out


def compute_srs_ratings(
    team_logs: pd.DataFrame,
    *,
    iterations: int | None = None,
) -> dict[str, float]:
    """
    Iterative SRS-style schedule-strength adjusted margin ratings.
    Returns team -> SRS points (approx expected margin vs average team).
    """
    iterations = iterations if iterations is not None else config.SRS_ITERATIONS
    if team_logs is None or (isinstance(team_logs, pd.DataFrame) and team_logs.empty):
        return {}

    df = team_logs.copy()
    colmap = {c.upper(): c for c in df.columns}
    team_c = colmap.get("TEAM_ABBREVIATION") or (
        "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None
    )
    date_c = colmap.get("GAME_DATE") or ("GAME_DATE" if "GAME_DATE" in df.columns else None)
    plus_c = colmap.get("PLUS_MINUS") or ("PLUS_MINUS" if "PLUS_MINUS" in df.columns else None)
    match_c = colmap.get("MATCHUP") or ("MATCHUP" if "MATCHUP" in df.columns else None)
    if not team_c or not plus_c:
        return {}

    df["_abbr"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_pm"] = pd.to_numeric(df[plus_c], errors="coerce")
    if date_c:
        df["_date"] = df[date_c].apply(_parse_game_date)
        df = df.dropna(subset=["_date", "_pm"])
    else:
        df = df.dropna(subset=["_pm"])

    # Infer opponent from MATCHUP when possible
    def _opp(row: pd.Series) -> str:
        if not match_c or not isinstance(row.get(match_c), str):
            return ""
        m = str(row[match_c])
        parts = re.split(r"\s+(@|vs\.?)\s+", m, flags=re.I)
        tokens = [normalize_abbrev(t) for t in re.findall(r"[A-Za-z]{2,3}", m)]
        me = row["_abbr"]
        for t in tokens:
            if t and t != me and t in {normalize_abbrev(x) for x in df["_abbr"].unique()}:
                return t
        return ""

    df["_opp"] = df.apply(_opp, axis=1)
    teams = sorted({a for a in df["_abbr"].unique() if a})
    if not teams:
        return {}

    # Seed with average margin
    srs = {t: float(df.loc[df["_abbr"] == t, "_pm"].mean()) for t in teams}
    for _ in range(max(1, iterations)):
        nxt: dict[str, float] = {}
        for t in teams:
            g = df[df["_abbr"] == t]
            vals = []
            for _, r in g.iterrows():
                opp = r["_opp"]
                opp_r = srs.get(opp, 0.0) if opp else 0.0
                vals.append(float(r["_pm"]) + opp_r)
            nxt[t] = float(np.mean(vals)) if vals else 0.0
        # Center to league mean 0
        mean_r = float(np.mean(list(nxt.values()))) if nxt else 0.0
        srs = {t: v - mean_r for t, v in nxt.items()}
    return srs


def apply_srs_to_ratings(
    ratings: pd.DataFrame,
    srs: dict[str, float],
    *,
    blend: float | None = None,
) -> pd.DataFrame:
    """Blend SRS net into team off/def so soft schedules don't inflate raw ratings."""
    if ratings is None or ratings.empty or not srs:
        return ratings
    blend = config.SRS_BLEND if blend is None else blend
    out = ratings.copy()
    for i, row in out.iterrows():
        team = normalize_abbrev(str(row.get("team") or ""))
        if team not in srs:
            continue
        raw_net = float(row.get("net_rtg") or (float(row.get("off_rtg", 0)) - float(row.get("def_rtg", 0))))
        adj_net = (1.0 - blend) * raw_net + blend * float(srs[team])
        delta = adj_net - raw_net
        out.at[i, "off_rtg"] = float(row["off_rtg"]) + 0.55 * delta
        out.at[i, "def_rtg"] = float(row["def_rtg"]) - 0.45 * delta
        out.at[i, "net_rtg"] = float(out.at[i, "off_rtg"]) - float(out.at[i, "def_rtg"])
        out.at[i, "srs"] = float(srs[team])
    return out


def apply_form_to_ratings(
    ratings: pd.DataFrame,
    team_form: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Fold multi-horizon form_edge into off/def (moderate; predict also uses notes)."""
    if ratings is None or ratings.empty or not team_form:
        return ratings
    out = ratings.copy()
    for i, row in out.iterrows():
        team = normalize_abbrev(str(row.get("team") or ""))
        edge = float((team_form.get(team) or {}).get("form_edge") or 0.0)
        if abs(edge) < 0.25:
            continue
        # Half of form already applied here; predict still uses a shrunk residual
        bump = 0.35 * edge
        out.at[i, "off_rtg"] = float(row["off_rtg"]) + 0.6 * bump
        out.at[i, "def_rtg"] = float(row["def_rtg"]) - 0.4 * bump
        out.at[i, "net_rtg"] = float(out.at[i, "off_rtg"]) - float(out.at[i, "def_rtg"])
    return out


def shot_profile_residual(
    team_adv: pd.DataFrame | None,
    home: str,
    away: str,
) -> tuple[float, list[str]]:
    """
    Shot/defense matchup: offense 3PA/g vs opponent 3PA allowed proxy +
    FG3% vs opp points allowed tilt. Heavy shrinkage; ASCII notes.
    """
    if team_adv is None or (isinstance(team_adv, pd.DataFrame) and team_adv.empty):
        return 0.0, []
    df = team_adv.copy()
    team_c = next(
        (c for c in ("TEAM_ABBREVIATION", "team", "TEAM") if c in df.columns),
        None,
    )
    if not team_c:
        return 0.0, []
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)

    def _row(abbr: str) -> dict[str, Any] | None:
        hit = df[df[team_c].map(lambda x: normalize_abbrev(str(x))) == abbr]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

    hr, ar = _row(home), _row(away)
    if not hr or not ar:
        return 0.0, []

    def _num(row: dict[str, Any], *keys: str) -> float | None:
        for k in keys:
            if k in row and row[k] is not None:
                try:
                    v = float(row[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    return v
            # Nested RAW dict from ESPN
            raw = row.get("RAW") if isinstance(row.get("RAW"), dict) else {}
            if k in raw and raw[k] is not None:
                try:
                    v = float(raw[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    return v
        return None

    h3 = _num(hr, "AVG_FG3A", "avgThreePointFieldGoalsAttempted", "FG3A")
    a3 = _num(ar, "AVG_FG3A", "avgThreePointFieldGoalsAttempted", "FG3A")
    # Opp 3PA allowed proxy: league ± (opp_pts tilt) when explicit allow missing
    h_opp = _num(hr, "OPP_PTS", "avgPointsAllowed", "avgOpponentPoints")
    a_opp = _num(ar, "OPP_PTS", "avgPointsAllowed", "avgOpponentPoints")
    h_fg3 = _num(hr, "FG3_PCT", "threePointPct", "threePointFieldGoalPct")
    a_fg3 = _num(ar, "FG3_PCT", "threePointPct", "threePointFieldGoalPct")
    if h3 is None or a3 is None:
        return 0.0, []
    league_3 = 37.0
    league_opp = config.LEAGUE_AVG_TOTAL / 2.0
    # Offense volume vs opponent "softness" (higher opp pts allowed => softer D)
    h_soft = ((a_opp or league_opp) - league_opp) / 10.0
    a_soft = ((h_opp or league_opp) - league_opp) / 10.0
    vol_edge = 0.035 * ((h3 - league_3) * (1.0 + 0.15 * h_soft) - (a3 - league_3) * (1.0 + 0.15 * a_soft))
    pct_edge = 0.0
    if h_fg3 is not None and a_fg3 is not None:
        # Scale FG3% (0.36 style) to points-ish residual
        pct_edge = 8.0 * ((h_fg3 - 0.36) - (a_fg3 - 0.36))
        pct_edge *= 0.08
    edge = float(np.clip(vol_edge + pct_edge, -1.4, 1.4))
    notes = []
    if abs(h3 - a3) >= 5.0:
        notes.append(f"3PA/g: {home} {h3:.1f} vs {away} {a3:.1f}")
    if h_opp is not None and a_opp is not None and abs((a_opp or 0) - (h_opp or 0)) >= 4.0:
        notes.append(f"Opp pts allow: {home} {h_opp:.1f} / {away} {a_opp:.1f}")
    return edge, notes


def four_factor_matchup_residual(
    ratings: pd.DataFrame | None,
    home: str,
    away: str,
) -> tuple[float, list[str]]:
    """
    Degrade path when Synergy play-types are empty: eFG / TOV / ORB / FT rate
    offense vs opponent defense proxies from the ratings frame.
    """
    if ratings is None or (isinstance(ratings, pd.DataFrame) and ratings.empty):
        return 0.0, []
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)

    def _row(abbr: str) -> dict[str, Any] | None:
        hit = ratings[ratings["team"].map(lambda x: normalize_abbrev(str(x))) == abbr]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

    hr, ar = _row(home), _row(away)
    if not hr or not ar:
        return 0.0, []

    def _f(row: dict[str, Any], key: str, default: float) -> float:
        try:
            v = float(row.get(key))
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    # Home offense vs away "defense" (inverse of away eFG allowed ≈ use away eFG as softness)
    h_off = (
        12.0 * (_f(hr, "efg", 0.53) - 0.53)
        - 20.0 * (_f(hr, "tov_pct", 0.13) - 0.13)
        + 6.0 * (_f(hr, "orb_pct", 0.28) - 0.28)
        + 5.0 * (_f(hr, "ft_rate", 0.25) - 0.25)
    )
    a_off = (
        12.0 * (_f(ar, "efg", 0.53) - 0.53)
        - 20.0 * (_f(ar, "tov_pct", 0.13) - 0.13)
        + 6.0 * (_f(ar, "orb_pct", 0.28) - 0.28)
        + 5.0 * (_f(ar, "ft_rate", 0.25) - 0.25)
    )
    # Opponent defensive quality from net/def_rtg
    h_def_q = (_f(hr, "def_rtg", config.DEFAULT_DEF_RTG) - config.DEFAULT_DEF_RTG) / 8.0
    a_def_q = (_f(ar, "def_rtg", config.DEFAULT_DEF_RTG) - config.DEFAULT_DEF_RTG) / 8.0
    # Positive edge = home advantage from FF matchup
    edge = 0.12 * ((h_off - a_def_q) - (a_off - h_def_q))
    edge = float(np.clip(edge, -config.MATCHUP_RESIDUAL_CAP, config.MATCHUP_RESIDUAL_CAP))
    notes = []
    if abs(edge) >= 0.35:
        notes.append(
            f"FF matchup: {home} eFG {_f(hr, 'efg', 0.53):.3f} vs {away} eFG {_f(ar, 'efg', 0.53):.3f}"
        )
    return edge, notes


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


def project_quarter_breakdown(
    profiles: dict[str, dict[str, Any]],
    home: str,
    away: str,
    home_pts: float,
    away_pts: float,
    *,
    clutch: bool = False,
    pace: float | None = None,
) -> dict[str, Any]:
    """
    Projected Q1-Q4 scores for both teams from tendency profiles, scaled so
    each team's quarters sum to the sim-mean team total.
    """
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)
    hp = (profiles or {}).get(home) or {}
    ap = (profiles or {}).get(away) or {}
    lg = 28.0

    def _shares(prof: dict[str, Any]) -> list[float]:
        raw: list[float] = []
        for q in (1, 2, 3, 4):
            v = prof.get(f"q{q}_pts")
            try:
                fv = float(v) if v is not None else lg
            except (TypeError, ValueError):
                fv = lg
            if not np.isfinite(fv) or fv <= 0:
                fv = lg
            raw.append(fv)
        s = sum(raw) or 1.0
        return [x / s for x in raw]

    def _scale(shares: list[float], total: float) -> list[float]:
        total = float(total) if np.isfinite(total) else 4.0 * lg
        return [round(total * w, 1) for w in shares]

    h_shares = _shares(hp)
    a_shares = _shares(ap)
    h_proj = _scale(h_shares, home_pts)
    a_proj = _scale(a_shares, away_pts)

    # Tiny residual so displayed quarters still sum to team totals after rounding
    def _fix_sum(vals: list[float], target: float) -> list[float]:
        if not vals:
            return vals
        diff = round(float(target) - sum(vals), 1)
        vals = list(vals)
        vals[-1] = round(vals[-1] + diff, 1)
        return vals

    h_proj = _fix_sum(h_proj, home_pts)
    a_proj = _fix_sum(a_proj, away_pts)

    quarters: list[dict[str, Any]] = []
    for i, q in enumerate((1, 2, 3, 4)):
        h_off = hp.get(f"q{q}_pts")
        a_off = ap.get(f"q{q}_pts")
        h_def = hp.get(f"q{q}_opp")
        a_def = ap.get(f"q{q}_opp")
        h_tend = round(float(h_off), 1) if h_off is not None else None
        a_tend = round(float(a_off), 1) if a_off is not None else None
        h_allow = round(float(h_def), 1) if h_def is not None else None
        a_allow = round(float(a_def), 1) if a_def is not None else None
        edge = round(h_proj[i] - a_proj[i], 1)
        # Tendency matchup edge (offense vs opp allow), unscaled
        tend_edge = None
        if None not in (h_off, a_off, h_def, a_def):
            h_edge = (float(h_off) - lg) + (float(a_def) - lg)
            a_edge = (float(a_off) - lg) + (float(h_def) - lg)
            tend_edge = round(0.10 * (h_edge - a_edge), 2)
        quarters.append(
            {
                "quarter": q,
                "label": f"Q{q}",
                "away_pts": a_proj[i],
                "home_pts": h_proj[i],
                "edge_home": edge,
                "away_tendency": a_tend,
                "home_tendency": h_tend,
                "away_allow": a_allow,
                "home_allow": h_allow,
                "tendency_edge_home": tend_edge,
            }
        )

    pace_f = float(pace) if pace is not None and np.isfinite(pace) else None
    # Rough possessions proxy per quarter (~25% of pace)
    poss_q = round(pace_f / 4.0, 1) if pace_f else None
    efficiency = {
        "pace": round(pace_f, 1) if pace_f else None,
        "poss_per_quarter_proxy": poss_q,
        "away_pts_per_poss_proxy": (
            round(float(away_pts) / pace_f, 3) if pace_f and pace_f > 0 else None
        ),
        "home_pts_per_poss_proxy": (
            round(float(home_pts) / pace_f, 3) if pace_f and pace_f > 0 else None
        ),
        "clutch_q4": bool(clutch),
        "away_games_sample": int(ap.get("games") or 0),
        "home_games_sample": int(hp.get("games") or 0),
    }

    return {
        "away_abbr": away,
        "home_abbr": home,
        "away_total": round(float(away_pts), 1),
        "home_total": round(float(home_pts), 1),
        "quarters": quarters,
        "efficiency": efficiency,
        "basis": "tendency_scaled_to_sim_mean",
    }


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
    key = normalize_player_name(name)
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
        # Injury keys are already folded via index_injuries
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
    recent_minutes: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Projected minutes = blend(last-N minutes, season) × injury factor.
    Injury / day-of inactive always wins (factor 0 for Out).
    Healthy roster players with no/thin season+recent minutes get a conservative
    rotation floor so join failures never silently drop them from the active roster.
    """
    if players is None or players.empty:
        return pd.DataFrame(columns=["player", "team", "proj_minutes", "impact", "adj_minutes"])

    df = players.copy()
    name_c = next((c for c in ("PLAYER_NAME", "player") if c in df.columns), None)
    team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in df.columns), None)
    min_c = next((c for c in ("MIN", "minutes") if c in df.columns), None)
    if not name_c:
        return pd.DataFrame()

    recent_minutes = recent_minutes or {}
    w_recent = float(np.clip(config.MINUTES_RECENT_WEIGHT, 0.0, 1.0))
    floor = float(config.HEALTHY_ROSTER_MINUTES_FLOOR)
    rows = []
    for _, r in df.iterrows():
        name = str(r.get(name_c))
        key = normalize_player_name(name)
        team = normalize_abbrev(str(r.get(team_c, ""))) if team_c else ""
        season_mins = float(pd.to_numeric(r.get(min_c), errors="coerce") or 0.0) if min_c else 0.0
        # Prior/career-ish minutes column if a fetch path already attached one
        prior_mins = None
        for pc in ("PRIOR_MIN", "prior_minutes", "CAREER_MIN", "career_minutes"):
            if pc in df.columns and r.get(pc) is not None:
                try:
                    prior_mins = float(pd.to_numeric(r.get(pc), errors="coerce"))
                except (TypeError, ValueError):
                    prior_mins = None
                if prior_mins is not None and (not np.isfinite(prior_mins) or prior_mins <= 0):
                    prior_mins = None
                if prior_mins is not None:
                    break
        rec = recent_minutes.get(key) or {}
        recent_mins = rec.get("recent_minutes")
        try:
            recent_mins_f = float(recent_mins) if recent_mins is not None else None
        except (TypeError, ValueError):
            recent_mins_f = None
        if recent_mins_f is not None and recent_mins_f > 0:
            if season_mins > 0:
                base_season = season_mins
                minutes_source = "recent+season"
            elif prior_mins is not None and prior_mins > 0:
                base_season = prior_mins
                minutes_source = "recent+prior"
            else:
                base_season = 0.0
                minutes_source = "recent"
            mins = w_recent * recent_mins_f + (1.0 - w_recent) * base_season
        elif season_mins > 0:
            mins = season_mins
            minutes_source = "season"
        elif prior_mins is not None and prior_mins > 0:
            mins = prior_mins
            minutes_source = "prior"
        else:
            mins = 0.0
            minutes_source = "none"
        # Usage nudge: high FGA/36 slightly protects minutes for stars
        usage = rec.get("usage_fga36")
        try:
            usage_f = float(usage) if usage is not None else None
        except (TypeError, ValueError):
            usage_f = None
        if usage_f is not None and usage_f >= 18.0 and mins >= 20.0:
            mins = min(mins + 0.8, 40.0)

        inj = _match_injury(name, team, injuries)
        factor = inj["minutes_factor"] if inj else 1.0
        # Healthy roster-only / join-miss players: never silent 0 (Out still zeros via factor)
        roster_status = str(r.get("ROSTER_STATUS") or "active").lower()
        source = str(r.get("SOURCE") or "")
        on_roster = bool(r.get("ON_ROSTER", True))
        if (
            mins < 1.0
            and factor > 0.0
            and on_roster
            and roster_status not in {"inactive", "out"}
            and source == "roster_only"
        ):
            mins = floor
            minutes_source = "rotation_floor"
        rows.append(
            {
                "player": name,
                "team": team,
                "proj_minutes": round(mins, 2),
                "season_minutes": round(season_mins, 2),
                "recent_minutes": round(recent_mins_f, 2) if recent_mins_f is not None else None,
                "minutes_source": minutes_source,
                "usage_fga36": round(usage_f, 2) if usage_f is not None else None,
                "minutes_factor": factor,
                "adj_minutes": round(mins * factor, 2),
                "injury_status": (inj or {}).get("status"),
                "injury_source": (inj or {}).get("source"),
                # Possession/usage proxy: FGA/36 from recent logs (filled below)
                "usage_share": None,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _attach_usage_shares(out)
    return out


def _attach_usage_shares(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per-team usage proxy (FGA/36 * adj_minutes) into shares."""
    df = minute_df.copy()
    shares = np.zeros(len(df), dtype=float)
    for team, idx in df.groupby("team").groups.items():
        if not team:
            continue
        sub = df.loc[idx]
        usage = pd.to_numeric(sub.get("usage_fga36"), errors="coerce").fillna(12.0)
        mins = pd.to_numeric(sub.get("adj_minutes"), errors="coerce").fillna(0.0)
        # Only active players contribute to share mass
        mass = (np.maximum(usage, 4.0) ** float(config.USAGE_SHARE_POWER)) * np.maximum(mins, 0.0)
        total = float(mass.sum())
        if total <= 1e-9:
            # Equal among anyone with minutes
            active = mins > 0
            n_act = int(active.sum()) or len(sub)
            for i, ix in enumerate(idx):
                shares[df.index.get_loc(ix)] = (1.0 / n_act) if (mins.iloc[i] > 0 or n_act == len(sub)) else 0.0
        else:
            for i, ix in enumerate(idx):
                shares[df.index.get_loc(ix)] = float(mass.iloc[i] / total)
    df["usage_share"] = np.round(shares, 4)
    return df


def redistribute_minutes_for_injuries(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spill minutes from Out/near-Out rotation players onto healthy teammates.
    Injury-first: Out stay at 0; healthy get capped bumps proportional to usage.
    """
    if minute_df is None or minute_df.empty or not config.INJURY_MINUTES_REDISTRIBUTE:
        return minute_df
    df = minute_df.copy()
    if "adj_minutes" not in df.columns or "proj_minutes" not in df.columns:
        return df
    df["minutes_spill"] = 0.0
    for team, idx in df.groupby("team").groups.items():
        if not team:
            continue
        sub = df.loc[list(idx)].copy()
        lost = 0.0
        for _, r in sub.iterrows():
            base = float(r.get("proj_minutes") or 0.0)
            adj = float(r.get("adj_minutes") or 0.0)
            if base >= 12.0 and adj < base - 0.5:
                lost += base - adj
        lost = float(np.clip(lost, 0.0, config.INJURY_REDIST_TEAM_CAP))
        if lost < 1.0:
            continue
        # Recipients: healthy enough (factor >= 0.85) with some role
        recipients = []
        for ix in idx:
            r = df.loc[ix]
            try:
                factor = float(r["minutes_factor"]) if r.get("minutes_factor") is not None else 1.0
            except (TypeError, ValueError):
                factor = 1.0
            try:
                base = float(r["proj_minutes"]) if r.get("proj_minutes") is not None else 0.0
            except (TypeError, ValueError):
                base = 0.0
            try:
                adj_now = float(r["adj_minutes"]) if r.get("adj_minutes") is not None else 0.0
            except (TypeError, ValueError):
                adj_now = 0.0
            # Never spill onto Out/Doubtful (keep injury-first zeros)
            if factor < 0.85 or adj_now < 1.0 or base < 8.0:
                continue
            usage = 12.0
            try:
                raw_u = r.get("usage_fga36")
                # pandas stores missing floats as NaN; treat those as default usage
                if raw_u is not None and not (isinstance(raw_u, float) and np.isnan(raw_u)):
                    usage_f = float(raw_u)
                    if np.isfinite(usage_f):
                        usage = usage_f
            except (TypeError, ValueError):
                usage = 12.0
            weight = max(usage, 6.0) * max(base, 1.0)
            recipients.append((ix, weight))
        if not recipients:
            continue
        wsum = sum(w for _, w in recipients) or 1.0
        for ix, w in recipients:
            bump = lost * (w / wsum)
            bump = float(np.clip(bump, 0.0, config.INJURY_REDIST_CAP_PER_PLAYER))
            try:
                cur = float(df.at[ix, "adj_minutes"])
            except (TypeError, ValueError):
                cur = 0.0
            new_adj = float(np.clip(cur + bump, 0.0, 42.0))
            df.at[ix, "minutes_spill"] = round(bump, 2)
            df.at[ix, "adj_minutes"] = round(new_adj, 2)
    # Recompute usage shares after spill
    df = _attach_usage_shares(df)
    return df


def team_impact_from_players(
    minute_df: pd.DataFrame,
    impact_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Minutes-weighted sum of EPM/RAPM → team strength delta + injury notes."""
    impact_by_name: dict[str, float] = {}
    for row in impact_rows or []:
        name = normalize_player_name(row.get("player") or "")
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
            key = normalize_player_name(r["player"])
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
    # Recent game dates + was-away flags for density / road-B2B patterns
    recent_dates: dict[str, list[date]] = {}
    recent_away: dict[str, list[bool]] = {}
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
                recent_dates.setdefault(abbr, []).append(r["_date"])
                # MATCHUP like "BOS @ MIA" or "BOS vs. MIA"
                venue = abbr
                was_away = False
                if match_c and isinstance(r.get(match_c), str):
                    m = str(r[match_c])
                    if "@" in m:
                        parts = m.split("@")
                        venue = normalize_abbrev(parts[-1].strip().split()[-1])
                        was_away = True
                    else:
                        venue = abbr
                        was_away = False
                last_venue[abbr] = venue
                recent_away.setdefault(abbr, []).append(was_away)

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
            dates = recent_dates.get(team) or []
            aways = recent_away.get(team) or []
            # Density: games in last 4 / 6 calendar days (excluding today)
            in4 = sum(1 for d in dates if 0 < (target - d).days <= 4)
            in6 = sum(1 for d in dates if 0 < (target - d).days <= 6)
            # Last game was on the road?
            last_was_away = bool(aways[-1]) if aways else False
            # Point residual: B2B / travel / TZ (small)
            adj = 0.0
            pace_adj = 0.0  # possessions delta
            flags = []
            team_pace = pace_by_team.get(team, config.DEFAULT_PACE)
            if b2b:
                adj += float(config.REST_B2B_POINTS)
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
            # 3-in-4 / 4-in-6 density (includes tonight as +1 conceptually)
            if in4 >= 2:
                # Tonight would be 3rd in 4
                adj += float(config.REST_3IN4_POINTS)
                pace_adj -= 0.4
                flags.append("3-in-4")
            if in6 >= 3:
                adj += float(config.REST_4IN6_POINTS)
                pace_adj -= 0.3
                flags.append("4-in-6")
            # Home after road B2B
            if is_home and b2b and last_was_away:
                adj -= 0.45
                flags.append("home-after-road-B2B")
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
            # Denver altitude: boost home, slight pace up for both;
            # travelers (away) get an extra hit (altitude traveler)
            if home in config.ALTITUDE_TEAMS:
                if is_home:
                    adj += config.ALTITUDE_HOME_POINTS
                    flags.append("altitude-home")
                else:
                    adj -= 0.35 * config.ALTITUDE_HOME_POINTS
                    # Extra traveler fatigue into altitude
                    if miles >= 600 or last_was_away:
                        adj -= 0.35
                        flags.append("altitude-traveler")
                    else:
                        flags.append("altitude-away")
                pace_adj += 0.5 * config.ALTITUDE_PACE_BONUS
            ctx[f"{g.get('game_id')}:{team}"] = {
                "team": team,
                "game_id": g.get("game_id"),
                "rest_days": rest,
                "b2b": b2b,
                "games_in_4": in4,
                "games_in_6": in6,
                "travel_miles": miles,
                "tz_shift": tz,
                "context_points": adj,
                "pace_adj": float(np.clip(pace_adj, -3.5, 2.5)),
                "flags": flags,
            }
    return ctx


def playtype_matchup_residual(
    play_types: pd.DataFrame,
    home: str,
    away: str,
    *,
    ratings: pd.DataFrame | None = None,
) -> tuple[float, list[str]]:
    """
    Play-type PPP residual when Synergy rows exist; otherwise degrade to
    four-factor matchup from ratings (eFG/TOV/ORB/FT vs opp def).
    Prefers defensive grouping rows when PLAY_TYPE contains 'defense'.
    """
    home = normalize_abbrev(home)
    away = normalize_abbrev(away)
    if play_types is None or (isinstance(play_types, pd.DataFrame) and play_types.empty):
        return four_factor_matchup_residual(ratings, home, away)

    df = play_types.copy()
    team_c = next((c for c in ("TEAM_ABBREVIATION", "TeamAbbreviation", "team") if c in df.columns), None)
    pt_c = next((c for c in ("PLAY_TYPE", "play_type") if c in df.columns), None)
    ppp_c = next((c for c in ("PPP", "POINTS", "PTS") if c in df.columns), None)
    side_c = next(
        (c for c in ("TYPE_GROUPING", "type_grouping", "SIDE", "offenseDefense") if c in df.columns),
        None,
    )
    if not team_c or not pt_c or not ppp_c:
        return four_factor_matchup_residual(ratings, home, away)

    df["_team"] = df[team_c].map(lambda x: normalize_abbrev(str(x)))
    df["_ppp"] = pd.to_numeric(df[ppp_c], errors="coerce")
    if side_c:
        df["_side"] = df[side_c].astype(str).str.lower()
    else:
        df["_side"] = "offensive"

    off = df[~df["_side"].str.contains("def")]
    deff = df[df["_side"].str.contains("def")]
    if off.empty:
        off = df
    league = off.groupby(pt_c)["_ppp"].mean()

    notes = []
    edge = 0.0
    for team, opp in ((home, away), (away, home)):
        tg = off[off["_team"] == team]
        if tg.empty:
            continue
        for _, r in tg.iterrows():
            pt = r[pt_c]
            ppp = r["_ppp"]
            if pd.isna(ppp) or pt not in league.index:
                continue
            delta = float(ppp - league.loc[pt])
            # Opponent defensive PPP allowed for same play type when present
            if not deff.empty:
                opp_d = deff[(deff["_team"] == opp) & (deff[pt_c] == pt)]
                if not opp_d.empty:
                    try:
                        opp_ppp = float(opp_d.iloc[0]["_ppp"])
                        # Higher defensive PPP allowed = softer
                        delta += 0.5 * (opp_ppp - float(league.loc[pt]))
                    except (TypeError, ValueError):
                        pass
            contrib = 0.15 * delta * (1 if team == home else -1)
            edge += contrib
            if abs(delta) >= 0.08:
                notes.append(f"{team} {pt} PPP {ppp:.2f} (lg {league.loc[pt]:.2f})")

    if abs(edge) < 1e-9 and notes == []:
        return four_factor_matchup_residual(ratings, home, away)

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
    # ESPN prior-season seed when nba_api prior logs are empty
    if prior_roll is None or prior_roll.empty:
        prior_roll = _ratings_from_espn_season(raw.get("prior_espn"))
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

    # Opponent-adjusted (SRS) + multi-horizon form into the rating stack
    srs = compute_srs_ratings(raw.get("team_logs"))
    ratings = apply_srs_to_ratings(ratings, srs)
    team_form = build_team_form(raw.get("team_logs"))
    ratings = apply_form_to_ratings(ratings, team_form)

    players = align_players_to_rosters(raw.get("players"), raw.get("rosters") or [])
    inj_idx = index_injuries(raw.get("injuries") or [])
    minutes = project_player_minutes(
        players,
        inj_idx,
        recent_minutes=raw.get("recent_minutes") or {},
    )
    minutes = redistribute_minutes_for_injuries(minutes)
    impact = team_impact_from_players(minutes, raw.get("impact") or [])
    context = rest_and_travel_context(
        raw.get("games") or [],
        raw.get("team_logs"),
        target,
        ratings=ratings,
    )
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
        "player_rates": raw.get("player_rates") or {},
        "home_road": home_road,
        "quarter_profiles": quarter_profiles,
        "srs": srs,
        "play_types": raw.get("play_types"),
        "team_adv": raw.get("team_adv"),
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

