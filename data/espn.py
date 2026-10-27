"""ESPN public JSON APIs — reliable free fallback when stats.nba.com is blocked."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import requests

import config
from data.static_arenas import normalize_abbrev

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json"})

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"


def _get(url: str, **kwargs) -> Any:
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)
    resp = SESSION.get(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


def list_teams() -> list[dict[str, Any]]:
    data = _get(f"{SITE}/teams")
    out = []
    for wrap in data["sports"][0]["leagues"][0]["teams"]:
        t = wrap["team"]
        out.append(
            {
                "id": str(t["id"]),
                "abbr": normalize_abbrev(t.get("abbreviation")),
                "name": t.get("displayName"),
                "short": t.get("shortDisplayName"),
            }
        )
    return out


def scoreboard(d: date) -> list[dict[str, Any]]:
    stamp = d.strftime("%Y%m%d")
    data = _get(f"{SITE}/scoreboard", params={"dates": stamp})
    games = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        home = away = None
        for c in comp.get("competitors") or []:
            team = c.get("team") or {}
            entry = {
                "abbr": normalize_abbrev(team.get("abbreviation")),
                "name": team.get("displayName") or team.get("name"),
                "id": team.get("id"),
                "score": (c.get("score") or {}).get("value")
                if isinstance(c.get("score"), dict)
                else c.get("score"),
            }
            if c.get("homeAway") == "home":
                home = entry
            else:
                away = entry
        if not home or not away:
            continue
        games.append(
            {
                "game_id": str(ev.get("id")),
                "game_date": d.isoformat(),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "home_abbr": home["abbr"],
                "away_abbr": away["abbr"],
                "home_name": home["name"],
                "away_name": away["name"],
                "status": ((comp.get("status") or {}).get("type") or {}).get("description")
                or ev.get("status", {}).get("type", {}).get("description")
                or "",
                "source": "espn",
            }
        )
    return games


def _score_val(competitor: dict) -> float | None:
    sc = competitor.get("score")
    if sc is None:
        return None
    if isinstance(sc, dict):
        try:
            return float(sc.get("value"))
        except (TypeError, ValueError):
            return None
    try:
        return float(sc)
    except (TypeError, ValueError):
        return None


def team_game_logs_from_schedules() -> list[dict[str, Any]]:
    """Build nba_api-like team game log rows from ESPN team schedules."""
    teams = list_teams()
    rows: list[dict[str, Any]] = []
    for t in teams:
        try:
            data = _get(f"{SITE}/teams/{t['id']}/schedule")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN schedule %s failed: %s", t["abbr"], exc)
            continue
        for ev in data.get("events") or []:
            comp = (ev.get("competitions") or [None])[0]
            if not comp:
                continue
            status = ((comp.get("status") or {}).get("type") or {}).get("name") or ""
            if status != "STATUS_FINAL":
                continue
            comps = comp.get("competitors") or []
            mine = next(
                (c for c in comps if normalize_abbrev((c.get("team") or {}).get("abbreviation")) == t["abbr"]),
                None,
            )
            opp = next(
                (c for c in comps if c is not mine),
                None,
            )
            if not mine or not opp:
                continue
            my_pts = _score_val(mine)
            opp_pts = _score_val(opp)
            if my_pts is None or opp_pts is None:
                continue
            is_home = mine.get("homeAway") == "home"
            opp_abbr = normalize_abbrev((opp.get("team") or {}).get("abbreviation"))
            matchup = f"{t['abbr']} vs. {opp_abbr}" if is_home else f"{t['abbr']} @ {opp_abbr}"
            gdate = (ev.get("date") or "")[:10]
            rows.append(
                {
                    "TEAM_ABBREVIATION": t["abbr"],
                    "TEAM_NAME": t["name"],
                    "GAME_DATE": gdate,
                    "MATCHUP": matchup,
                    "WL": "W" if my_pts > opp_pts else "L",
                    "PTS": my_pts,
                    "PLUS_MINUS": my_pts - opp_pts,
                    # Box-score details unavailable here — process.py falls back for FF
                    "FGA": None,
                    "FTA": None,
                    "TOV": None,
                    "OREB": None,
                    "FGM": None,
                    "FG3M": None,
                    "FG3A": None,
                    "SOURCE": "espn_schedule",
                }
            )
    return rows


def team_season_stats() -> list[dict[str, Any]]:
    """Per-team season averages (for four-factor / rating enrichment)."""
    teams = list_teams()
    rows = []
    for t in teams:
        try:
            data = _get(f"{SITE}/teams/{t['id']}/statistics")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN team stats %s failed: %s", t["abbr"], exc)
            continue
        cats = (((data.get("results") or {}).get("stats") or {}).get("categories")) or []
        flat: dict[str, float] = {}
        for cat in cats:
            for s in cat.get("stats") or []:
                name = s.get("name")
                val = s.get("value")
                if name is None or val is None:
                    continue
                try:
                    flat[name] = float(val)
                except (TypeError, ValueError):
                    continue
        # Approximate offensive/defensive ratings from scoring averages when advanced missing
        pts = flat.get("avgPoints")
        opp = flat.get("avgPointsAllowed") or flat.get("avgOpponentPoints")
        pace_proxy = flat.get("avgFieldGoalsAttempted")
        rows.append(
            {
                "TEAM_ABBREVIATION": t["abbr"],
                "TEAM_NAME": t["name"],
                "GP": flat.get("gamesPlayed"),
                "PTS": pts,
                "OPP_PTS": opp,
                "FG_PCT": flat.get("fieldGoalPct") or flat.get("avgFieldGoalPct"),
                "FG3_PCT": flat.get("threePointPct"),
                "FT_PCT": flat.get("freeThrowPct"),
                "AVG_FGA": flat.get("avgFieldGoalsAttempted"),
                "AVG_FGM": flat.get("avgFieldGoalsMade"),
                "AVG_FG3M": flat.get("avgThreePointFieldGoalsMade"),
                "AVG_FG3A": flat.get("avgThreePointFieldGoalsAttempted"),
                "AVG_FTA": flat.get("avgFreeThrowsAttempted"),
                "AVG_FTM": flat.get("avgFreeThrowsMade"),
                "AVG_OREB": flat.get("avgOffensiveRebounds"),
                "AVG_TOV": flat.get("avgTurnovers"),
                "PACE_PROXY": pace_proxy,
                "RAW": flat,
                "SOURCE": "espn_team_stats",
            }
        )
    return rows


def team_rosters() -> list[dict[str, Any]]:
    """Current ESPN team rosters (active + two-ways as listed)."""
    rows: list[dict[str, Any]] = []
    for t in list_teams():
        try:
            data = _get(f"{SITE}/teams/{t['id']}/roster")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN roster %s failed: %s", t["abbr"], exc)
            continue
        for a in data.get("athletes") or []:
            name = a.get("displayName") or a.get("fullName")
            if not name:
                continue
            status = a.get("status")
            if isinstance(status, dict):
                status_type = status.get("type") or status.get("name") or ""
            else:
                status_type = str(status or "active")
            pos = a.get("position") or {}
            rows.append(
                {
                    "PLAYER_NAME": name,
                    "TEAM_ABBREVIATION": t["abbr"],
                    "TEAM_NAME": t["name"],
                    "PLAYER_ID": str(a.get("id") or ""),
                    "POSITION": pos.get("abbreviation") or pos.get("name") or "",
                    "STATUS": str(status_type).lower() or "active",
                    "SOURCE": "espn_roster",
                }
            )
    return rows


def injuries_api() -> list[dict[str, Any]]:
    """Structured injury report from ESPN site API (preferred over HTML scrape)."""
    data = _get(f"{SITE}/injuries")
    teams = {t["name"]: t["abbr"] for t in list_teams()}
    # Also index short names
    for t in list_teams():
        if t.get("short"):
            teams[t["short"]] = t["abbr"]
    out: list[dict[str, Any]] = []
    for group in data.get("injuries") or []:
        team_name = group.get("displayName") or ""
        team_abbr = teams.get(team_name) or normalize_abbrev(group.get("id"))
        # id is often ESPN team id numeric — map via list_teams ids
        if not team_abbr or team_abbr.isdigit():
            tid = str(group.get("id") or "")
            hit = next((t for t in list_teams() if t["id"] == tid), None)
            team_abbr = hit["abbr"] if hit else ""
        for inj in group.get("injuries") or []:
            athlete = inj.get("athlete") or {}
            player = athlete.get("displayName") or (
                f"{athlete.get('firstName', '')} {athlete.get('lastName', '')}".strip()
            )
            if not player:
                continue
            status = str(inj.get("status") or "").strip()
            comment = str(inj.get("shortComment") or inj.get("longComment") or "")
            out.append(
                {
                    "team": team_name,
                    "team_abbr": team_abbr,
                    "player": player,
                    "status": status,
                    "comment": comment,
                    "date": inj.get("date"),
                    "source": "espn_api",
                }
            )
    return out


def player_stats(limit_pages: int = 12) -> list[dict[str, Any]]:
    """League player per-game stats including minutes (paginate for fuller coverage)."""
    page = 1
    rows: list[dict[str, Any]] = []
    labels_by_cat: dict[str, list[str]] = {}
    while page <= limit_pages:
        data = _get(
            f"{WEB}/statistics/byathlete",
            params={
                "region": "us",
                "lang": "en",
                "contentorigin": "espn",
                "limit": 50,
                "page": page,
                "sort": "general.avgMinutes:desc",
            },
        )
        if page == 1:
            for cat in data.get("categories") or []:
                labels_by_cat[cat["name"]] = cat.get("names") or []
        athletes = data.get("athletes") or []
        if not athletes:
            break
        for a in athletes:
            athlete = a.get("athlete") or {}
            name = athlete.get("displayName")
            team_abbr = ""
            team = athlete.get("team") or {}
            if team:
                team_abbr = normalize_abbrev(team.get("abbreviation"))
            values: dict[str, float] = {}
            for cat in a.get("categories") or []:
                names = labels_by_cat.get(cat.get("name"), [])
                for i, val in enumerate(cat.get("values") or []):
                    if i < len(names):
                        try:
                            values[names[i]] = float(val)
                        except (TypeError, ValueError):
                            continue
            def _first(*keys: str) -> float | None:
                for k in keys:
                    if values.get(k) is not None:
                        return values.get(k)
                return None

            rows.append(
                {
                    "PLAYER_NAME": name,
                    "TEAM_ABBREVIATION": team_abbr,
                    "MIN": _first("avgMinutes") or 0.0,
                    "PTS": _first("avgPoints"),
                    "REB": _first("avgRebounds", "avgTotalRebounds", "avgDefensiveRebounds"),
                    "AST": _first("avgAssists"),
                    "STL": _first("avgSteals"),
                    "BLK": _first("avgBlocks"),
                    "PLUS_MINUS": _first("avgPlusMinus", "plusMinus"),
                    "NBA_FANTASY_PTS": _first("avgPoints"),
                    "GP": _first("gamesPlayed"),
                    "SOURCE": "espn_athletes",
                }
            )
        pages = (data.get("pagination") or {}).get("pages") or page
        if page >= int(pages):
            break
        page += 1
    # Attach team via roster map if missing
    if any(not r["TEAM_ABBREVIATION"] for r in rows):
        roster_map = _roster_name_to_team()
        for r in rows:
            if not r["TEAM_ABBREVIATION"]:
                r["TEAM_ABBREVIATION"] = roster_map.get(
                    (r["PLAYER_NAME"] or "").lower(), ""
                )
    return rows


def _roster_name_to_team() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for r in team_rosters():
        name = (r.get("PLAYER_NAME") or "").lower()
        if name:
            mapping[name] = r["TEAM_ABBREVIATION"]
    return mapping


def _period_points(linescores: list[dict[str, Any]] | None) -> dict[int, float]:
    """Map period number -> points from ESPN linescores."""
    out: dict[int, float] = {}
    if not linescores:
        return out
    for i, ls in enumerate(linescores, start=1):
        period = int(ls.get("period") or i)
        if period > 4:
            continue  # ignore OT for tendency baselines
        val = ls.get("value")
        if val is None:
            try:
                val = float(ls.get("displayValue"))
            except (TypeError, ValueError):
                continue
        try:
            out[period] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def recent_quarter_game_rows(days_back: int = 21) -> list[dict[str, Any]]:
    """
    Pull completed scoreboards for the last N days and emit per-team quarter
    scoring rows (Q1-Q4). Free ESPN scoreboard already includes linescores.
    """
    from datetime import date, timedelta

    today = date.today()
    rows: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    for delta in range(1, max(1, days_back) + 1):
        d = today - timedelta(days=delta)
        try:
            data = _get(f"{SITE}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN scoreboard %s failed: %s", d, exc)
            continue
        for ev in data.get("events") or []:
            gid = str(ev.get("id") or "")
            if not gid or gid in seen_games:
                continue
            comp = (ev.get("competitions") or [None])[0]
            if not comp:
                continue
            status = ((comp.get("status") or {}).get("type") or {}).get("name") or ""
            if status != "STATUS_FINAL":
                continue
            seen_games.add(gid)
            comps = comp.get("competitors") or []
            parsed: list[tuple[str, bool, dict[int, float], float | None]] = []
            for c in comps:
                abbr = normalize_abbrev((c.get("team") or {}).get("abbreviation"))
                if not abbr:
                    continue
                qpts = _period_points(c.get("linescores"))
                if len(qpts) < 4:
                    continue
                sc = _score_val(c)
                parsed.append((abbr, c.get("homeAway") == "home", qpts, sc))
            if len(parsed) != 2:
                continue
            for abbr, is_home, qpts, sc in parsed:
                opp = next(p[0] for p in parsed if p[0] != abbr)
                opp_q = next(p[2] for p in parsed if p[0] != abbr)
                rows.append(
                    {
                        "game_id": gid,
                        "game_date": d.isoformat(),
                        "TEAM_ABBREVIATION": abbr,
                        "OPP_ABBREVIATION": opp,
                        "IS_HOME": is_home,
                        "PTS": sc,
                        "Q1": qpts.get(1),
                        "Q2": qpts.get(2),
                        "Q3": qpts.get(3),
                        "Q4": qpts.get(4),
                        "OPP_Q1": opp_q.get(1),
                        "OPP_Q2": opp_q.get(2),
                        "OPP_Q3": opp_q.get(3),
                        "OPP_Q4": opp_q.get(4),
                        "SOURCE": "espn_scoreboard_quarters",
                    }
                )
    return rows


def player_gamelog_recent(
    athlete_id: str,
    *,
    season_end_year: int | None = None,
    last_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Recent per-game counting stats for one ESPN athlete id.
    season_end_year: ending calendar year of the NBA season (2026-27 -> 2027).
    """
    if not athlete_id:
        return []
    year = season_end_year
    if year is None:
        year = config.current_season_year() + 1
    try:
        data = _get(
            f"{WEB}/athletes/{athlete_id}/gamelog",
            params={"season": int(year)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN gamelog %s failed: %s", athlete_id, exc)
        return []

    names = list(data.get("names") or [])
    try:
        pts_i = names.index("points")
    except ValueError:
        pts_i = len(names) - 1 if names else -1
    try:
        min_i = names.index("minutes")
    except ValueError:
        min_i = 0 if names else -1

    events_meta = data.get("events") or {}
    dated: list[tuple[str, dict[str, Any]]] = []
    for st in data.get("seasonTypes") or []:
        for cat in st.get("categories") or []:
            for ev in cat.get("events") or []:
                eid = str(ev.get("eventId") or "")
                stats = ev.get("stats") or []
                meta = events_meta.get(eid) or {}
                gdate = str(meta.get("gameDate") or "")[:10]
                if not gdate or pts_i < 0 or pts_i >= len(stats):
                    continue
                try:
                    pts = float(stats[pts_i])
                except (TypeError, ValueError):
                    continue
                mins = None
                if 0 <= min_i < len(stats):
                    try:
                        mins = float(stats[min_i])
                    except (TypeError, ValueError):
                        mins = None
                dated.append(
                    (
                        gdate,
                        {
                            "game_date": gdate,
                            "pts": pts,
                            "minutes": mins,
                            "event_id": eid,
                            "result": meta.get("gameResult"),
                        },
                    )
                )
    dated.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in dated[: max(1, last_n)]]


def enrich_ratings_from_season_stats(season_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ESPN season averages into process.build_team_rolling-compatible summary rows."""
    out = []
    for r in season_rows:
        pts = r.get("PTS")
        opp = r.get("OPP_PTS")
        fga = r.get("AVG_FGA") or 88.0
        fta = r.get("AVG_FTA") or 22.0
        tov = r.get("AVG_TOV") or 13.0
        oreb = r.get("AVG_OREB") or 10.0
        fgm = r.get("AVG_FGM") or 42.0
        fg3m = r.get("AVG_FG3M") or 12.0
        poss = float(np.clip(fga - oreb + tov + 0.44 * fta, 80.0, 120.0))
        off_rtg = 100.0 * float(pts) / poss if pts else config.DEFAULT_OFF_RTG
        # If opponent points missing, assume league-average total split
        if pts is not None and opp is None:
            opp = config.LEAGUE_AVG_TOTAL - float(pts)
        def_rtg = 100.0 * float(opp) / poss if opp is not None else config.DEFAULT_DEF_RTG
        pace = poss
        efg = (fgm + 0.5 * fg3m) / fga if fga else 0.53
        tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + fta + tov) else 0.13
        orb_pct = oreb / (oreb + 25.0)
        ft_rate = fta / fga if fga else 0.25
        out.append(
            {
                "team": r["TEAM_ABBREVIATION"],
                "games": int(r.get("GP") or 0),
                "last_game": None,
                "off_rtg": off_rtg,
                "def_rtg": def_rtg,
                "net_rtg": off_rtg - def_rtg,
                "pace": pace,
                "efg": efg,
                "tov_pct": tov_pct,
                "orb_pct": orb_pct,
                "ft_rate": ft_rate,
                "recent_dates": [],
                "source": "espn_season",
            }
        )
    return out
