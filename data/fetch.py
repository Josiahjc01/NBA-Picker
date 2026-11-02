"""Pull free NBA data sources with aggressive caching and graceful fallbacks."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from data.cache_util import (
    content_fingerprint,
    invalidate_cache,
    load_or_fetch_json,
    read_cache_meta,
    read_json_cache,
    write_json_cache,
)
from data.static_arenas import normalize_abbrev

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})


def _get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)
    resp = SESSION.get(url, **kwargs)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Schedule / games (nba_api)
# ---------------------------------------------------------------------------

_NBA_API_AVAILABLE: bool | None = None


def nba_api_reachable() -> bool:
    """One-shot probe — many cloud IPs are blocked by stats.nba.com."""
    global _NBA_API_AVAILABLE
    if _NBA_API_AVAILABLE is not None:
        return _NBA_API_AVAILABLE
    if os.getenv("SKIP_NBA_API", "").lower() in {"1", "true", "yes"}:
        _NBA_API_AVAILABLE = False
        return False
    try:
        resp = SESSION.get(
            "https://stats.nba.com/stats/scoreboardv2",
            params={"GameDate": "01/01/2020", "LeagueID": "00", "DayOffset": "0"},
            headers={
                "Referer": "https://www.nba.com/",
                "Origin": "https://www.nba.com",
                "Accept": "application/json",
            },
            timeout=min(5, config.NBA_API_TIMEOUT),
        )
        _NBA_API_AVAILABLE = resp.status_code == 200
    except Exception:  # noqa: BLE001
        _NBA_API_AVAILABLE = False
    if not _NBA_API_AVAILABLE:
        logger.info("stats.nba.com unreachable — using ESPN / scrape fallbacks")
    return _NBA_API_AVAILABLE


def _configure_nba_api_timeout() -> None:
    """Shorten nba_api HTTP timeouts — stats.nba.com is frequently unreachable."""
    try:
        from nba_api.stats.library import http as nba_http

        nba_http.TIMEOUT = config.NBA_API_TIMEOUT
    except Exception:  # noqa: BLE001
        pass


def _scoreboard_nba_api(d: date) -> list[dict[str, Any]]:
    """Official scoreboard via nba_api (may fail from cloud IPs)."""
    from nba_api.stats.endpoints import scoreboardv2

    _configure_nba_api_timeout()
    date_str = d.strftime("%m/%d/%Y")
    time.sleep(0.4)
    sb = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0, timeout=config.NBA_API_TIMEOUT)
    frames = sb.get_data_frames()
    if not frames:
        return []
    games_df = frames[0]
    line_df = frames[1] if len(frames) > 1 else pd.DataFrame()

    games: list[dict[str, Any]] = []
    for _, row in games_df.iterrows():
        gid = str(row.get("GAME_ID", ""))
        home_id = row.get("HOME_TEAM_ID")
        away_id = row.get("VISITOR_TEAM_ID")
        home_abbr = away_abbr = ""
        home_name = away_name = ""
        if not line_df.empty:
            home_rows = line_df[line_df["TEAM_ID"] == home_id]
            away_rows = line_df[line_df["TEAM_ID"] == away_id]
            if not home_rows.empty:
                home_abbr = normalize_abbrev(str(home_rows.iloc[0].get("TEAM_ABBREVIATION", "")))
                home_name = str(home_rows.iloc[0].get("TEAM_NAME", home_abbr))
            if not away_rows.empty:
                away_abbr = normalize_abbrev(str(away_rows.iloc[0].get("TEAM_ABBREVIATION", "")))
                away_name = str(away_rows.iloc[0].get("TEAM_NAME", away_abbr))
        games.append(
            {
                "game_id": gid,
                "game_date": d.isoformat(),
                "home_team_id": int(home_id) if pd.notna(home_id) else None,
                "away_team_id": int(away_id) if pd.notna(away_id) else None,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_name": home_name,
                "away_name": away_name,
                "status": str(row.get("GAME_STATUS_TEXT", "")),
                "source": "nba_api",
            }
        )
    return games


def _scoreboard_for_date(d: date) -> list[dict[str, Any]]:
    """Prefer ESPN (reliable); fall back to nba_api."""
    try:
        from data import espn

        games = espn.scoreboard(d)
        if games:
            logger.info("Scoreboard from ESPN (%d games)", len(games))
            return games
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN scoreboard failed: %s", exc)
    if nba_api_reachable():
        try:
            games = _scoreboard_nba_api(d)
            if games:
                return games
        except Exception as exc:  # noqa: BLE001
            logger.warning("nba_api scoreboard failed: %s", exc)
    return []


def fetch_games_for_date(d: date, force: bool = False) -> list[dict[str, Any]]:
    key = f"games_{d.isoformat()}"

    def _fetch():
        return _scoreboard_for_date(d)

    try:
        return load_or_fetch_json(key, _fetch, force=force) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scoreboard fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Team game logs / league dashboards
# ---------------------------------------------------------------------------

def _team_game_logs_nba_api(season: str) -> list[dict[str, Any]]:
    from nba_api.stats.endpoints import leaguegamelog

    _configure_nba_api_timeout()
    time.sleep(0.4)
    lg = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=config.NBA_API_TIMEOUT,
    )
    df = lg.get_data_frames()[0]
    return df.to_dict(orient="records")


def _team_game_logs(season: str) -> list[dict[str, Any]]:
    if nba_api_reachable():
        try:
            rows = _team_game_logs_nba_api(season)
            if rows:
                logger.info("Team logs from nba_api (%d)", len(rows))
                return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("nba_api team logs failed: %s", exc)
    from data import espn

    rows = espn.team_game_logs_from_schedules()
    # Fill FF/pace inputs from ESPN boxscore summaries (free fallback)
    try:
        rows = espn.enrich_team_logs_with_boxscores(rows, max_games=90)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN boxscore enrichment failed: %s", exc)
    logger.info("Team logs from ESPN schedules (+box) (%d)", len(rows))
    return rows


def fetch_team_game_logs(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    # Bump cache key when box enrichment lands so stale score-only logs refresh
    key = f"team_logs_box_{season}"

    def _fetch():
        return _team_game_logs(season)

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Team game logs failed for %s: %s", season, exc)
        return pd.DataFrame()


def fetch_prior_season_logs(force: bool = False) -> pd.DataFrame:
    """Prior season logs for early-season blend. Soft-fail if nba_api blocked."""
    prior = config.season_string(config.current_season_year() - 1)
    key = f"team_logs_{prior}"

    def _fetch():
        if not nba_api_reachable():
            return []
        try:
            return _team_game_logs_nba_api(prior)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prior-season logs unavailable (%s); skip blend prior", exc)
            return []

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prior season fetch failed: %s", exc)
        return pd.DataFrame()


def fetch_prior_espn_seed(force: bool = False) -> pd.DataFrame:
    """
    Prior-season team rating seed from ESPN (works when nba_api is blocked).
    season_end_year = current_season_year (e.g. 2026 for 2025-26 when now is 2026-27).
    """
    end_year = config.current_season_year()  # ending year of prior season
    key = f"prior_espn_seed_{end_year}"

    def _fetch():
        from data import espn

        rows = espn.team_season_stats_for_year(end_year)
        if not rows:
            raise RuntimeError("Empty ESPN prior season stats")
        logger.info("Prior ESPN seed for %s (%d teams)", end_year, len(rows))
        return rows

    try:
        rows = load_or_fetch_json(key, _fetch, force=force, max_age_hours=48.0)
        return pd.DataFrame(rows or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prior ESPN seed failed: %s", exc)
        return pd.DataFrame()


def _league_dash_team(season: str, measure: str = "Base") -> list[dict[str, Any]]:
    from nba_api.stats.endpoints import leaguedashteamstats

    _configure_nba_api_timeout()
    time.sleep(0.4)
    ep = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
        timeout=config.NBA_API_TIMEOUT,
    )
    return ep.get_data_frames()[0].to_dict(orient="records")


def fetch_team_advanced(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    key = f"team_adv_{season}"

    def _fetch():
        if nba_api_reachable():
            try:
                return _league_dash_team(season, "Advanced")
            except Exception as exc:  # noqa: BLE001
                logger.warning("nba_api advanced failed (%s); using ESPN season stats", exc)
        from data import espn

        return espn.team_season_stats()

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Advanced team stats failed: %s", exc)
        return pd.DataFrame()


def fetch_team_four_factors(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    key = f"team_ff_{season}"

    def _fetch():
        if nba_api_reachable():
            try:
                return _league_dash_team(season, "Four Factors")
            except Exception as exc:  # noqa: BLE001
                logger.warning("nba_api four factors failed (%s); using ESPN", exc)
        from data import espn

        return espn.team_season_stats()

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Four Factors failed: %s", exc)
        return pd.DataFrame()


def _player_dash(season: str) -> list[dict[str, Any]]:
    if nba_api_reachable():
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats

            _configure_nba_api_timeout()
            time.sleep(0.4)
            ep = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                timeout=config.NBA_API_TIMEOUT,
            )
            rows = ep.get_data_frames()[0].to_dict(orient="records")
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("nba_api player stats failed: %s", exc)
    from data import espn

    return espn.player_stats()


def fetch_player_stats(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    key = f"player_stats_{season}"

    def _fetch():
        return _player_dash(season)

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Player stats failed: %s", exc)
        return pd.DataFrame()


def _synergy_playtypes(season: str) -> list[dict[str, Any]]:
    """Team play-type offensive efficiencies via Synergy endpoint when available."""
    from nba_api.stats.endpoints import synergyplaytypes

    _configure_nba_api_timeout()
    results: list[dict[str, Any]] = []
    # Keep the set small — each call can hang when stats.nba.com is blocked
    play_types = ["Isolation", "Transition", "PRBallHandler", "Spotup", "Postup"]
    for pt in play_types:
        try:
            time.sleep(0.3)
            ep = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                season=season,
                season_type_all_star="Regular Season",
                play_type_nullable=pt,
                player_or_team_abbreviation="T",
                type_grouping_nullable="offensive",
                per_mode_simple="PerGame",
                timeout=config.NBA_API_TIMEOUT,
            )
            df = ep.get_data_frames()[0]
            for _, row in df.iterrows():
                rec = row.to_dict()
                rec["PLAY_TYPE"] = pt
                results.append(rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Synergy %s failed: %s", pt, exc)
            # If the first call fails hard, skip the rest
            if not results:
                break
    return results


def fetch_play_types(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    key = f"playtypes_{season}"

    def _fetch():
        if not nba_api_reachable():
            return []
        try:
            return _synergy_playtypes(season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Play types unavailable: %s", exc)
            return []

    try:
        rows = load_or_fetch_json(key, _fetch, force=force)
        return pd.DataFrame(rows or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Play types unavailable: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Rosters — ESPN current team membership (source of truth for who is on each team)
# ---------------------------------------------------------------------------

def _roster_fingerprint(rows: list[dict[str, Any]]) -> str:
    pairs = sorted(
        {
            f"{normalize_abbrev(str(r.get('TEAM_ABBREVIATION') or ''))}|{(r.get('PLAYER_NAME') or '').strip().lower()}"
            for r in rows or []
            if r.get("PLAYER_NAME")
        }
    )
    return content_fingerprint(pairs)


def fetch_rosters(force: bool = False) -> list[dict[str, Any]]:
    """Current NBA rosters (ESPN). Prefer over stale team tags on season stats."""
    key = "rosters_current"
    prev_fp = (read_cache_meta(key) or {}).get("fingerprint")

    def _fetch():
        from data import espn

        rows = espn.team_rosters()
        if not rows:
            raise RuntimeError("Empty ESPN roster response")
        logger.info(
            "Fetched ESPN rosters (%d players, %d teams)",
            len(rows),
            len({r["TEAM_ABBREVIATION"] for r in rows}),
        )
        return rows

    def _validate(rows: Any) -> bool:
        return isinstance(rows, list) and len(rows) >= 200

    try:
        rows = load_or_fetch_json(
            key,
            _fetch,
            force=force,
            max_age_hours=config.ROSTER_CACHE_MAX_AGE_HOURS,
            fingerprint_fn=_roster_fingerprint,
            validate=_validate,
        ) or []
        new_fp = _roster_fingerprint(rows) if rows else None
        if prev_fp and new_fp and prev_fp != new_fp:
            # Roster membership changed — player season tags may be stale
            season = config.season_string()
            invalidate_cache(f"player_stats_{season}")
            logger.info("Roster fingerprint changed; invalidated player_stats cache")
        return rows
    except Exception as exc:  # noqa: BLE001
        logger.warning("Roster fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Injuries — ESPN API first; HTML scrape fallback (skip junk official landing pages)
# ---------------------------------------------------------------------------

def _parse_espn_injuries_html() -> list[dict[str, Any]]:
    url = "https://www.espn.com/nba/injuries"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    injuries: list[dict[str, Any]] = []
    tables = soup.select("div.ResponsiveTable")
    for table in tables:
        team_el = table.select_one(".Table__Title, div.Table__Title")
        team_name = team_el.get_text(strip=True) if team_el else ""
        for row in table.select("tbody tr"):
            cells = [c.get_text(" ", strip=True) for c in row.select("td")]
            if len(cells) < 4:
                continue
            player = cells[0]
            status = cells[3] if len(cells) >= 4 else cells[-1]
            comment = cells[4] if len(cells) >= 5 else ""
            if not player or len(player) < 3:
                continue
            # Skip non-player boilerplate rows
            if "report" in player.lower() and "update" in player.lower():
                continue
            injuries.append(
                {
                    "team": team_name,
                    "team_abbr": "",
                    "player": player,
                    "status": status,
                    "comment": comment,
                    "source": "espn_html",
                }
            )
    return injuries


def _looks_like_player_injury(row: dict[str, Any]) -> bool:
    """Filter out official.nba.com landing-page junk that used to poison the cache."""
    player = str(row.get("player") or "").strip()
    status = str(row.get("status") or "").strip()
    if len(player) < 3 or not re.search(r"[A-Za-z]", player):
        return False
    low = player.lower()
    if any(x in low for x in ("reports are updated", "injury report", "continual basis", "click here")):
        return False
    if not re.search(r"\b(out|doubtful|questionable|probable|available|gtd|day-to-day|dtd)\b", status, re.I):
        # Allow if status column is short token like "Out"
        if status.lower() not in {"out", "doubtful", "questionable", "probable", "available"}:
            return False
    return True


def _injury_fingerprint(rows: list[dict[str, Any]]) -> str:
    pairs = sorted(
        {
            f"{(r.get('player') or '').strip().lower()}|{(r.get('status') or '').strip().lower()}|"
            f"{normalize_abbrev(str(r.get('team_abbr') or ''))}"
            for r in rows or []
            if r.get("player")
        }
    )
    return content_fingerprint(pairs)


def fetch_injuries(force: bool = False) -> list[dict[str, Any]]:
    key = "injuries_v2"
    prev_fp = (read_cache_meta(key) or {}).get("fingerprint")

    def _fetch():
        from data import espn

        errors: list[str] = []
        try:
            rows = espn.injuries_api()
            rows = [r for r in rows if _looks_like_player_injury(r)]
            if rows:
                logger.info("Injuries from ESPN API (%d)", len(rows))
                return rows
        except Exception as exc:  # noqa: BLE001
            errors.append(f"espn_api: {exc}")
        try:
            rows = _parse_espn_injuries_html()
            rows = [r for r in rows if _looks_like_player_injury(r)]
            if rows:
                logger.info("Injuries from ESPN HTML (%d)", len(rows))
                return rows
        except Exception as exc:  # noqa: BLE001
            errors.append(f"espn_html: {exc}")
        raise RuntimeError("; ".join(errors) or "No injury source available")

    def _validate(rows: Any) -> bool:
        if not isinstance(rows, list) or not rows:
            return False
        return sum(1 for r in rows if _looks_like_player_injury(r)) >= max(5, len(rows) // 2)

    try:
        rows = load_or_fetch_json(
            key,
            _fetch,
            force=force,
            max_age_hours=config.INJURY_CACHE_MAX_AGE_HOURS,
            fingerprint_fn=_injury_fingerprint,
            validate=_validate,
        ) or []
        new_fp = _injury_fingerprint(rows) if rows else None
        if prev_fp and new_fp and prev_fp != new_fp:
            logger.info("Injury fingerprint changed (%s -> %s)", prev_fp, new_fp)
        return rows
    except Exception as exc:  # noqa: BLE001
        logger.warning("Injury fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# EPM / RAPM scrapes
# ---------------------------------------------------------------------------

def _valid_impact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows with real player names and EPM-like magnitudes."""
    cleaned = []
    for r in rows:
        name = str(r.get("player") or "").strip()
        epm = r.get("epm")
        if not name or len(name) < 3:
            continue
        # Reject rank indices / non-names ("1", "23", "#4")
        if re.fullmatch(r"#?\d+", name):
            continue
        if not re.search(r"[A-Za-z]", name):
            continue
        if epm is None:
            continue
        try:
            epm_f = float(epm)
        except (TypeError, ValueError):
            continue
        # True EPM/RAPM rarely exceeds ±15; reject score-like scrapes
        if abs(epm_f) > 15:
            continue
        cleaned.append({**r, "player": name, "epm": epm_f})
    return cleaned


def _scrape_dunksandthrees_epm() -> list[dict[str, Any]]:
    url = "https://dunksandthrees.com/epm"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict[str, Any]] = []
    scripts = soup.find_all("script")
    for script in scripts:
        text = script.string or script.get_text() or ""
        if "epm" in text.lower() and ("player" in text.lower() or "[" in text):
            for match in re.finditer(r"(\[\{.*?\}\])", text, re.DOTALL):
                try:
                    import json

                    data = json.loads(match.group(1))
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        for item in data:
                            rows.append(
                                {
                                    "player": item.get("player")
                                    or item.get("name")
                                    or item.get("Player"),
                                    "team": normalize_abbrev(
                                        str(item.get("team") or item.get("Team") or "")
                                    ),
                                    "epm": _to_float(
                                        item.get("epm") or item.get("EPM") or item.get("value")
                                    ),
                                    "source": "dunksandthrees",
                                }
                            )
                        cleaned = _valid_impact_rows(rows)
                        if cleaned:
                            return cleaned
                        rows = []
                except Exception:  # noqa: BLE001
                    continue
    tables = soup.find_all("table")
    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.select("th")]
        for tr in table.select("tbody tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if len(cells) < 2:
                continue
            player = cells[0]
            team = ""
            epm_val = None
            for i, h in enumerate(headers):
                if i >= len(cells):
                    break
                if "team" in h:
                    team = cells[i]
                if h == "epm" or h.endswith(" epm"):
                    epm_val = _to_float(cells[i])
            if epm_val is None:
                for c in reversed(cells):
                    v = _to_float(c)
                    if v is not None and abs(v) <= 15:
                        epm_val = v
                        break
            rows.append(
                {
                    "player": player,
                    "team": normalize_abbrev(team),
                    "epm": epm_val if epm_val is not None else 0.0,
                    "source": "dunksandthrees-html",
                }
            )
    cleaned = _valid_impact_rows(rows)
    if not cleaned:
        raise RuntimeError("dunksandthrees EPM table empty / JS-rendered / invalid")
    return cleaned


def _scrape_nbarapm() -> list[dict[str, Any]]:
    url = "https://www.nbarapm.com/"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        for tr in table.select("tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if len(cells) < 3:
                continue
            player = cells[0]
            team = normalize_abbrev(cells[1]) if len(cells) > 1 else ""
            rapm = None
            for c in cells[2:]:
                v = _to_float(c)
                if v is not None and abs(v) <= 15:
                    rapm = v
                    break
            if rapm is None:
                continue
            rows.append(
                {"player": player, "team": team, "epm": rapm, "source": "nbarapm"}
            )
    cleaned = _valid_impact_rows(rows)
    if not cleaned:
        raise RuntimeError("nbarapm scrape empty/invalid")
    return cleaned


def _scrape_darko() -> list[dict[str, Any]]:
    """Best-effort darko.app scrape; structure varies."""
    url = "https://darko.app/"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        for tr in table.select("tbody tr") or table.select("tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if len(cells) < 2:
                continue
            player = cells[0]
            val = None
            for c in cells[1:]:
                v = _to_float(c)
                if v is not None and abs(v) <= 15:
                    val = v
                    break
            if val is None:
                continue
            rows.append(
                {"player": player, "team": "", "epm": val, "source": "darko"}
            )
    cleaned = _valid_impact_rows(rows)
    if not cleaned:
        raise RuntimeError("darko scrape empty/invalid")
    return cleaned


def _bpm_proxy_from_nba() -> list[dict[str, Any]]:
    """Fallback impact: minutes-weighted scoring z-score scaled to EPM-like range."""
    df = fetch_player_stats()
    if df.empty:
        return []
    name_col = "PLAYER_NAME" if "PLAYER_NAME" in df.columns else None
    team_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None
    impact_col = None
    for c in ("PLUS_MINUS", "NET_RATING", "PTS", "NBA_FANTASY_PTS"):
        if c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().sum() >= 10:
            impact_col = c
            break
    if not name_col or not impact_col:
        return []
    series = pd.to_numeric(df[impact_col], errors="coerce")
    mean = float(series.mean(skipna=True))
    std = float(series.std(skipna=True)) or 1.0
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
        return []
    rows = []
    for _, r in df.iterrows():
        raw = pd.to_numeric(r.get(impact_col), errors="coerce")
        if pd.isna(raw):
            continue
        epm = float(np.clip(3.0 * (float(raw) - mean) / std, -8.0, 8.0))
        if not np.isfinite(epm):
            continue
        rows.append(
            {
                "player": str(r.get(name_col)),
                "team": normalize_abbrev(str(r.get(team_col, ""))),
                "epm": round(epm, 3),
                "source": f"proxy:{impact_col}",
            }
        )
    return rows


def fetch_player_impact(force: bool = False) -> list[dict[str, Any]]:
    """
    Multi-source player prior inventory: blend dunksandthrees / nbarapm / darko
    / nba_api proxy by normalized name (average when multiple sources hit).
    """
    key = "player_impact_blend_v2"

    def _fetch():
        buckets: dict[str, list[float]] = {}
        meta: dict[str, dict[str, Any]] = {}
        for name, fn in (
            ("dunksandthrees", _scrape_dunksandthrees_epm),
            ("nbarapm", _scrape_nbarapm),
            ("darko", _scrape_darko),
        ):
            try:
                rows = fn() or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Impact scrape %s failed: %s", name, exc)
                rows = []
            if rows:
                logger.info("Player impact source %s (%d rows)", name, len(rows))
            for row in rows:
                pname = re.sub(r"\s+", " ", str(row.get("player") or "").strip()).lower()
                if not pname:
                    continue
                try:
                    epm = float(row.get("epm"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(epm):
                    continue
                buckets.setdefault(pname, []).append(epm)
                meta[pname] = {
                    "player": row.get("player") or pname.title(),
                    "sources": list(
                        dict.fromkeys((meta.get(pname) or {}).get("sources", []) + [name])
                    ),
                }
        proxy = _bpm_proxy_from_nba() or []
        if proxy:
            logger.info("Merging nba_api z-score proxy (%d)", len(proxy))
        for row in proxy:
            pname = re.sub(r"\s+", " ", str(row.get("player") or "").strip()).lower()
            if not pname:
                continue
            try:
                epm = float(row.get("epm"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(epm):
                continue
            # Proxy gets half weight via duplicate-average trick: append once
            buckets.setdefault(pname, []).append(epm)
            prev = meta.get(pname) or {"player": row.get("player") or pname.title(), "sources": []}
            prev["sources"] = list(dict.fromkeys(prev.get("sources", []) + ["nba_proxy"]))
            meta[pname] = prev
        out = []
        for pname, vals in buckets.items():
            m = meta.get(pname) or {"player": pname.title(), "sources": []}
            out.append(
                {
                    "player": m["player"],
                    "epm": round(float(np.mean(vals)), 3),
                    "epm_n_sources": len(vals),
                    "sources": ",".join(m.get("sources") or []),
                    "source": "blend",
                }
            )
        logger.info("Blended player impact inventory: %d players", len(out))
        return out

    try:
        return load_or_fetch_json(key, _fetch, force=force) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Player impact unavailable: %s", exc)
        return []


def fetch_quarter_logs(force: bool = False) -> pd.DataFrame:
    """Recent completed games with Q1-Q4 scoring (ESPN scoreboard linescores)."""
    key = f"quarter_logs_{config.QUARTER_LOOKBACK_DAYS}d"

    def _fetch():
        from data import espn

        rows = espn.recent_quarter_game_rows(days_back=config.QUARTER_LOOKBACK_DAYS)
        logger.info("Quarter logs from ESPN scoreboards (%d team-games)", len(rows))
        return rows

    try:
        rows = load_or_fetch_json(key, _fetch, force=force, max_age_hours=12.0)
        return pd.DataFrame(rows or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quarter logs fetch failed: %s", exc)
        return pd.DataFrame()


def _slate_teams(games: list[dict[str, Any]]) -> set[str]:
    teams = {
        normalize_abbrev(g.get("home_abbr")) for g in games or []
    } | {
        normalize_abbrev(g.get("away_abbr")) for g in games or []
    }
    teams.discard("")
    return teams


def _athlete_gamelog_cached(
    athlete_id: str,
    *,
    last_n: int,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Shared ESPN gamelog cache (enough rows for minutes + form)."""
    season_end = config.current_season_year() + 1
    n = max(last_n, config.MINUTES_LAST_N, config.FORM_L10_N, config.FORM_LAST_N)
    cache_key = f"player_gamelog_{athlete_id}_{season_end}_n{n}"

    def _fetch(aid=athlete_id, take=n):
        from data import espn

        return espn.player_gamelog_recent(
            aid,
            season_end_year=season_end,
            last_n=take,
        )

    try:
        return load_or_fetch_json(cache_key, _fetch, force=force, max_age_hours=8.0) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Gamelog %s failed: %s", athlete_id, exc)
        return []


def fetch_recent_player_minutes(
    games: list[dict[str, Any]],
    rosters: list[dict[str, Any]],
    players: pd.DataFrame | None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Last-N minutes + FGA usage for slate rotation players (ESPN gamelogs).
    Keyed by normalized player name.
    """
    slate_teams = _slate_teams(games)
    if not slate_teams:
        return {}

    id_by_name: dict[str, str] = {}
    team_by_name: dict[str, str] = {}
    display_by_name: dict[str, str] = {}
    for r in rosters or []:
        raw_name = str(r.get("PLAYER_NAME") or "").strip()
        name = re.sub(r"\s+", " ", raw_name).lower()
        pid = str(r.get("PLAYER_ID") or "").strip()
        abbr = normalize_abbrev(str(r.get("TEAM_ABBREVIATION") or ""))
        if name and pid and abbr in slate_teams:
            id_by_name[name] = pid
            team_by_name[name] = abbr
            display_by_name[name] = raw_name

    season_min: dict[str, float] = {}
    if players is not None and not players.empty:
        name_c = next((c for c in ("PLAYER_NAME", "player") if c in players.columns), None)
        team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in players.columns), None)
        min_c = next((c for c in ("MIN", "minutes") if c in players.columns), None)
        if name_c and team_c and min_c:
            for _, row in players.iterrows():
                abbr = normalize_abbrev(str(row.get(team_c) or ""))
                if abbr not in slate_teams:
                    continue
                key = re.sub(r"\s+", " ", str(row.get(name_c) or "").strip()).lower()
                try:
                    season_min[key] = float(row.get(min_c))
                except (TypeError, ValueError):
                    pass

    out: dict[str, dict[str, Any]] = {}
    for team in slate_teams:
        ranked = sorted(
            (
                (n, season_min.get(n, 0.0))
                for n, t in team_by_name.items()
                if t == team and n in id_by_name
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, _sm in ranked[:12]:
            recent = _athlete_gamelog_cached(
                id_by_name[name],
                last_n=config.MINUTES_LAST_N,
                force=force,
            )
            if not recent:
                continue
            mins = [
                float(r["minutes"])
                for r in recent[: config.MINUTES_LAST_N]
                if r.get("minutes") is not None
            ]
            fgas = [
                float(r["fga"])
                for r in recent[: config.MINUTES_LAST_N]
                if r.get("fga") is not None
            ]
            if len(mins) < 2:
                continue
            avg_min = float(np.mean(mins))
            avg_fga = float(np.mean(fgas)) if fgas else None
            usage = None
            if avg_fga is not None and avg_min > 0:
                usage = avg_fga / max(avg_min, 1.0) * 36.0  # FGA per 36
            out[name] = {
                "player": display_by_name.get(name, name.title()),
                "team": team,
                "recent_minutes": round(avg_min, 1),
                "season_minutes": round(float(season_min.get(name) or 0.0), 1),
                "usage_fga36": round(usage, 2) if usage is not None else None,
                "n": len(mins),
            }
    logger.info("Recent minutes profiles: %d slate players", len(out))
    return out


def fetch_game_day_injuries(
    games: list[dict[str, Any]],
    force: bool = False,
) -> list[dict[str, Any]]:
    """Day-of inactives / injury statuses from ESPN game summaries."""
    from data import espn

    out: list[dict[str, Any]] = []
    for g in games or []:
        gid = str(g.get("game_id") or "")
        if not gid:
            continue
        key = f"gameday_injuries_{gid}"

        def _fetch(event_id=gid):
            return espn.game_day_injuries(event_id)

        try:
            rows = load_or_fetch_json(
                key,
                _fetch,
                force=force,
                max_age_hours=1.0,
            ) or []
            out.extend(rows)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Game-day injuries %s failed: %s", gid, exc)
    logger.info("Game-day injury rows: %d", len(out))
    return out


def fetch_player_form(
    games: list[dict[str, Any]],
    rosters: list[dict[str, Any]],
    players: pd.DataFrame | None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Hot/cold for key rotation players on tonight's slate teams.
    Compares last-N PTS to season average (ESPN athlete gamelog).
    """
    slate_teams = _slate_teams(games)
    if not slate_teams:
        return {}

    id_by_name: dict[str, str] = {}
    team_by_name: dict[str, str] = {}
    for r in rosters or []:
        name = re.sub(r"\s+", " ", str(r.get("PLAYER_NAME") or "").strip()).lower()
        pid = str(r.get("PLAYER_ID") or "").strip()
        abbr = normalize_abbrev(str(r.get("TEAM_ABBREVIATION") or ""))
        if name and pid and abbr in slate_teams:
            id_by_name[name] = pid
            team_by_name[name] = abbr

    season_pts: dict[str, float] = {}
    season_min: dict[str, float] = {}
    if players is not None and not players.empty:
        name_c = next((c for c in ("PLAYER_NAME", "player") if c in players.columns), None)
        team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in players.columns), None)
        pts_c = next((c for c in ("PTS", "pts") if c in players.columns), None)
        min_c = next((c for c in ("MIN", "minutes") if c in players.columns), None)
        if name_c and team_c:
            for _, row in players.iterrows():
                abbr = normalize_abbrev(str(row.get(team_c) or ""))
                if abbr not in slate_teams:
                    continue
                key = re.sub(r"\s+", " ", str(row.get(name_c) or "").strip()).lower()
                if pts_c:
                    try:
                        season_pts[key] = float(row.get(pts_c))
                    except (TypeError, ValueError):
                        pass
                if min_c:
                    try:
                        season_min[key] = float(row.get(min_c))
                    except (TypeError, ValueError):
                        pass

    candidates: list[tuple[str, str, str]] = []
    for team in slate_teams:
        team_names = [
            (n, season_min.get(n, 0.0))
            for n, t in team_by_name.items()
            if t == team and n in id_by_name
        ]
        team_names.sort(key=lambda x: x[1], reverse=True)
        for name, mins in team_names[:5]:
            if mins < 18.0 and name not in season_pts:
                continue
            candidates.append((name, team, id_by_name[name]))

    out: dict[str, dict[str, Any]] = {}
    for name, team, athlete_id in candidates:
        recent = _athlete_gamelog_cached(
            athlete_id,
            last_n=config.FORM_LAST_N,
            force=force,
        )
        if len(recent) < 3:
            continue
        recent_pts = [
            float(r["pts"])
            for r in recent[: config.FORM_LAST_N]
            if r.get("pts") is not None
        ]
        if not recent_pts:
            continue
        avg_recent = float(np.mean(recent_pts))
        base = season_pts.get(name)
        if base is None or not np.isfinite(base):
            continue
        delta = avg_recent - float(base)
        tag = None
        if delta >= config.FORM_PLAYER_PTS_DELTA:
            tag = "HOT"
        elif delta <= -config.FORM_PLAYER_PTS_DELTA:
            tag = "COLD"
        if not tag:
            continue
        display = next(
            (
                str(r.get("PLAYER_NAME"))
                for r in (rosters or [])
                if re.sub(r"\s+", " ", str(r.get("PLAYER_NAME") or "").strip()).lower() == name
            ),
            name.title(),
        )
        out[name] = {
            "player": display,
            "team": team,
            "tag": tag,
            "recent_pts": round(avg_recent, 1),
            "season_pts": round(float(base), 1),
            "delta": round(delta, 1),
            "n": len(recent_pts),
            "note": (
                f"{display} {tag}: last {len(recent_pts)}g {avg_recent:.1f} PPG "
                f"vs season {float(base):.1f} ({delta:+.1f})"
            ),
        }
    logger.info("Player form tags: %d hot/cold on slate", len(out))
    return out


def fetch_player_rate_profiles(
    games: list[dict[str, Any]],
    rosters: list[dict[str, Any]],
    players: pd.DataFrame | None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    L5 / L10 per-game counting rates for slate rotation players (ESPN gamelogs).
    Used by player_sim generative means (#2).
    """
    slate_teams = _slate_teams(games)
    if not slate_teams:
        return {}

    id_by_name: dict[str, str] = {}
    team_by_name: dict[str, str] = {}
    for r in rosters or []:
        name = re.sub(r"\s+", " ", str(r.get("PLAYER_NAME") or "").strip()).lower()
        pid = str(r.get("PLAYER_ID") or "").strip()
        abbr = normalize_abbrev(str(r.get("TEAM_ABBREVIATION") or ""))
        if name and pid and abbr in slate_teams:
            id_by_name[name] = pid
            team_by_name[name] = abbr

    season_min: dict[str, float] = {}
    if players is not None and not players.empty:
        name_c = next((c for c in ("PLAYER_NAME", "player") if c in players.columns), None)
        team_c = next((c for c in ("TEAM_ABBREVIATION", "team") if c in players.columns), None)
        min_c = next((c for c in ("MIN", "minutes") if c in players.columns), None)
        if name_c and team_c and min_c:
            for _, row in players.iterrows():
                abbr = normalize_abbrev(str(row.get(team_c) or ""))
                if abbr not in slate_teams:
                    continue
                key = re.sub(r"\s+", " ", str(row.get(name_c) or "").strip()).lower()
                try:
                    season_min[key] = float(row.get(min_c))
                except (TypeError, ValueError):
                    pass

    stat_keys = ("pts", "reb", "ast", "stl", "blk", "fg3m", "minutes", "fga")
    out: dict[str, dict[str, Any]] = {}
    for team in slate_teams:
        ranked = sorted(
            (
                (n, season_min.get(n, 0.0))
                for n, t in team_by_name.items()
                if t == team and n in id_by_name
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, _sm in ranked[:12]:
            recent = _athlete_gamelog_cached(
                id_by_name[name],
                last_n=max(config.FORM_L10_N, 10),
                force=force,
            )
            if len(recent) < 2:
                continue
            entry: dict[str, Any] = {"player": name, "team": team, "n": len(recent)}
            for window, prefix in ((5, "l5"), (10, "l10")):
                chunk = recent[:window]
                if not chunk:
                    continue
                for sk in ("pts", "reb", "ast", "stl", "blk", "fg3m"):
                    vals = [
                        float(r[sk])
                        for r in chunk
                        if r.get(sk) is not None
                    ]
                    if vals:
                        # Map to PLAYER_SIM_STATS keys
                        key = {
                            "pts": "PTS",
                            "reb": "REB",
                            "ast": "AST",
                            "stl": "STL",
                            "blk": "BLK",
                            "fg3m": "FG3M",
                        }[sk]
                        entry[f"{prefix}_{key}"] = round(float(np.mean(vals)), 2)
            out[name] = entry
    logger.info("Player rate profiles: %d slate players", len(out))
    return out


def fetch_all(target_date: date, force: bool = False) -> dict[str, Any]:
    """Orchestrate all free-source pulls for a target slate."""
    logger.info("Fetching data for %s (force=%s)", target_date, force)
    games = fetch_games_for_date(target_date, force=force)
    team_logs = fetch_team_game_logs(force=force)
    prior_logs = fetch_prior_season_logs(force=force)
    prior_espn = fetch_prior_espn_seed(force=force)
    team_adv = fetch_team_advanced(force=force)
    four_factors = fetch_team_four_factors(force=force)
    rosters = fetch_rosters(force=force)
    players = fetch_player_stats(force=force)
    play_types = fetch_play_types(force=force)
    injuries = fetch_injuries(force=force)
    gameday_injuries = fetch_game_day_injuries(games, force=force)
    # Day-of / tip-time statuses override league injury board when present
    if gameday_injuries:
        by_player = {
            re.sub(r"\s+", " ", str(r.get("player") or "").strip()).lower(): r
            for r in injuries
        }
        for row in gameday_injuries:
            key = re.sub(r"\s+", " ", str(row.get("player") or "").strip()).lower()
            by_player[key] = row
        injuries = list(by_player.values())
    impact = fetch_player_impact(force=force)
    quarter_logs = fetch_quarter_logs(force=force)
    recent_minutes = fetch_recent_player_minutes(games, rosters, players, force=force)
    player_form = fetch_player_form(games, rosters, players, force=force)
    player_rates = fetch_player_rate_profiles(games, rosters, players, force=force)

    bundle = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "games": games,
        "team_logs_rows": len(team_logs),
        "roster_count": len(rosters),
        "injuries_count": len(injuries),
        "gameday_injuries_count": len(gameday_injuries),
        "impact_count": len(impact),
        "player_stats_rows": int(len(players)) if players is not None else 0,
        "quarter_log_rows": int(len(quarter_logs)) if quarter_logs is not None else 0,
        "recent_minutes_players": len(recent_minutes),
        "player_form_tags": len(player_form),
        "player_rate_profiles": len(player_rates),
        "prior_espn_rows": int(len(prior_espn)) if prior_espn is not None else 0,
        "roster_fp": (read_cache_meta("rosters_current") or {}).get("fingerprint"),
        "injury_fp": (read_cache_meta("injuries_v2") or {}).get("fingerprint"),
    }
    write_json_cache(config.CACHE_DIR / f"bundle_meta_{target_date.isoformat()}.json", bundle)

    return {
        "games": games,
        "team_logs": team_logs,
        "prior_logs": prior_logs,
        "prior_espn": prior_espn,
        "team_adv": team_adv,
        "four_factors": four_factors,
        "rosters": rosters,
        "players": players,
        "play_types": play_types,
        "injuries": injuries,
        "gameday_injuries": gameday_injuries,
        "impact": impact,
        "quarter_logs": quarter_logs,
        "recent_minutes": recent_minutes,
        "player_form": player_form,
        "player_rates": player_rates,
        "meta": bundle,
    }


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        s = str(val).replace("%", "").replace(",", "").strip()
        if s in ("", "-", "—", "N/A"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None
