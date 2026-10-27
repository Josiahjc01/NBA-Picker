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
from data.cache_util import load_or_fetch_json, write_json_cache, read_json_cache
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
    logger.info("Team logs from ESPN schedules (%d)", len(rows))
    return rows


def fetch_team_game_logs(season: str | None = None, force: bool = False) -> pd.DataFrame:
    season = season or config.season_string()
    key = f"team_logs_{season}"

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
# Injuries — official NBA PDF/HTML or ESPN fallback
# ---------------------------------------------------------------------------

def _parse_espn_injuries() -> list[dict[str, Any]]:
    url = "https://www.espn.com/nba/injuries"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    injuries: list[dict[str, Any]] = []
    # ESPN ResponsiveTable: NAME, POS, EST. RETURN, STATUS, COMMENT
    tables = soup.select("div.ResponsiveTable")
    for table in tables:
        team_el = table.select_one(".Table__Title, div.Table__Title")
        team_name = team_el.get_text(strip=True) if team_el else ""
        for row in table.select("tbody tr"):
            cells = [c.get_text(" ", strip=True) for c in row.select("td")]
            if len(cells) < 4:
                continue
            player = cells[0]
            # Prefer explicit STATUS column (index 3); fall back to last short token
            status = cells[3] if len(cells) >= 4 else cells[-1]
            comment = cells[4] if len(cells) >= 5 else ""
            injuries.append(
                {
                    "team": team_name,
                    "player": player,
                    "status": status,
                    "comment": comment,
                    "source": "espn",
                }
            )
    return injuries


def _parse_official_injury_report() -> list[dict[str, Any]]:
    """Try official.nba.com injury report landing / CDN listing."""
    # Official reports are often PDFs named by date; HTML index varies.
    candidates = [
        "https://official.nba.com/nba-injury-report/",
        "https://ak-static.cms.nba.com/referee/injury/",
    ]
    injuries: list[dict[str, Any]] = []
    for url in candidates:
        try:
            resp = _get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text("\n", strip=True)
            # Soft parse: look for Out/Doubtful/Questionable lines
            for line in text.splitlines():
                low = line.lower()
                if any(k in low for k in ("out", "doubtful", "questionable", "available")):
                    if len(line) < 8 or len(line) > 200:
                        continue
                    injuries.append(
                        {
                            "team": "",
                            "player": line.split("-")[0].strip()[:80],
                            "status": line,
                            "comment": "",
                            "source": "official.nba.com",
                        }
                    )
            if injuries:
                return injuries[:500]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Official injury scrape %s failed: %s", url, exc)
    return injuries


def fetch_injuries(force: bool = False) -> list[dict[str, Any]]:
    key = "injuries"

    def _fetch():
        official = _parse_official_injury_report()
        if official:
            return official
        return _parse_espn_injuries()

    try:
        return load_or_fetch_json(key, _fetch, force=force) or []
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
    key = "player_impact"

    def _fetch():
        for name, fn in (
            ("dunksandthrees", _scrape_dunksandthrees_epm),
            ("nbarapm", _scrape_nbarapm),
            ("darko", _scrape_darko),
        ):
            try:
                rows = fn()
                if rows:
                    logger.info("Player impact from %s (%d rows)", name, len(rows))
                    return rows
            except Exception as exc:  # noqa: BLE001
                logger.warning("Impact scrape %s failed: %s", name, exc)
        proxy = _bpm_proxy_from_nba()
        if proxy:
            logger.warning("Using nba_api z-score proxy for player impact")
            return proxy
        return []

    try:
        return load_or_fetch_json(key, _fetch, force=force) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Player impact unavailable: %s", exc)
        return []


def fetch_all(target_date: date, force: bool = False) -> dict[str, Any]:
    """Orchestrate all free-source pulls for a target slate."""
    logger.info("Fetching data for %s (force=%s)", target_date, force)
    games = fetch_games_for_date(target_date, force=force)
    team_logs = fetch_team_game_logs(force=force)
    prior_logs = fetch_prior_season_logs(force=force)
    team_adv = fetch_team_advanced(force=force)
    four_factors = fetch_team_four_factors(force=force)
    players = fetch_player_stats(force=force)
    play_types = fetch_play_types(force=force)
    injuries = fetch_injuries(force=force)
    impact = fetch_player_impact(force=force)

    bundle = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "games": games,
        "team_logs_rows": len(team_logs),
        "injuries_count": len(injuries),
        "impact_count": len(impact),
    }
    write_json_cache(config.CACHE_DIR / f"bundle_meta_{target_date.isoformat()}.json", bundle)

    return {
        "games": games,
        "team_logs": team_logs,
        "prior_logs": prior_logs,
        "team_adv": team_adv,
        "four_factors": four_factors,
        "players": players,
        "play_types": play_types,
        "injuries": injuries,
        "impact": impact,
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
