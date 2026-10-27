"""Runtime configuration for NBA Picker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"

# Reuse cached payloads newer than this (hours)
CACHE_MAX_AGE_HOURS = float(os.getenv("CACHE_MAX_AGE_HOURS", "6"))

# EWM / blend knobs
EWM_SPAN = int(os.getenv("EWM_SPAN", "15"))
PRIOR_BLEND_GAMES = int(os.getenv("PRIOR_BLEND_GAMES", "10"))

# HTTP
USER_AGENT = (
    "Mozilla/5.0 (compatible; NBA-Picker/1.0; local)"
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))
# nba_api / stats.nba.com is often blocked from cloud IPs — keep attempts short
NBA_API_TIMEOUT = int(os.getenv("NBA_API_TIMEOUT", "12"))

# League pacing defaults when data is thin
DEFAULT_PACE = 100.0
DEFAULT_OFF_RTG = 112.0
DEFAULT_DEF_RTG = 112.0
LEAGUE_AVG_TOTAL = 224.0

# Home court residual (points)
HOME_COURT_POINTS = 2.5

# Matchup residual cap (points) — keep small vs player/team layer
MATCHUP_RESIDUAL_CAP = 2.0


def current_season_year() -> int:
    """NBA season start year (Oct→next calendar year still same season)."""
    override = os.getenv("NBA_SEASON_YEAR")
    if override:
        return int(override)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def season_string(year: int | None = None) -> str:
    y = year if year is not None else current_season_year()
    return f"{y}-{str(y + 1)[-2:]}"
