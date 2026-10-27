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

# EWM / blend knobs — prior stays strong until ~PRIOR_BLEND_GAMES
EWM_SPAN = int(os.getenv("EWM_SPAN", "15"))
PRIOR_BLEND_GAMES = int(os.getenv("PRIOR_BLEND_GAMES", "25"))

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
HOME_COURT_POINTS = 2.8

# Matchup residual cap (points) — keep small vs player/team layer
MATCHUP_RESIDUAL_CAP = 2.0

# Monte Carlo score sims (local, vectorized — no extra scrapes)
SIM_COUNT = int(os.getenv("SIM_COUNT", "10000"))
# Margin / score noise (NBA game margin SD ~11–12; team score SD ~11)
MARGIN_SIGMA = float(os.getenv("MARGIN_SIGMA", "11.5"))
SCORE_SIGMA = float(os.getenv("SCORE_SIGMA", "11.0"))
# Shrink absolute player-impact vs team ratings (injury_delta still full strength)
PLAYER_TALENT_SHRINK = float(os.getenv("PLAYER_TALENT_SHRINK", "0.18"))
# Rotation player Out: minimum EPM-like cost if impact table is weak/noisy
ROTATION_OUT_EPM_FLOOR = float(os.getenv("ROTATION_OUT_EPM_FLOOR", "1.5"))
ROTATION_MIN_MINUTES = float(os.getenv("ROTATION_MIN_MINUTES", "18.0"))


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
