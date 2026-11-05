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
# Defined after OUTPUT_DIR; grades live under output/grades/

# Reuse cached payloads newer than this (hours)
CACHE_MAX_AGE_HOURS = float(os.getenv("CACHE_MAX_AGE_HOURS", "6"))

# EWM / blend knobs — prior stays strong until ~PRIOR_BLEND_GAMES
EWM_SPAN = int(os.getenv("EWM_SPAN", "15"))
PRIOR_BLEND_GAMES = int(os.getenv("PRIOR_BLEND_GAMES", "25"))
# Extra-strong prior weight when both teams have fewer than this many games
EARLY_SEASON_GAMES = int(os.getenv("EARLY_SEASON_GAMES", "10"))

# Hot/cold form (last-N vs season EWM)
FORM_LAST_N = int(os.getenv("FORM_LAST_N", "5"))
FORM_L10_N = int(os.getenv("FORM_L10_N", "10"))
FORM_EDGE_CAP = float(os.getenv("FORM_EDGE_CAP", "3.0"))
FORM_PLAYER_PTS_DELTA = float(os.getenv("FORM_PLAYER_PTS_DELTA", "4.0"))
# Multi-horizon blend weights into form_edge (L5 / L10 / season residual)
FORM_W_L5 = float(os.getenv("FORM_W_L5", "0.45"))
FORM_W_L10 = float(os.getenv("FORM_W_L10", "0.35"))
FORM_W_SEASON = float(os.getenv("FORM_W_SEASON", "0.20"))

# Recent minutes / usage for projections (injury still wins)
MINUTES_LAST_N = int(os.getenv("MINUTES_LAST_N", "8"))
MINUTES_RECENT_WEIGHT = float(os.getenv("MINUTES_RECENT_WEIGHT", "0.65"))
# Healthy roster player with no season/recent minutes (join failure / thin sample)
HEALTHY_ROSTER_MINUTES_FLOOR = float(os.getenv("HEALTHY_ROSTER_MINUTES_FLOOR", "12.0"))
# Include slate gamelog pulls for players at/above this |EPM| even if season MIN ranks low
IMPACT_GAMELOG_EPM_FLOOR = float(os.getenv("IMPACT_GAMELOG_EPM_FLOOR", "2.5"))
# Max gamelog pulls per slate team (season MIN top-N plus high-impact extras)
GAMELOG_PER_TEAM_CAP = int(os.getenv("GAMELOG_PER_TEAM_CAP", "16"))
# Spill minutes from Out/Doubtful stars onto healthy teammates
INJURY_MINUTES_REDISTRIBUTE = os.getenv("INJURY_MINUTES_REDISTRIBUTE", "1") not in {
    "0",
    "false",
    "False",
}
INJURY_REDIST_CAP_PER_PLAYER = float(os.getenv("INJURY_REDIST_CAP_PER_PLAYER", "6.0"))
INJURY_REDIST_TEAM_CAP = float(os.getenv("INJURY_REDIST_TEAM_CAP", "18.0"))
# Player-sim rate blend: recent L5 / L10 / season
PLAYER_RATE_W_L5 = float(os.getenv("PLAYER_RATE_W_L5", "0.35"))
PLAYER_RATE_W_L10 = float(os.getenv("PLAYER_RATE_W_L10", "0.25"))
PLAYER_RATE_W_SEASON = float(os.getenv("PLAYER_RATE_W_SEASON", "0.40"))
# Usage share power in PTS allocation (higher = more star concentration)
USAGE_SHARE_POWER = float(os.getenv("USAGE_SHARE_POWER", "1.15"))
# Probabilistic injury: sample minutes factor in MC (mean still uses point estimate)
INJURY_MC_SAMPLE = os.getenv("INJURY_MC_SAMPLE", "1") not in {"0", "false", "False"}
# Auto-apply calibrated sigma when enough graded games
CALIBRATION_AUTO_APPLY = os.getenv("CALIBRATION_AUTO_APPLY", "1") not in {
    "0",
    "false",
    "False",
}
CALIBRATION_MIN_GAMES = int(os.getenv("CALIBRATION_MIN_GAMES", "20"))
CALIBRATION_SIGMA_MIN = float(os.getenv("CALIBRATION_SIGMA_MIN", "8.0"))
CALIBRATION_SIGMA_MAX = float(os.getenv("CALIBRATION_SIGMA_MAX", "14.0"))
GRADES_DIR = OUTPUT_DIR / "grades"

# Opponent-adjusted (SRS-style) iterations
SRS_ITERATIONS = int(os.getenv("SRS_ITERATIONS", "10"))
SRS_BLEND = float(os.getenv("SRS_BLEND", "0.55"))  # how much SRS net replaces raw net

# Allowed Monte Carlo sim counts (GUI + CLI)
SIM_COUNT_CHOICES = (5000, 10000, 25000, 50000, 100000, 250000, 1000000)
# Warn / progress logging when sim count is at or above this
SIM_PROGRESS_THRESHOLD = int(os.getenv("SIM_PROGRESS_THRESHOLD", "100000"))

# Denver altitude residual (points / pace)
ALTITUDE_HOME_POINTS = float(os.getenv("ALTITUDE_HOME_POINTS", "1.4"))
ALTITUDE_PACE_BONUS = float(os.getenv("ALTITUDE_PACE_BONUS", "1.0"))
ALTITUDE_TEAMS = frozenset({"DEN"})

# Rest / density residuals (points) — also safe learning knobs
REST_B2B_POINTS = float(os.getenv("REST_B2B_POINTS", "-1.5"))
REST_3IN4_POINTS = float(os.getenv("REST_3IN4_POINTS", "-0.7"))
REST_4IN6_POINTS = float(os.getenv("REST_4IN6_POINTS", "-0.5"))

# Offline learning store (cache/learning/) — paste grades + capped knobs
LEARNING_DIR = CACHE_DIR / "learning"
LEARNING_MIN_GAMES = int(os.getenv("LEARNING_MIN_GAMES", "20"))
LEARNING_TEAM_BIAS_MIN_GAMES = int(os.getenv("LEARNING_TEAM_BIAS_MIN_GAMES", "30"))
LEARNING_TEAM_BIAS_CAP = float(os.getenv("LEARNING_TEAM_BIAS_CAP", "1.5"))
LEARNING_STEP_FRAC = float(os.getenv("LEARNING_STEP_FRAC", "0.10"))  # <=10% toward target
LEARNING_AUTO_APPLY = os.getenv("LEARNING_AUTO_APPLY", "1") not in {
    "0",
    "false",
    "False",
}
LEARNING_SUGGEST_CONFIDENCE_MIN = float(os.getenv("LEARNING_SUGGEST_CONFIDENCE_MIN", "0.7"))
# Hard caps for safe knobs (lo, hi)
LEARNING_KNOB_CAPS: dict[str, tuple[float, float]] = {
    "SCORE_SIGMA": (8.0, 14.0),
    "MARGIN_SIGMA": (8.0, 16.0),
    "HOME_COURT_POINTS": (1.5, 4.0),
    "REST_B2B_POINTS": (-3.0, 1.0),
    "REST_3IN4_POINTS": (-3.0, 1.0),
    "REST_4IN6_POINTS": (-3.0, 1.0),
    "ALTITUDE_HOME_POINTS": (0.5, 3.0),
}

# Quarter tendency matchup residual (points, shrunk)
QUARTER_EDGE_CAP = float(os.getenv("QUARTER_EDGE_CAP", "1.5"))
QUARTER_LOOKBACK_DAYS = int(os.getenv("QUARTER_LOOKBACK_DAYS", "21"))
# Extra Q4 weight when projected margin is tight
CLUTCH_MARGIN_THRESHOLD = float(os.getenv("CLUTCH_MARGIN_THRESHOLD", "5.0"))
CLUTCH_Q4_MULT = float(os.getenv("CLUTCH_Q4_MULT", "1.35"))

# Home/road offensive/defensive split residual
HOME_ROAD_EDGE_CAP = float(os.getenv("HOME_ROAD_EDGE_CAP", "1.5"))

# Sanity clamps on projected totals / margins (pre-sim means)
TOTAL_MIN = float(os.getenv("TOTAL_MIN", "190.0"))
TOTAL_MAX = float(os.getenv("TOTAL_MAX", "260.0"))
MARGIN_ABS_CAP = float(os.getenv("MARGIN_ABS_CAP", "28.0"))

# Shorter TTL for volatile personnel sources (hours)
ROSTER_CACHE_MAX_AGE_HOURS = float(os.getenv("ROSTER_CACHE_MAX_AGE_HOURS", "3"))
INJURY_CACHE_MAX_AGE_HOURS = float(os.getenv("INJURY_CACHE_MAX_AGE_HOURS", "1.5"))

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
