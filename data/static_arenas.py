"""Static NBA arena locations for rest / travel / timezone context."""

from __future__ import annotations

import math
from typing import Any

# Approximate arena lat/lon + IANA timezone
ARENAS: dict[str, dict[str, Any]] = {
    "ATL": {"city": "Atlanta", "lat": 33.7573, "lon": -84.3963, "tz": "America/New_York"},
    "BOS": {"city": "Boston", "lat": 42.3662, "lon": -71.0621, "tz": "America/New_York"},
    "BKN": {"city": "Brooklyn", "lat": 40.6826, "lon": -73.9754, "tz": "America/New_York"},
    "CHA": {"city": "Charlotte", "lat": 35.2251, "lon": -80.8392, "tz": "America/New_York"},
    "CHI": {"city": "Chicago", "lat": 41.8807, "lon": -87.6742, "tz": "America/Chicago"},
    "CLE": {"city": "Cleveland", "lat": 41.4965, "lon": -81.6882, "tz": "America/New_York"},
    "DAL": {"city": "Dallas", "lat": 32.7905, "lon": -96.8103, "tz": "America/Chicago"},
    "DEN": {"city": "Denver", "lat": 39.7487, "lon": -105.0077, "tz": "America/Denver"},
    "DET": {"city": "Detroit", "lat": 42.3411, "lon": -83.0550, "tz": "America/New_York"},
    "GSW": {"city": "San Francisco", "lat": 37.7680, "lon": -122.3877, "tz": "America/Los_Angeles"},
    "HOU": {"city": "Houston", "lat": 29.7508, "lon": -95.3621, "tz": "America/Chicago"},
    "IND": {"city": "Indianapolis", "lat": 39.7640, "lon": -86.1555, "tz": "America/Indiana/Indianapolis"},
    "LAC": {"city": "Inglewood", "lat": 33.9581, "lon": -118.3416, "tz": "America/Los_Angeles"},
    "LAL": {"city": "Los Angeles", "lat": 34.0430, "lon": -118.2673, "tz": "America/Los_Angeles"},
    "MEM": {"city": "Memphis", "lat": 35.1382, "lon": -90.0506, "tz": "America/Chicago"},
    "MIA": {"city": "Miami", "lat": 25.7814, "lon": -80.1870, "tz": "America/New_York"},
    "MIL": {"city": "Milwaukee", "lat": 43.0436, "lon": -87.9169, "tz": "America/Chicago"},
    "MIN": {"city": "Minneapolis", "lat": 44.9795, "lon": -93.2761, "tz": "America/Chicago"},
    "NOP": {"city": "New Orleans", "lat": 29.9490, "lon": -90.0821, "tz": "America/Chicago"},
    "NYK": {"city": "New York", "lat": 40.7505, "lon": -73.9934, "tz": "America/New_York"},
    "OKC": {"city": "Oklahoma City", "lat": 35.4634, "lon": -97.5151, "tz": "America/Chicago"},
    "ORL": {"city": "Orlando", "lat": 28.5392, "lon": -81.3839, "tz": "America/New_York"},
    "PHI": {"city": "Philadelphia", "lat": 39.9012, "lon": -75.1720, "tz": "America/New_York"},
    "PHX": {"city": "Phoenix", "lat": 33.4457, "lon": -112.0712, "tz": "America/Phoenix"},
    "POR": {"city": "Portland", "lat": 45.5316, "lon": -122.6668, "tz": "America/Los_Angeles"},
    "SAC": {"city": "Sacramento", "lat": 38.5803, "lon": -121.4997, "tz": "America/Los_Angeles"},
    "SAS": {"city": "San Antonio", "lat": 29.4270, "lon": -98.4375, "tz": "America/Chicago"},
    "TOR": {"city": "Toronto", "lat": 43.6435, "lon": -79.3791, "tz": "America/Toronto"},
    "UTA": {"city": "Salt Lake City", "lat": 40.7683, "lon": -111.9011, "tz": "America/Denver"},
    "WAS": {"city": "Washington", "lat": 38.8981, "lon": -77.0209, "tz": "America/New_York"},
}

# Alternate abbreviations seen in feeds
ABBREV_ALIASES = {
    "BRK": "BKN",
    "BRO": "BKN",
    "GS": "GSW",
    "GOLDEN STATE": "GSW",
    "NO": "NOP",
    "NOH": "NOP",
    "NY": "NYK",
    "PHO": "PHX",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "CHO": "CHA",
    "UTH": "UTA",
    "SAN": "SAS",
}


def normalize_abbrev(abbr: str | None) -> str:
    if not abbr:
        return ""
    a = abbr.strip().upper()
    return ABBREV_ALIASES.get(a, a)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_miles(from_abbr: str, to_abbr: str) -> float:
    a = normalize_abbrev(from_abbr)
    b = normalize_abbrev(to_abbr)
    if a not in ARENAS or b not in ARENAS or a == b:
        return 0.0
    fa, ta = ARENAS[a], ARENAS[b]
    return haversine_miles(fa["lat"], fa["lon"], ta["lat"], ta["lon"])


def timezone_shift_hours(from_abbr: str, to_abbr: str) -> float:
    """Rough TZ hour delta using fixed US offsets (no DST precision)."""
    offsets = {
        "America/New_York": -5,
        "America/Toronto": -5,
        "America/Chicago": -6,
        "America/Indiana/Indianapolis": -5,
        "America/Denver": -7,
        "America/Phoenix": -7,
        "America/Los_Angeles": -8,
    }
    a = normalize_abbrev(from_abbr)
    b = normalize_abbrev(to_abbr)
    if a not in ARENAS or b not in ARENAS:
        return 0.0
    oa = offsets.get(ARENAS[a]["tz"], -5)
    ob = offsets.get(ARENAS[b]["tz"], -5)
    return float(ob - oa)
