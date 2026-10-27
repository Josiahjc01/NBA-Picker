"""Compare model projections to sportsbook lines and rank +EV bets."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import config
from data.static_arenas import normalize_abbrev
from model.predict import margin_to_win_prob, MARGIN_SIGMA

logger = logging.getLogger(__name__)


def american_to_decimal(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / abs(o)


def american_to_implied(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def remove_vig_two_way(p1: float, p2: float) -> tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1 / s, p2 / s


def kelly_fraction(p: float, decimal_odds: float, fraction: float | None = None) -> float:
    """Fractional Kelly stake as fraction of bankroll."""
    fraction = fraction if fraction is not None else config.KELLY_FRACTION
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f * fraction)


def _norm_team_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", "", (name or "").lower())
    n = re.sub(r"\s+", " ", n).strip()
    # map common full names → abbr via last token heuristics later
    return n


TEAM_NAME_TO_ABBR = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "philadelphia sixers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def name_to_abbr(name: str) -> str:
    n = _norm_team_name(name)
    if n in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[n]
    # try last word
    for full, abbr in TEAM_NAME_TO_ABBR.items():
        if n and (n in full or full.endswith(n.split()[-1] if n else "")):
            if n.split()[-1] == full.split()[-1]:
                return abbr
    return normalize_abbrev(name)


def match_odds_to_prediction(
    pred: dict[str, Any], odds_events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    home = pred["home_abbr"]
    away = pred["away_abbr"]
    for ev in odds_events or []:
        eh = name_to_abbr(str(ev.get("home_team") or ""))
        ea = name_to_abbr(str(ev.get("away_team") or ""))
        if {eh, ea} == {home, away}:
            return ev
        # soft: either side matches both names containing
        hn = _norm_team_name(str(ev.get("home_team") or ""))
        an = _norm_team_name(str(ev.get("away_team") or ""))
        if home.lower() in hn or away.lower() in an:
            if (home.lower() in hn or home.lower() in an) and (
                away.lower() in hn or away.lower() in an
            ):
                return ev
    return None


def _iter_market_outcomes(event: dict[str, Any]):
    for book in event.get("bookmakers") or []:
        book_name = book.get("title") or book.get("key") or "book"
        for market in book.get("markets") or []:
            yield book_name, market.get("key"), market.get("outcomes") or []


def spread_cover_prob(pred_margin: float, spread_line: float) -> float:
    """P(home covers) when spread is home line (e.g. -4.5)."""
    # Home covers if home_margin + (-spread)? Wait: standard home spread -4.5 means
    # home must win by > 4.5. Model margin is home-away.
    # Cover if margin > -spread_line? If spread=-4.5, need margin > 4.5
    # Actually american: home -4.5 → need margin >= 4.5; threshold = -line when line negative.
    # General: home covers when margin + spread > 0  (spread is home's number)
    adj = pred_margin + spread_line
    return margin_to_win_prob(adj, MARGIN_SIGMA)


def total_over_prob(pred_total: float, line: float) -> float:
    diff = pred_total - line
    return margin_to_win_prob(diff, sigma=14.0)


def extract_candidates(
    predictions: list[dict[str, Any]], odds_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    if not odds_events:
        return cands

    for pred in predictions:
        ev = match_odds_to_prediction(pred, odds_events)
        if not ev:
            continue
        game_label = f"{pred['away_abbr']} @ {pred['home_abbr']}"

        # Collect best ML / spread / total across books by EV later
        for book_name, mkey, outcomes in _iter_market_outcomes(ev):
            if not outcomes:
                continue
            if mkey in ("h2h", "moneyline", "ml"):
                # two-way ML
                implied = []
                parsed = []
                for o in outcomes:
                    name = str(o.get("name") or "")
                    price = o.get("price")
                    if price is None:
                        continue
                    abbr = name_to_abbr(name)
                    side = "home" if abbr == pred["home_abbr"] else "away" if abbr == pred["away_abbr"] else None
                    if side is None:
                        # compare names
                        if _norm_team_name(name) == _norm_team_name(pred["home_name"]):
                            side = "home"
                        elif _norm_team_name(name) == _norm_team_name(pred["away_name"]):
                            side = "away"
                    if side is None:
                        continue
                    imp = american_to_implied(float(price))
                    model_p = pred["home_win_prob"] if side == "home" else pred["away_win_prob"]
                    parsed.append((side, abbr or name, float(price), imp, model_p))
                if len(parsed) >= 2:
                    # de-vig using first two
                    p1, p2 = remove_vig_two_way(parsed[0][3], parsed[1][3])
                    vig_map = {parsed[0][0]: p1, parsed[1][0]: p2}
                else:
                    vig_map = {p[0]: p[3] for p in parsed}
                for side, label, price, imp, model_p in parsed:
                    fair = vig_map.get(side, imp)
                    edge = (model_p - fair) * 100.0
                    dec = american_to_decimal(price)
                    stake_frac = kelly_fraction(model_p, dec)
                    stake = min(stake_frac * config.BANKROLL_UNITS, config.MAX_STAKE_PCT)
                    cands.append(
                        {
                            "game": game_label,
                            "side": f"ML {label}",
                            "market": "ml",
                            "book": book_name,
                            "odds": int(price),
                            "model_prob": round(model_p, 4),
                            "implied_fair": round(fair, 4),
                            "edge_pct": round(edge, 2),
                            "suggested_stake": round(stake, 2),
                            "confidence": pred["confidence"],
                        }
                    )

            elif mkey in ("spreads", "spread"):
                for o in outcomes:
                    price = o.get("price")
                    point = o.get("point")
                    name = str(o.get("name") or "")
                    if price is None or point is None:
                        continue
                    abbr = name_to_abbr(name)
                    is_home = abbr == pred["home_abbr"]
                    line = float(point)
                    # If outcome is away team, home spread = -away_line
                    if is_home:
                        cover_p = spread_cover_prob(pred["margin"], line)
                        label = f"{pred['home_abbr']} {line:+g}"
                    else:
                        # away line; home equivalent = -line
                        cover_p = 1.0 - spread_cover_prob(pred["margin"], -line)
                        label = f"{pred['away_abbr']} {line:+g}"
                    imp = american_to_implied(float(price))
                    # approx fair: assume ~4.5% vig on two-way → scale
                    fair = imp / 1.045
                    edge = (cover_p - fair) * 100.0
                    dec = american_to_decimal(float(price))
                    stake = min(
                        kelly_fraction(cover_p, dec) * config.BANKROLL_UNITS,
                        config.MAX_STAKE_PCT,
                    )
                    cands.append(
                        {
                            "game": game_label,
                            "side": f"Spread {label}",
                            "market": "spread",
                            "book": book_name,
                            "odds": int(price),
                            "line": line,
                            "model_prob": round(cover_p, 4),
                            "implied_fair": round(fair, 4),
                            "edge_pct": round(edge, 2),
                            "suggested_stake": round(stake, 2),
                            "confidence": pred["confidence"],
                        }
                    )

            elif mkey in ("totals", "total"):
                for o in outcomes:
                    price = o.get("price")
                    point = o.get("point")
                    name = str(o.get("name") or "").lower()
                    if price is None or point is None:
                        continue
                    line = float(point)
                    over_p = total_over_prob(pred["total"], line)
                    model_p = over_p if "over" in name else 1.0 - over_p
                    label = f"{name.title()} {line:g}"
                    imp = american_to_implied(float(price))
                    fair = imp / 1.045
                    edge = (model_p - fair) * 100.0
                    dec = american_to_decimal(float(price))
                    stake = min(
                        kelly_fraction(model_p, dec) * config.BANKROLL_UNITS,
                        config.MAX_STAKE_PCT,
                    )
                    cands.append(
                        {
                            "game": game_label,
                            "side": f"Total {label}",
                            "market": "total",
                            "book": book_name,
                            "odds": int(price),
                            "line": line,
                            "model_prob": round(model_p, 4),
                            "implied_fair": round(fair, 4),
                            "edge_pct": round(edge, 2),
                            "suggested_stake": round(stake, 2),
                            "confidence": pred["confidence"],
                        }
                    )
    return cands


def rank_top_wagers(
    predictions: list[dict[str, Any]],
    odds_events: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    cands = extract_candidates(predictions, odds_events)
    filtered = [
        c
        for c in cands
        if c["edge_pct"] >= config.MIN_EDGE_PCT and c["confidence"] >= 0.4 and c["suggested_stake"] > 0
    ]
    # Deduplicate similar sides: keep best edge per game+side label
    best: dict[str, dict[str, Any]] = {}
    for c in filtered:
        key = f"{c['game']}|{c['side']}"
        if key not in best or c["edge_pct"] > best[key]["edge_pct"]:
            best[key] = c
    ranked = sorted(best.values(), key=lambda x: (x["edge_pct"], x["confidence"]), reverse=True)
    return ranked[:top_n]
