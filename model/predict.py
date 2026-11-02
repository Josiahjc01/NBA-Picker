"""Generate team ratings, projected scores, win probs via Monte Carlo sims."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

import config
from data.process import (
    home_road_matchup_residual,
    playtype_matchup_residual,
    project_quarter_breakdown,
    quarter_matchup_residual,
    shot_profile_residual,
)
from data.static_arenas import normalize_abbrev
from model.calibrate import effective_margin_sigma, effective_score_sigma
from model.player_sim import run_matchup_player_sims

logger = logging.getLogger(__name__)


def _team_row(ratings: pd.DataFrame, abbr: str) -> dict[str, float]:
    abbr = normalize_abbrev(abbr)
    defaults = {
        "off_rtg": config.DEFAULT_OFF_RTG,
        "def_rtg": config.DEFAULT_DEF_RTG,
        "net_rtg": 0.0,
        "pace": config.DEFAULT_PACE,
        "efg": 0.53,
        "tov_pct": 0.13,
        "orb_pct": 0.28,
        "ft_rate": 0.25,
        "games": 0,
        "blend_weight_current": 0.0,
    }
    if ratings is None or ratings.empty:
        return defaults
    hit = ratings[ratings["team"] == abbr]
    if hit.empty:
        return defaults
    r = hit.iloc[0].to_dict()
    for k, v in defaults.items():
        if k not in r or r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])):
            r[k] = v
    return r


def _finite(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def margin_to_win_prob(margin: float, sigma: float | None = None) -> float:
    """P(home win) from expected margin via logistic approx to normal CDF."""
    sig = sigma if sigma is not None else effective_margin_sigma()
    z = margin / max(sig, 1e-6)
    return float(1.0 / (1.0 + math.exp(-1.7 * z)))


def _early_season_weight(games_n: int) -> float:
    """How much to trust current-season sample vs durable prior (0..1)."""
    k = float(max(config.PRIOR_BLEND_GAMES, 1))
    early = float(max(config.EARLY_SEASON_GAMES, 1))
    if games_n < early:
        # Extra-conservative under EARLY_SEASON_GAMES
        return float((games_n / early) ** 2 * (early / (early + k)))
    return float(games_n / (games_n + k))


def _sanity_clamp_scores(home_pts: float, away_pts: float) -> tuple[float, float]:
    """Clamp absurd totals / margins while preserving direction of the margin."""
    margin = home_pts - away_pts
    total = home_pts + away_pts
    if abs(margin) > config.MARGIN_ABS_CAP:
        margin = math.copysign(config.MARGIN_ABS_CAP, margin)
    total = float(np.clip(total, config.TOTAL_MIN, config.TOTAL_MAX))
    home_pts = (total + margin) / 2.0
    away_pts = (total - margin) / 2.0
    home_pts = float(np.clip(home_pts, 88.0, 140.0))
    away_pts = float(np.clip(away_pts, 88.0, 140.0))
    return home_pts, away_pts


def _mean_scores(
    game: dict[str, Any],
    features: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    """
    Deterministic mean home/away points from durable team priors + injury-first
    player layer. Absolute player impact is heavily shrunk so early-season
    z-score noise cannot turn mismatches into coin-flips.
    """
    ratings = features.get("ratings")
    home = normalize_abbrev(game.get("home_abbr"))
    away = normalize_abbrev(game.get("away_abbr"))
    hr = _team_row(ratings, home)
    ar = _team_row(ratings, away)

    games_n = min(int(hr.get("games") or 0), int(ar.get("games") or 0))
    w_cur = _early_season_weight(games_n)
    # Prefer explicit blend weight when present (from process.blend_with_prior)
    w_h = _finite(hr.get("blend_weight_current"), w_cur)
    w_a = _finite(ar.get("blend_weight_current"), w_cur)
    w_sample = min(w_h, w_a, w_cur)

    ctx = features.get("context") or {}
    h_ctx = ctx.get(f"{game.get('game_id')}:{home}") or {}
    a_ctx = ctx.get(f"{game.get('game_id')}:{away}") or {}

    # Pace: blend toward league default early; rest/fatigue slows tired teams
    pace_raw = 0.5 * (float(hr["pace"]) + float(ar["pace"]))
    pace = (1.0 - w_sample) * config.DEFAULT_PACE + w_sample * pace_raw
    pace += 0.5 * (_finite(h_ctx.get("pace_adj")) + _finite(a_ctx.get("pace_adj")))
    pace = float(np.clip(0.5 * (pace_raw + pace), 90.0, 110.0))

    impact = features.get("team_impact") or {}
    hi = impact.get(home) or {}
    ai = impact.get(away) or {}

    # Team offense/defense from Net Rating stack (prior already blended in ratings)
    h_off = float(np.clip(hr["off_rtg"], 98.0, 125.0))
    h_def = float(np.clip(hr["def_rtg"], 98.0, 125.0))
    a_off = float(np.clip(ar["off_rtg"], 98.0, 125.0))
    a_def = float(np.clip(ar["def_rtg"], 98.0, 125.0))

    # Four-factors soft nudge (scaled by sample trust) — eFG, TOV, ORB, FT rate
    ff = 0.35 + 0.65 * w_sample
    def _ff_off(row: dict[str, float]) -> float:
        return (
            12.0 * (float(row["efg"]) - 0.53)
            - 25.0 * (float(row["tov_pct"]) - 0.13)
            + 8.0 * (float(row.get("orb_pct") or 0.28) - 0.28)
            + 6.0 * (float(row.get("ft_rate") or 0.25) - 0.25)
        )

    h_off += ff * _ff_off(hr)
    a_off += ff * _ff_off(ar)

    # Injury-first: full injury_delta; talent residual heavily shrunk
    h_inj = _finite(hi.get("injury_delta"))
    a_inj = _finite(ai.get("injury_delta"))
    h_talent = _finite(hi.get("healthy_impact"))
    a_talent = _finite(ai.get("healthy_impact"))
    talent_scale = config.PLAYER_TALENT_SHRINK * (0.25 + 0.75 * w_sample)
    h_pi = h_inj + talent_scale * h_talent
    a_pi = a_inj + talent_scale * a_talent
    h_pi = float(np.clip(h_pi, -8.0, 8.0))
    a_pi = float(np.clip(a_pi, -8.0, 8.0))

    h_off += 0.55 * h_pi
    h_def -= 0.25 * h_pi
    a_off += 0.55 * a_pi
    a_def -= 0.25 * a_pi

    league = config.DEFAULT_OFF_RTG
    # Opp-adjusted scoring
    home_pts = pace * (h_off - (a_def - league) * 0.55) / 100.0
    away_pts = pace * (a_off - (h_def - league) * 0.55) / 100.0

    # SRS-style net pull so large Net Rating gaps show up in the margin
    h_net = h_off - h_def
    a_net = a_off - a_def
    net_margin = 0.45 * (h_net - a_net)
    home_pts += net_margin / 2.0
    away_pts -= net_margin / 2.0

    # Full context (home court, B2B, travel) — do not damp to 0.6
    home_pts += _finite(
        h_ctx.get("context_points"),
        config.HOME_COURT_POINTS if not h_ctx else 0.0,
    )
    away_pts += _finite(a_ctx.get("context_points"), 0.0)

    # Offline learning: tiny per-team additive biases (capped; min-N gated at write time)
    try:
        from data.learning_store import team_bias_for

        home_pts += team_bias_for(home)
        away_pts += team_bias_for(away)
    except Exception:  # noqa: BLE001
        pass

    # Multi-horizon form residual (ratings already absorbed part of form_edge)
    form = features.get("team_form") or {}
    h_form = form.get(home) or {}
    a_form = form.get(away) or {}
    form_scale = 0.35 + 0.30 * w_sample
    h_form_edge = form_scale * _finite(h_form.get("form_edge"))
    a_form_edge = form_scale * _finite(a_form.get("form_edge"))
    home_pts += 0.40 * h_form_edge
    away_pts += 0.40 * a_form_edge

    # Key player hot/cold (small nudge; notes carry the signal)
    player_form = features.get("player_form") or {}
    player_notes: list[str] = []
    h_player_form = 0.0
    a_player_form = 0.0
    for pf in player_form.values():
        if normalize_abbrev(pf.get("team")) == home:
            delta = _finite(pf.get("delta"))
            bump = float(np.clip(0.06 * delta, -0.8, 0.8))
            h_player_form += bump
            if pf.get("note"):
                player_notes.append(str(pf["note"]))
        elif normalize_abbrev(pf.get("team")) == away:
            delta = _finite(pf.get("delta"))
            bump = float(np.clip(0.06 * delta, -0.8, 0.8))
            a_player_form += bump
            if pf.get("note"):
                player_notes.append(str(pf["note"]))
    home_pts += h_player_form
    away_pts += a_player_form

    matchup_edge, matchup_notes = playtype_matchup_residual(
        features.get("play_types"),
        home,
        away,
        ratings=features.get("ratings"),
    )
    hr_edge, hr_notes = home_road_matchup_residual(
        features.get("home_road") or {}, home, away
    )
    shot_edge, shot_notes = shot_profile_residual(
        features.get("team_adv"), home, away
    )
    matchup_edge += hr_edge + shot_edge
    matchup_notes = list(matchup_notes) + list(hr_notes) + list(shot_notes)

    # Preliminary margin for clutch/Q4 decision (before quarter residual)
    prelim_margin = home_pts - away_pts
    clutch = abs(prelim_margin) < config.CLUTCH_MARGIN_THRESHOLD
    q_edge, q_notes = quarter_matchup_residual(
        features.get("quarter_profiles") or {},
        home,
        away,
        clutch=clutch,
    )
    matchup_edge += q_edge
    # Keep quarter notes separate from play-type / shot / home-road notes
    playtype_notes = list(matchup_notes)

    matchup_edge = float(
        np.clip(matchup_edge, -config.MATCHUP_RESIDUAL_CAP - 1.5, config.MATCHUP_RESIDUAL_CAP + 1.5)
    )
    home_pts += matchup_edge / 2.0
    away_pts -= matchup_edge / 2.0

    # Mild regression of totals toward league average early
    total = home_pts + away_pts
    league_tot = config.LEAGUE_AVG_TOTAL
    total = (1.0 - 0.25 * (1.0 - w_sample)) * total + (0.25 * (1.0 - w_sample)) * league_tot
    # Keep margin; rescale points to adjusted total
    margin = home_pts - away_pts
    home_pts = (total + margin) / 2.0
    away_pts = (total - margin) / 2.0

    home_pts, away_pts = _sanity_clamp_scores(home_pts, away_pts)

    form_notes = [n for n in (h_form.get("note"), a_form.get("note")) if n]
    meta = {
        "home": home,
        "away": away,
        "hr": hr,
        "ar": ar,
        "hi": hi,
        "ai": ai,
        "h_off": h_off,
        "h_def": h_def,
        "a_off": a_off,
        "a_def": a_def,
        "h_pi": h_pi,
        "a_pi": a_pi,
        "pace": pace,
        "w_sample": w_sample,
        "games_n": games_n,
        "matchup_edge": matchup_edge,
        "matchup_notes": playtype_notes,
        "quarter_notes": q_notes,
        "form_notes": form_notes,
        "player_form_notes": player_notes[:4],
        "h_form": h_form,
        "a_form": a_form,
        "clutch": clutch,
        "h_ctx": h_ctx,
        "a_ctx": a_ctx,
    }
    return home_pts, away_pts, meta


def _normalize_sim_count(n: int | None) -> int:
    """Clamp to allowed GUI/CLI choices; default SIM_COUNT."""
    if n is None:
        return int(config.SIM_COUNT)
    n = int(n)
    if n in config.SIM_COUNT_CHOICES:
        return n
    # Nearest allowed choice
    return min(config.SIM_COUNT_CHOICES, key=lambda c: abs(c - n))


def _injury_uncertainty_sigma(features: dict[str, Any], home: str, away: str) -> float:
    """
    Extra score noise from probabilistic injury statuses (Q/Doubtful mass).
    Keeps mean projection on the point-estimate path; widens MC only.
    """
    if not config.INJURY_MC_SAMPLE:
        return 0.0
    minutes = features.get("minutes")
    if minutes is None or (isinstance(minutes, pd.DataFrame) and minutes.empty):
        return 0.0
    vol = 0.0
    for team in (normalize_abbrev(home), normalize_abbrev(away)):
        sub = minutes[minutes["team"].map(lambda x: normalize_abbrev(str(x))) == team]
        for _, r in sub.iterrows():
            status = str(r.get("injury_status") or "")
            if not status:
                continue
            from data.process import injury_factor_distribution

            p_out, p_lim, _p_full = injury_factor_distribution(status)
            base = float(r.get("proj_minutes") or 0.0)
            if base < 12.0:
                continue
            # Uncertainty mass * role minutes → points of sigma
            uncertain = (p_out + 0.5 * p_lim) * (base / 48.0)
            vol += 1.8 * uncertain
    return float(np.clip(vol, 0.0, 4.0))


def simulate_game(
    mean_home: float,
    mean_away: float,
    *,
    n: int | None = None,
    seed: int | None = None,
    return_scores: bool = False,
    extra_sigma: float = 0.0,
) -> dict[str, Any]:
    """
    Vectorized Monte Carlo: correlated score noise around mean projections.
    Returns win rates + score distribution summaries (not extra scrapes).
    When return_scores=True, also includes home_scores / away_scores arrays
    for player-level sims (not serialized to JSON).
    """
    n = _normalize_sim_count(n)
    n = max(100, n)
    if n >= config.SIM_PROGRESS_THRESHOLD:
        logger.info("Running %s Monte Carlo sims (vectorized)...", f"{n:,}")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    base_sig = float(effective_score_sigma())
    sig = math.sqrt(base_sig * base_sig + max(0.0, float(extra_sigma)) ** 2)
    # Correlation ~0.15 between team scores (pace/shared environment)
    rho = 0.15
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + math.sqrt(max(0.0, 1.0 - rho * rho)) * z[:, 1]
    # Optional injury-status sampling: shift means slightly per sim
    home = mean_home + sig * z[:, 0]
    away = mean_away + sig * z[:, 1]
    if config.INJURY_MC_SAMPLE and extra_sigma > 0.15:
        # Correlated tip-time uncertainty residual (same draw direction biases one side)
        inj_z = rng.standard_normal(n)
        home = home + 0.35 * extra_sigma * inj_z
        away = away - 0.25 * extra_sigma * inj_z
    # Floor
    home = np.clip(home, 70.0, 160.0)
    away = np.clip(away, 70.0, 160.0)
    home_wins = home > away
    ties = np.isclose(home, away)
    # Ties → coin flip for win%
    if ties.any():
        flip = rng.random(ties.sum()) < 0.5
        home_wins = home_wins.copy()
        home_wins[ties] = flip

    margins = home - away
    totals = home + away
    elapsed = time.perf_counter() - t0
    if n >= config.SIM_PROGRESS_THRESHOLD:
        logger.info("Finished %s team-score sims in %.3fs", f"{n:,}", elapsed)
    if n >= 1_000_000 and elapsed > 2.0:
        logger.warning(
            "1,000,000 sims took %.1fs - still usable; prefer 100k/250k for interactive runs",
            elapsed,
        )
    out: dict[str, Any] = {
        "sim_count": n,
        "home_win_prob": float(home_wins.mean()),
        "away_win_prob": float(1.0 - home_wins.mean()),
        "home_pts_mean": float(home.mean()),
        "away_pts_mean": float(away.mean()),
        "home_pts_p10": float(np.percentile(home, 10)),
        "home_pts_p90": float(np.percentile(home, 90)),
        "away_pts_p10": float(np.percentile(away, 10)),
        "away_pts_p90": float(np.percentile(away, 90)),
        "total_mean": float(totals.mean()),
        "total_p10": float(np.percentile(totals, 10)),
        "total_p90": float(np.percentile(totals, 90)),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "cover_home_minus_3": float((margins > 3.0).mean()),
        "cover_home_minus_5": float((margins > 5.0).mean()),
        "over_225": float((totals > 225.0).mean()),
        "elapsed_sec": round(elapsed, 4),
    }
    if return_scores:
        out["home_scores"] = home
        out["away_scores"] = away
    return out


def project_game(
    game: dict[str, Any],
    features: dict[str, Any],
    *,
    sim_count: int | None = None,
) -> dict[str, Any]:
    home_pts, away_pts, meta = _mean_scores(game, features)
    home = meta["home"]
    away = meta["away"]
    hr, ar = meta["hr"], meta["ar"]
    hi, ai = meta["hi"], meta["ai"]

    # Deterministic seed from game_id for stable re-runs
    gid = str(game.get("game_id") or f"{away}{home}{game.get('game_date')}")
    seed = abs(hash(gid)) % (2**32 - 1)
    inj_sigma = _injury_uncertainty_sigma(features, home, away)
    sims = simulate_game(
        home_pts,
        away_pts,
        n=sim_count,
        seed=seed,
        return_scores=True,
        extra_sigma=inj_sigma,
    )

    # Prefer sim win% ; mean scores from sims (close to analytic means)
    win_p = sims["home_win_prob"]
    mean_h = sims["home_pts_mean"]
    mean_a = sims["away_pts_mean"]
    margin = mean_h - mean_a

    # Player-level MC on the same score paths (means for AI paste, maxes for Highlights)
    t_player = time.perf_counter()
    if sims["sim_count"] >= config.SIM_PROGRESS_THRESHOLD:
        logger.info(
            "Running player-level MC for %s @ %s (%s sims)...",
            away,
            home,
            f"{sims['sim_count']:,}",
        )
    player_sim = run_matchup_player_sims(
        features,
        home,
        away,
        sims["home_scores"],
        sims["away_scores"],
        seed=(seed + 17) % (2**32 - 1),
    )
    player_elapsed = time.perf_counter() - t_player
    if sims["sim_count"] >= config.SIM_PROGRESS_THRESHOLD:
        logger.info(
            "Player-level MC finished in %.3fs (team-score MC %.3fs)",
            player_elapsed,
            float(sims.get("elapsed_sec") or 0.0),
        )
    # Drop raw score arrays — not JSON-serializable at 1M
    sims.pop("home_scores", None)
    sims.pop("away_scores", None)

    games_n = meta["games_n"]
    confidence = float(np.clip(0.40 + 0.025 * games_n, 0.40, 0.88))
    confidence *= 0.85 + 0.15 * meta["w_sample"]
    if abs(margin) < 2.0:
        confidence = min(confidence, 0.55)
    if abs(margin) >= 6.0:
        confidence = min(0.9, confidence + 0.08)

    injury_notes = list(hi.get("injury_notes") or []) + list(ai.get("injury_notes") or [])
    if hi.get("injury_delta"):
        injury_notes.insert(0, f"{home} injury dNet {hi['injury_delta']:+.2f}")
    if ai.get("injury_delta"):
        injury_notes.insert(0, f"{away} injury dNet {ai['injury_delta']:+.2f}")

    flags = list(
        dict.fromkeys(
            (meta["h_ctx"].get("flags") or []) + (meta["a_ctx"].get("flags") or [])
        )
    )

    form_tags = []
    for side_form in (meta.get("a_form") or {}, meta.get("h_form") or {}):
        tag = side_form.get("tag")
        if tag in {"HOT", "COLD"}:
            form_tags.append(f"{side_form.get('team')} {tag}")

    quarters = project_quarter_breakdown(
        features.get("quarter_profiles") or {},
        home,
        away,
        mean_h,
        mean_a,
        clutch=bool(meta.get("clutch")),
        pace=meta.get("pace"),
    )

    total_elapsed = round(
        float(sims.get("elapsed_sec") or 0.0) + player_elapsed, 4
    )

    return {
        "game_id": game.get("game_id"),
        "game_date": game.get("game_date"),
        "home_abbr": home,
        "away_abbr": away,
        "home_name": game.get("home_name") or home,
        "away_name": game.get("away_name") or away,
        "home_pts": round(mean_h, 1),
        "away_pts": round(mean_a, 1),
        "total": round(mean_h + mean_a, 1),
        "margin": round(margin, 1),
        "home_win_prob": round(win_p, 4),
        "away_win_prob": round(1.0 - win_p, 4),
        "confidence": round(confidence, 3),
        "pace": round(meta["pace"], 1),
        "sim_count": sims["sim_count"],
        "sim_elapsed_sec": total_elapsed,
        "clutch_q4": bool(meta.get("clutch")),
        "form_tags": form_tags,
        "quarters": quarters,
        "player_sim": player_sim,
        "sims": {
            "home_pts_p10": round(sims["home_pts_p10"], 1),
            "home_pts_p90": round(sims["home_pts_p90"], 1),
            "away_pts_p10": round(sims["away_pts_p10"], 1),
            "away_pts_p90": round(sims["away_pts_p90"], 1),
            "total_p10": round(sims["total_p10"], 1),
            "total_p90": round(sims["total_p90"], 1),
            "margin_std": round(sims["margin_std"], 2),
        },
        "notes": {
            "injuries": injury_notes[:6],
            "rest_travel": flags,
            "play_types": meta["matchup_notes"],
            "form": list(meta.get("form_notes") or [])
            + list(meta.get("player_form_notes") or [])[:4],
            "quarters": list(meta.get("quarter_notes") or [])[:3],
        },
        "components": {
            "home_off": round(meta["h_off"], 2),
            "home_def": round(meta["h_def"], 2),
            "away_off": round(meta["a_off"], 2),
            "away_def": round(meta["a_def"], 2),
            "matchup_residual": round(meta["matchup_edge"], 2),
            "home_player_impact": round(meta["h_pi"], 3),
            "away_player_impact": round(meta["a_pi"], 3),
            "sample_weight": round(meta["w_sample"], 3),
            "home_net": round(meta["h_off"] - meta["h_def"], 2),
            "away_net": round(meta["a_off"] - meta["a_def"], 2),
            "home_form_edge": round(_finite((meta.get("h_form") or {}).get("form_edge")), 2),
            "away_form_edge": round(_finite((meta.get("a_form") or {}).get("form_edge")), 2),
        },
    }


def predict_slate(
    features: dict[str, Any],
    *,
    sim_count: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from data.learning_store import apply_learning_knobs

        apply_learning_knobs()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Learning knobs not applied in predict_slate: %s", exc)
    games = features.get("games") or []
    preds = []
    for g in games:
        try:
            preds.append(project_game(g, features, sim_count=sim_count))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prediction failed for %s: %s", g.get("game_id"), exc)
    return preds
