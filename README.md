# NBA Picker

Free, on-demand NBA game predictor for **local personal use**. Pulls public stats (nba_api), injury reports, and player-impact tables (EPM/RAPM scrapes with fallbacks). No paid data feeds and no sportsbook odds.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional overrides
cp .env.example .env

python main.py
python main.py --date 2026-10-27
python main.py --force-refresh

# Desktop UI (stdlib tkinter) — Away vs Home + Date
python gui.py
python main.py --gui
```

The GUI asks for **Away**, **Home**, and **Date**, then runs that scheduled matchup and shows an **AI paste** block (win %, avg combined score, top-3 PTS/REB/AST/STL/BLK). If the matchup is not on that day’s schedule, it tells you clearly and does not invent a custom game.

### Windows `.exe`

Prebuilt zip: GitHub Release [`windows-exe`](https://github.com/Josiahjc01/NBA-Picker/releases/tag/windows-exe), or rebuild on Windows:

```bat
build_windows.bat
```

Produces `dist\NBA-Picker.exe` (windowed). Needs PyInstaller via `requirements-build.txt`.

Outputs:

- Terminal / GUI summary for the slate
- `output/today_predictions.md` and `output/today_predictions.json`
- Dated copies under `output/predictions_YYYY-MM-DD.*`
- Local cache under `cache/` (reused if newer than 6 hours)
- Offline learning under `cache/learning/` (paste grades + capped knobs)

### Offline learning (Calibration tab)

Paste a `real_box` / `ai_comparison` JSON (or ASCII `DATE` / `AWAY @ HOME` / `ACT` / `PRED`) into the **Calibration** tab. Grades use projection **means/win% only** — never Highlights sim-maxes. Safe knobs (`SCORE_SIGMA`, `MARGIN_SIGMA`, home court, rest/B2B/altitude) auto-step ≤10% toward targets after ~20 graded games, with hard caps and `knob_history.jsonl` audit. `suggested_code_themes` go to `suggestions_inbox/` only (never auto code).

```bash
python main.py --import-paste path/to/paste.json
python main.py --learning
python main.py --apply-knobs
python main.py --reset-knobs
python main.py --grade
python main.py --calibration
```

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `CACHE_MAX_AGE_HOURS` | No | Default `6` |
| `NBA_SEASON_YEAR` | No | Override season start year (e.g. `2025`) |
| `SKIP_NBA_API` | No | Set `1` to skip `stats.nba.com` and use ESPN only |

## Architecture

```
data/fetch.py          free sources + cache
data/process.py        EWM ratings, injuries, rest/travel
data/learning_store.py offline paste grades + capped knobs
data/static_arenas.py  arena lat/lon + TZ for travel
model/predict.py       scores, margins, win probs
model/calibrate.py     ESPN grade + calibration report
main.py                CLI entry point
gui.py                 desktop UI
config.py              knobs + paths
```

### Model sketch

1. **Team layer** — EWM (span ~15) Net Rating, Pace, Four Factors from nba_api game logs; blend prior season until ~10 games.
2. **Player layer** — recent minutes × injury status; minutes-weighted EPM/RAPM (dunksandthrees → nbarapm → darko → nba_api z-score proxy).
3. **Matchup residual** — small, shrunk play-type PPP edges (Synergy when available).
4. **Context** — rest / B2B / travel miles / timezone + home court.
5. **Injuries first** — official/ESPN status drives minutes → rating change before other layers.

Win probability comes from **10,000** Monte Carlo score simulations (configurable via `SIM_COUNT`) around mean projections. Team priors stay strong early (`PRIOR_BLEND_GAMES`, default 25); injuries adjust first; absolute player-impact is shrunk so tiny samples do not force coin-flips.

### Rosters / injuries / minutes

- **Rosters:** ESPN team roster endpoints are the membership source of truth (trades, two-ways).
- **Stats:** Season per-game lines (nba_api when reachable, else ESPN) are **remapped** onto current rosters; stale off-roster rows are dropped; roster-only players are kept for depth/injury.
- **Injuries:** ESPN injuries API (HTML scrape fallback). Status → minutes factor (Out=0, Doubtful≈0.15, Questionable≈0.55, Probable≈0.9). Junk official.nba.com landing text is ignored.
- **Verification:** `python main.py --away BOS --home NYK --date YYYY-MM-DD` prints roster size, key minutes, and injury flags for that matchup.

## Graceful degradation

| Source fails | Behavior |
| --- | --- |
| `stats.nba.com` / nba_api | Auto-detect; ESPN scoreboard, schedules, team/player stats |
| Scoreboard | Empty slate message — try another `--date` |
| Advanced / Synergy | Game-log / ESPN season proxies |
| EPM scrapes | dunksandthrees → nbarapm → darko → PTS z-score proxy |
| Injuries | ESPN API → ESPN HTML; empty if both fail |

Set `SKIP_NBA_API=1` to force ESPN-only mode (faster on blocked networks).

## Disclaimer

Research / entertainment projections only. Not advice of any kind.
