# NBA Picker

Free, on-demand NBA game predictor and wager recommender. Pulls public stats (nba_api), injury reports, player-impact tables (EPM/RAPM scrapes with fallbacks), and optional free-tier odds. No paid data feeds required for projections.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional odds keys
cp .env.example .env
# edit .env → ODDS_API_KEY=... and/or THERUNDOWN_API_KEY=...

python main.py
python main.py --date 2026-10-27
python main.py --force-refresh

# Desktop UI (stdlib tkinter)
python gui.py
python main.py --gui
```

### Windows `.exe` (build on Windows only)

```bat
build_windows.bat
```

Produces `dist\NBA-Picker.exe` (windowed). Needs PyInstaller via `requirements-build.txt`. A Linux/mac build is **not** an `.exe`.

Outputs:

- Terminal summary for the slate
- `output/today_predictions.md` and `output/today_predictions.json`
- Dated copies under `output/predictions_YYYY-MM-DD.*`
- Local cache under `cache/` (reused if newer than 6 hours)

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `ODDS_API_KEY` | No | [The Odds API](https://the-odds-api.com/) free tier — ML / spreads / totals |
| `THERUNDOWN_API_KEY` | No | TheRundown (RapidAPI) free tier fallback |
| `CACHE_MAX_AGE_HOURS` | No | Default `6` |
| `NBA_SEASON_YEAR` | No | Override season start year (e.g. `2025`) |
| `MIN_EDGE_PCT` | No | Default `3` — skip thinner edges |
| `KELLY_FRACTION` | No | Default `0.25` (¼ Kelly) |
| `BANKROLL_UNITS` | No | Default `100` for stake display |

Without an odds key the model still prints pure projections; the Top 5 wager section stays empty.

## Architecture

```
data/fetch.py          free sources + cache
data/process.py        EWM ratings, injuries, rest/travel
data/static_arenas.py  arena lat/lon + TZ for travel
model/predict.py       scores, margins, win probs
model/wagers.py        EV vs books, ¼ Kelly stakes
main.py                single entry point
config.py              knobs + paths
```

### Model sketch

1. **Team layer** — EWM (span ~15) Net Rating, Pace, Four Factors from nba_api game logs; blend prior season until ~10 games.
2. **Player layer** — recent minutes × injury status; minutes-weighted EPM/RAPM (dunksandthrees → nbarapm → darko → nba_api z-score proxy).
3. **Matchup residual** — small, shrunk play-type PPP edges (Synergy when available).
4. **Context** — rest / B2B / travel miles / timezone + home court.
5. **Injuries first** — official/ESPN status drives minutes → rating change before other layers.

Win probability uses a logistic of expected margin (σ ≈ 12.5). Wagers need model probability − vig-removed implied ≥ `MIN_EDGE_PCT`.

## Graceful degradation

| Source fails | Behavior |
| --- | --- |
| `stats.nba.com` / nba_api | Auto-detect; ESPN scoreboard, schedules, team/player stats |
| Scoreboard | Empty slate message — try another `--date` |
| Advanced / Synergy | Game-log / ESPN season proxies |
| EPM scrapes | dunksandthrees → nbarapm → darko → PTS z-score proxy |
| Injuries | ESPN table (official.nba.com tried first) |
| Odds APIs | Projections only; no wager ranks |

Set `SKIP_NBA_API=1` to force ESPN-only mode (faster on blocked networks).

## Disclaimer

For research / entertainment. Sports betting involves risk. This is **not** financial advice.
