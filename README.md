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

# Desktop UI (stdlib tkinter)
python gui.py
python main.py --gui
```

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
data/static_arenas.py  arena lat/lon + TZ for travel
model/predict.py       scores, margins, win probs
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

Win probability uses a logistic of expected margin (σ ≈ 12.5).

## Graceful degradation

| Source fails | Behavior |
| --- | --- |
| `stats.nba.com` / nba_api | Auto-detect; ESPN scoreboard, schedules, team/player stats |
| Scoreboard | Empty slate message — try another `--date` |
| Advanced / Synergy | Game-log / ESPN season proxies |
| EPM scrapes | dunksandthrees → nbarapm → darko → PTS z-score proxy |
| Injuries | ESPN table (official.nba.com tried first) |

Set `SKIP_NBA_API=1` to force ESPN-only mode (faster on blocked networks).

## Disclaimer

Research / entertainment projections only. Not advice of any kind.
