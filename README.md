# prednfl

A lightweight CLI for backtesting and predicting NFL game winners using `nflreadpy` data and a logistic regression model.

## Features
- Walk-forward backtesting (week-by-week)
- Predict a single matchup (auto-detects next game between two teams)
- Market-aware features (spread line + moneyline implied probabilities when available)
- Leakage-safe rolling features

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> Note: `nflreadpy` downloads data from the internet on first run.

## Quick start

Run a backtest on the default seasons (2021–2024):

```bash
prednfl backtest
```

Predict a matchup by explicitly setting home/away:

```bash
prednfl predict --season 2025 --week 21 --home DEN --away NE
```

Predict the next matchup between two teams in a season:

```bash
prednfl predict --season 2025 --teams NE DEN
```

If the matchup is early season and some rolling features are missing, allow mean-fill:

```bash
prednfl predict --season 2025 --teams NE DEN --allow-missing
```

## CLI reference

### `prednfl backtest`
Options:
- `--seasons` Season list or range (e.g. `2021-2024` or `2021,2022,2023,2024`)
- `--rolling-window` Rolling games window (default: 6)
- `--min-games` Minimum games before training (default: 6)
- `--c` Logistic regression C value (default: 0.05)

### `prednfl predict`
Options:
- `--season` Season year (required)
- `--week` Week number (required unless using `--teams` or `--game-id`)
- `--home` Home team (required unless using `--teams` or `--game-id`)
- `--away` Away team (required unless using `--teams` or `--game-id`)
- `--teams` Two teams to auto-detect the next matchup in that season
- `--game-id` Explicit game id (overrides other selectors)
- `--allow-missing` Fill missing matchup features from training means

## Notes
- Team abbreviations must match `nflreadpy` schedule abbreviations.
- The model retrains using all prior games before the selected week.
- Default settings live in `src/prednfl/config.py`.

## Development

Run the tests (no network needed; data loading is stubbed):

```bash
pip install -e ".[dev]"
pytest
```

Run the legacy scripts (thin wrappers around the CLI):

```bash
python scripts/run_backtest.py
python scripts/predict_matchup.py --season 2025 --teams NE DEN
```
