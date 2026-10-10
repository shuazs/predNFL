from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

TEAMS = ("AAA", "BBB", "CCC", "DDD")

# Rotating pairings for 4 teams: every team plays once per week.
_PAIRINGS = (
    (("AAA", "BBB"), ("CCC", "DDD")),
    (("AAA", "CCC"), ("BBB", "DDD")),
    (("AAA", "DDD"), ("BBB", "CCC")),
)


def make_schedule(
    seasons: tuple[int, ...] = (2021,),
    n_weeks: int = 6,
    seed: int = 0,
    **extra_columns,
) -> pd.DataFrame:
    """Build a small, deterministic nflreadpy-style schedule.

    Every team plays once per week, games are 7 days apart, and home/away
    flips each cycle so every team has both home and away games.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-09-10")
        for week in range(1, n_weeks + 1):
            pairing = _PAIRINGS[(week - 1) % len(_PAIRINGS)]
            flip = ((week - 1) // len(_PAIRINGS)) % 2 == 1
            for idx, (a, b) in enumerate(pairing):
                home, away = (b, a) if flip else (a, b)
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "season": season,
                        "week": week,
                        "game_date": (start + pd.Timedelta(days=7 * (week - 1))).strftime("%Y-%m-%d"),
                        "home_team": home,
                        "away_team": away,
                        "home_score": int(rng.integers(0, 45)),
                        "away_score": int(rng.integers(0, 45)),
                    }
                )
    schedule = pd.DataFrame(rows)
    for name, value in extra_columns.items():
        schedule[name] = value
    return schedule


def team_games(schedule: pd.DataFrame, team: str) -> pd.DataFrame:
    """A team's games in chronological order, with points for/against."""
    games = schedule[(schedule["home_team"] == team) | (schedule["away_team"] == team)].copy()
    is_home = games["home_team"] == team
    games["pf"] = np.where(is_home, games["home_score"], games["away_score"])
    games["pa"] = np.where(is_home, games["away_score"], games["home_score"])
    games["is_home"] = is_home
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games.sort_values(["season", "week", "game_date"]).reset_index(drop=True)


def make_team_stats(schedule: pd.DataFrame, metric: str = "passing_epa", seed: int = 1) -> pd.DataFrame:
    """Weekly team-stats rows (one per team-game), like nflreadpy.load_team_stats."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in schedule.itertuples(index=False):
        for team in (row.home_team, row.away_team):
            rows.append(
                {
                    "team": team,
                    "season": row.season,
                    "week": row.week,
                    metric: float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def schedule() -> pd.DataFrame:
    return make_schedule(seasons=(2021, 2022), n_weeks=6)
