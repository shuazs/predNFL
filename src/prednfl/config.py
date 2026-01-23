from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)
    rolling_window_games: int = 6
    min_games_before_train: int = 6
    random_state: int = 7
    logreg_c: float = 0.05
    team_stats_metrics: tuple[str, ...] = (
        "passing_epa",
        "rushing_epa",
        "receiving_epa",
        "def_sacks",
        "def_interceptions",
        "def_tackles_for_loss",
        "penalty_yards",
        "fg_pct",
    )


DEFAULT_CONFIG = BacktestConfig()
