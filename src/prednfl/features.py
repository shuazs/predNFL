from __future__ import annotations

import warnings

import pandas as pd

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "game_date",
}

SCHEDULE_DATE_CANDIDATES = ("game_date", "gameday", "date")

TEAM_COLUMN_CANDIDATES = ("team", "posteam", "team_abbr")
SEASON_COLUMN_CANDIDATES = ("season", "year")
WEEK_COLUMN_CANDIDATES = ("week", "week_num", "week_number")
DATE_COLUMN_CANDIDATES = ("game_date", "gameday", "date")


def _normalize_schedule_columns(schedules: pd.DataFrame) -> pd.DataFrame:
    schedules = schedules.copy()
    for candidate in SCHEDULE_DATE_CANDIDATES:
        if candidate in schedules.columns:
            if candidate != "game_date":
                schedules.rename(columns={candidate: "game_date"}, inplace=True)
            break
    return schedules


def _validate_schedule_columns(schedules: pd.DataFrame) -> None:
    missing = REQUIRED_SCHEDULE_COLUMNS - set(schedules.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Schedules missing required columns: {missing_list}")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    candidates_list = ", ".join(candidates)
    raise ValueError(f"Team stats missing {label} column. Tried: {candidates_list}")


def _resolve_optional_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _prior_rolling_mean(grouped: "pd.core.groupby.SeriesGroupBy", window: int) -> pd.Series:
    """Rolling mean over each team's previous `window` games, excluding the current one.

    The shift has to happen *inside* each group. Calling `.shift(1)` on the
    result of `grouped.rolling(...)` shifts the concatenated series, which
    leaks each team's last value into the next team's first row.
    """
    return grouped.transform(lambda s: s.rolling(window).mean().shift(1))


def _prepare_team_stats(
    team_stats: pd.DataFrame,
    rolling_window: int,
    metrics: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    team_stats = team_stats.copy()

    team_col = _resolve_column(team_stats, TEAM_COLUMN_CANDIDATES, "team")
    season_col = _resolve_column(team_stats, SEASON_COLUMN_CANDIDATES, "season")
    week_col = _resolve_column(team_stats, WEEK_COLUMN_CANDIDATES, "week")
    date_col = _resolve_optional_column(team_stats, DATE_COLUMN_CANDIDATES)

    rename_map = {team_col: "team", season_col: "season", week_col: "week"}
    if date_col and date_col != "game_date":
        rename_map[date_col] = "game_date"
    team_stats.rename(columns=rename_map, inplace=True)

    if "game_date" in team_stats.columns:
        team_stats["game_date"] = pd.to_datetime(team_stats["game_date"], errors="coerce")
        sort_cols = ["team", "season", "week", "game_date"]
    else:
        sort_cols = ["team", "season", "week"]

    team_stats.sort_values(sort_cols, inplace=True)

    if metrics:
        selected_metrics = [metric for metric in metrics if metric in team_stats.columns]
    else:
        selected_metrics = []

    if not selected_metrics:
        available = ", ".join(team_stats.columns)
        warnings.warn(
            "No configured team stats metrics found. "
            f"Available columns: {available}. "
            "Update team_stats_metrics in config to match your data.",
            stacklevel=2,
        )
        return team_stats, [], []

    group = team_stats.groupby("team", sort=False)
    rolled_columns = []
    for metric in selected_metrics:
        roll_name = f"{metric}_roll"
        team_stats[roll_name] = _prior_rolling_mean(group[metric], rolling_window)
        rolled_columns.append(roll_name)

    join_cols = ["game_id"] if "game_id" in team_stats.columns else ["season", "week"]
    return team_stats, rolled_columns, join_cols


def _build_team_game_frame(schedules: pd.DataFrame) -> pd.DataFrame:
    home = schedules[[
        "game_id",
        "season",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]].copy()
    home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        },
        inplace=True,
    )
    home["is_home"] = 1

    away = schedules[[
        "game_id",
        "season",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]].copy()
    away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        },
        inplace=True,
    )
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"], errors="coerce")
    return team_games


def _moneyline_to_prob(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if value == 0:
        return float("nan")
    if value > 0:
        return 100.0 / (value + 100.0)
    return (-value) / ((-value) + 100.0)


def build_features(
    schedules: pd.DataFrame,
    rolling_window: int,
    team_stats: pd.DataFrame | None = None,
    team_stats_metrics: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create point-in-time features for each game.

    Returns a game-level dataframe with rolling stats for home/away teams
    computed using only prior games.
    """
    schedules = _normalize_schedule_columns(schedules)
    _validate_schedule_columns(schedules)

    schedules = schedules.copy()
    schedules["game_date"] = pd.to_datetime(schedules["game_date"], errors="coerce")
    schedules.sort_values(["season", "week", "game_date", "game_id"], inplace=True)

    team_games = _build_team_game_frame(schedules)
    team_games.sort_values(["team", "season", "week", "game_date"], inplace=True)

    group = team_games.groupby("team", sort=False)
    team_games["pf_roll"] = _prior_rolling_mean(group["points_for"], rolling_window)
    team_games["pa_roll"] = _prior_rolling_mean(group["points_against"], rolling_window)
    team_games["last_game_date"] = group["game_date"].shift(1)
    team_games["rest_days"] = (
        team_games["game_date"] - team_games["last_game_date"]
    ).dt.days

    feature_cols = ["pf_roll", "pa_roll", "rest_days"]

    home_features = team_games[team_games["is_home"] == 1][
        ["game_id", "team"] + feature_cols
    ].copy()
    home_features.columns = [
        "game_id",
        "home_team",
        "home_pf_roll",
        "home_pa_roll",
        "home_rest_days",
    ]

    away_features = team_games[team_games["is_home"] == 0][
        ["game_id", "team"] + feature_cols
    ].copy()
    away_features.columns = [
        "game_id",
        "away_team",
        "away_pf_roll",
        "away_pa_roll",
        "away_rest_days",
    ]

    features = schedules.merge(home_features, on=["game_id", "home_team"], how="left")
    features = features.merge(away_features, on=["game_id", "away_team"], how="left")

    team_stat_columns: list[str] = []
    team_stat_diff_columns: list[str] = []
    if team_stats is not None:
        prepared_stats, rolled_columns, join_cols = _prepare_team_stats(
            team_stats, rolling_window=rolling_window, metrics=team_stats_metrics
        )
        if rolled_columns:
            home_stats = prepared_stats[join_cols + ["team"] + rolled_columns].copy()
            home_stats.rename(columns={"team": "home_team"}, inplace=True)
            for column in rolled_columns:
                home_stats.rename(columns={column: f"home_ts_{column}"}, inplace=True)

            away_stats = prepared_stats[join_cols + ["team"] + rolled_columns].copy()
            away_stats.rename(columns={"team": "away_team"}, inplace=True)
            for column in rolled_columns:
                away_stats.rename(columns={column: f"away_ts_{column}"}, inplace=True)

            features = features.merge(home_stats, on=join_cols + ["home_team"], how="left")
            features = features.merge(away_stats, on=join_cols + ["away_team"], how="left")

            team_stat_columns = [
                f"home_ts_{column}" for column in rolled_columns
            ] + [
                f"away_ts_{column}" for column in rolled_columns
            ]

            for column in rolled_columns:
                diff_name = f"ts_{column}_diff"
                features[diff_name] = (
                    features[f"home_ts_{column}"] - features[f"away_ts_{column}"]
                )
                team_stat_diff_columns.append(diff_name)

    features["home_win"] = (features["home_score"] > features["away_score"]).astype(float)
    features.loc[features["home_score"].isna() | features["away_score"].isna(), "home_win"] = float("nan")

    base_feature_columns = [
        "home_pf_roll",
        "home_pa_roll",
        "home_rest_days",
        "away_pf_roll",
        "away_pa_roll",
        "away_rest_days",
    ]

    if "spread_line" in features.columns:
        base_feature_columns.append("spread_line")

    if "home_moneyline" in features.columns and "away_moneyline" in features.columns:
        features["home_ml_prob"] = features["home_moneyline"].apply(_moneyline_to_prob)
        features["away_ml_prob"] = features["away_moneyline"].apply(_moneyline_to_prob)
        features["ml_prob_diff"] = features["home_ml_prob"] - features["away_ml_prob"]
        base_feature_columns.extend(["home_ml_prob", "away_ml_prob", "ml_prob_diff"])

    if "home_rest" in features.columns:
        features["home_rest_sched"] = features["home_rest"]
        base_feature_columns.append("home_rest_sched")
    if "away_rest" in features.columns:
        features["away_rest_sched"] = features["away_rest"]
        base_feature_columns.append("away_rest_sched")

    diff_feature_columns = []
    if "home_pf_roll" in features.columns and "away_pf_roll" in features.columns:
        features["pf_roll_diff"] = features["home_pf_roll"] - features["away_pf_roll"]
        diff_feature_columns.append("pf_roll_diff")
    if "home_pa_roll" in features.columns and "away_pa_roll" in features.columns:
        features["pa_roll_diff"] = features["home_pa_roll"] - features["away_pa_roll"]
        diff_feature_columns.append("pa_roll_diff")
    if "home_rest_days" in features.columns and "away_rest_days" in features.columns:
        features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
        diff_feature_columns.append("rest_days_diff")
    if "home_rest_sched" in features.columns and "away_rest_sched" in features.columns:
        features["rest_sched_diff"] = features["home_rest_sched"] - features["away_rest_sched"]
        diff_feature_columns.append("rest_sched_diff")

    return (
        features,
        base_feature_columns + diff_feature_columns + team_stat_diff_columns,
    )
