from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prednfl.features import _moneyline_to_prob, build_features

from conftest import TEAMS, make_schedule, make_team_stats, team_games


def _row(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.set_index("game_id").loc[game_id]


def _team_feature(features: pd.DataFrame, team: str, column: str) -> pd.Series:
    """Per-game value of home_<column>/away_<column> for `team`, chronological."""
    rows = features[(features["home_team"] == team) | (features["away_team"] == team)]
    rows = rows.sort_values(["season", "week", "game_date"])
    values = np.where(rows["home_team"] == team, rows[f"home_{column}"], rows[f"away_{column}"])
    return pd.Series(values, index=rows["game_id"].to_numpy(), dtype=float)


# --------------------------------------------------------------------------
# Moneyline conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "moneyline, expected",
    [(100, 0.5), (-100, 0.5), (-150, 0.6), (150, 0.4), (-300, 0.75), (300, 0.25), ("-150", 0.6)],
)
def test_moneyline_to_prob(moneyline, expected):
    assert _moneyline_to_prob(moneyline) == pytest.approx(expected)


@pytest.mark.parametrize("moneyline", [None, float("nan"), 0, "abc"])
def test_moneyline_to_prob_invalid_returns_nan(moneyline):
    assert math.isnan(_moneyline_to_prob(moneyline))


# --------------------------------------------------------------------------
# Rolling point-in-time features
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 2, 3])
def test_rolling_features_match_manual_per_team(schedule, window):
    features, _ = build_features(schedule, rolling_window=window)

    for team in TEAMS:
        games = team_games(schedule, team)
        expected_pf = games["pf"].rolling(window).mean().shift(1).to_numpy()
        expected_pa = games["pa"].rolling(window).mean().shift(1).to_numpy()

        actual_pf = _team_feature(features, team, "pf_roll").to_numpy()
        actual_pa = _team_feature(features, team, "pa_roll").to_numpy()

        np.testing.assert_allclose(actual_pf, expected_pf, equal_nan=True, err_msg=f"{team} pf_roll")
        np.testing.assert_allclose(actual_pa, expected_pa, equal_nan=True, err_msg=f"{team} pa_roll")


@pytest.mark.parametrize("window", [1, 2, 3])
def test_first_game_has_no_rolling_history(schedule, window):
    """Regression: shift(1) used to run across team boundaries, so each team's
    first game inherited the previous team's latest rolling value."""
    features, _ = build_features(schedule, rolling_window=window)

    for team in TEAMS:
        for column in ("pf_roll", "pa_roll"):
            values = _team_feature(features, team, column)
            assert values.iloc[:window].isna().all(), (
                f"{team} {column}: first {window} game(s) should have no history, got "
                f"{values.iloc[:window].tolist()}"
            )


def test_features_do_not_depend_on_current_or_future_scores(schedule):
    """Changing a game's result must not change features for that game or any
    earlier game."""
    base, columns = build_features(schedule, rolling_window=2)

    target = schedule.sort_values(["season", "week"]).iloc[len(schedule) // 2]
    perturbed = schedule.copy()
    mask = perturbed["game_id"] == target["game_id"]
    perturbed.loc[mask, "home_score"] += 50
    perturbed.loc[mask, "away_score"] += 17

    changed, _ = build_features(perturbed, rolling_window=2)

    cutoff = (base["season"] < target["season"]) | (
        (base["season"] == target["season"]) & (base["week"] <= target["week"])
    )
    pd.testing.assert_frame_equal(
        base.loc[cutoff, columns].reset_index(drop=True),
        changed.loc[cutoff, columns].reset_index(drop=True),
    )
    # ...but later games for the affected teams do change.
    later = ~cutoff
    assert not base.loc[later, columns].equals(changed.loc[later, columns])


def test_rest_days_are_per_team(schedule):
    features, _ = build_features(schedule, rolling_window=2)
    for team in TEAMS:
        games = team_games(schedule, team)
        expected = games["game_date"].diff().dt.days.to_numpy()
        actual = _team_feature(features, team, "rest_days").to_numpy()
        np.testing.assert_allclose(actual, expected, equal_nan=True, err_msg=team)


def test_diff_columns_are_home_minus_away(schedule):
    features, columns = build_features(schedule, rolling_window=2)
    for stat in ("pf_roll", "pa_roll", "rest_days"):
        assert f"{stat}_diff" in columns
        expected = features[f"home_{stat}"] - features[f"away_{stat}"]
        pd.testing.assert_series_equal(features[f"{stat}_diff"], expected, check_names=False)


def test_home_win_label(schedule):
    schedule = schedule.copy()
    schedule.loc[0, ["home_score", "away_score"]] = [24, 17]
    schedule.loc[1, ["home_score", "away_score"]] = [10, 31]
    schedule.loc[2, ["home_score", "away_score"]] = [20, 20]  # tie -> not a home win
    features, _ = build_features(schedule, rolling_window=2)

    assert _row(features, schedule.loc[0, "game_id"])["home_win"] == 1
    assert _row(features, schedule.loc[1, "game_id"])["home_win"] == 0
    assert _row(features, schedule.loc[2, "game_id"])["home_win"] == 0


def test_home_win_is_nan_for_unplayed_games(schedule):
    schedule = schedule.copy()
    schedule.loc[0, ["home_score", "away_score"]] = [float("nan"), 17]
    schedule.loc[1, ["home_score", "away_score"]] = [24, float("nan")]
    schedule.loc[2, ["home_score", "away_score"]] = [float("nan"), float("nan")]
    features, _ = build_features(schedule, rolling_window=2)

    assert np.isnan(_row(features, schedule.loc[0, "game_id"])["home_win"])
    assert np.isnan(_row(features, schedule.loc[1, "game_id"])["home_win"])
    assert np.isnan(_row(features, schedule.loc[2, "game_id"])["home_win"])


def test_output_has_one_row_per_game(schedule):
    features, _ = build_features(schedule, rolling_window=2)
    assert len(features) == len(schedule)
    assert features["game_id"].is_unique


# --------------------------------------------------------------------------
# Schema handling and optional columns
# --------------------------------------------------------------------------

def test_gameday_column_is_renamed(schedule):
    features, _ = build_features(schedule.rename(columns={"game_date": "gameday"}), rolling_window=2)
    assert "game_date" in features.columns


def test_missing_required_columns_raise(schedule):
    with pytest.raises(ValueError, match="home_score"):
        build_features(schedule.drop(columns=["home_score"]), rolling_window=2)


def test_base_feature_columns_without_market_data(schedule):
    _, columns = build_features(schedule, rolling_window=2)
    assert columns == [
        "home_pf_roll",
        "home_pa_roll",
        "home_rest_days",
        "away_pf_roll",
        "away_pa_roll",
        "away_rest_days",
        "pf_roll_diff",
        "pa_roll_diff",
        "rest_days_diff",
    ]


def test_market_and_schedule_rest_columns_are_used_when_present():
    schedule = make_schedule(
        spread_line=-3.0, home_moneyline=-150, away_moneyline=130, home_rest=7, away_rest=6
    )
    features, columns = build_features(schedule, rolling_window=2)

    for name in (
        "spread_line",
        "home_ml_prob",
        "away_ml_prob",
        "ml_prob_diff",
        "home_rest_sched",
        "away_rest_sched",
        "rest_sched_diff",
    ):
        assert name in columns

    assert features["home_ml_prob"].iloc[0] == pytest.approx(0.6)
    assert features["away_ml_prob"].iloc[0] == pytest.approx(100 / 230)
    assert features["rest_sched_diff"].iloc[0] == 1


# --------------------------------------------------------------------------
# External team stats
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 2])
def test_team_stats_rolling_is_point_in_time_per_team(schedule, window):
    stats = make_team_stats(schedule)
    features, columns = build_features(
        schedule, rolling_window=window, team_stats=stats, team_stats_metrics=("passing_epa",)
    )
    assert "ts_passing_epa_roll_diff" in columns
    # Raw home/away team-stat columns are built but only the diff is a model input.
    assert "home_ts_passing_epa_roll" not in columns

    for team in TEAMS:
        team_stats = stats[stats["team"] == team].sort_values(["season", "week"])
        expected = team_stats["passing_epa"].rolling(window).mean().shift(1).to_numpy()
        actual = _team_feature(features, team, "ts_passing_epa_roll").to_numpy()
        np.testing.assert_allclose(actual, expected, equal_nan=True, err_msg=team)
        # Regression: first game(s) must not inherit another team's stats.
        assert np.isnan(actual[:window]).all(), team

    expected_diff = features["home_ts_passing_epa_roll"] - features["away_ts_passing_epa_roll"]
    pd.testing.assert_series_equal(
        features["ts_passing_epa_roll_diff"], expected_diff, check_names=False
    )


def test_team_stats_column_aliases_are_resolved(schedule):
    stats = make_team_stats(schedule).rename(columns={"team": "team_abbr", "season": "year"})
    _, columns = build_features(
        schedule, rolling_window=2, team_stats=stats, team_stats_metrics=("passing_epa",)
    )
    assert "ts_passing_epa_roll_diff" in columns


def test_team_stats_missing_team_column_raises(schedule):
    stats = make_team_stats(schedule).drop(columns=["team"])
    with pytest.raises(ValueError, match="team"):
        build_features(schedule, rolling_window=2, team_stats=stats, team_stats_metrics=("passing_epa",))


def test_team_stats_with_no_matching_metrics_warns_and_is_skipped(schedule):
    stats = make_team_stats(schedule)
    with pytest.warns(UserWarning, match="No configured team stats metrics"):
        features, columns = build_features(
            schedule, rolling_window=2, team_stats=stats, team_stats_metrics=("not_a_metric",)
        )
    assert not any(column.startswith("ts_") for column in columns)
    assert len(features) == len(schedule)
