from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prednfl.backtest as backtest_module
from prednfl.backtest import Backtester, _prior_season_probs, _team_season_win_pct
from prednfl.features import build_features
from prednfl.metrics import evaluate

from conftest import make_schedule


def _keyed_features(schedule: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Features where one column encodes (season, week) so a spy model can see
    exactly which weeks it was trained and tested on."""
    features = schedule.copy()
    features["game_date"] = pd.to_datetime(features["game_date"])
    
    # Respect unplayed games: NaN scores -> NaN win
    features["home_win"] = (features["home_score"] > features["away_score"]).astype(float)
    features.loc[features["home_score"].isna() | features["away_score"].isna(), "home_win"] = np.nan
    
    features["week_key"] = features["season"] * 100 + features["week"]
    features["noise"] = np.random.default_rng(0).normal(size=len(features))
    return features, ["week_key", "noise"]


class _SpyModel:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __init__(self, random_state: int, c_value: float) -> None:
        self.train_keys = None

    def fit(self, X, y):
        self.train_keys = X[:, 0].copy()

    def predict_proba(self, X):
        _SpyModel.calls.append((self.train_keys, X[:, 0].copy()))
        return np.full(len(X), 0.5)


@pytest.fixture
def spy(monkeypatch):
    _SpyModel.calls = []
    monkeypatch.setattr(backtest_module, "LogisticRegressionModel", _SpyModel)
    return _SpyModel


def test_walk_forward_never_trains_on_test_week_or_later(spy):
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=6)
    features, columns = _keyed_features(schedule)

    Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)

    assert spy.calls, "model was never trained"
    for train_keys, test_keys in spy.calls:
        assert len(set(test_keys)) == 1, "each split should test exactly one (season, week)"
        assert train_keys.max() < test_keys.min()


def test_walk_forward_trains_on_all_prior_games(spy):
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=6)
    features, columns = _keyed_features(schedule)

    Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)

    for train_keys, test_keys in spy.calls:
        test_key = test_keys[0]
        expected = (features["week_key"] < test_key).sum()
        assert len(train_keys) == expected


def test_min_games_before_train_skips_early_weeks(spy):
    schedule = make_schedule(seasons=(2021,), n_weeks=6)  # 2 games per week
    features, columns = _keyed_features(schedule)

    result = Backtester(min_games_before_train=6, random_state=7, logreg_c=1.0).run(features, columns)

    # Weeks 1-3 have <6 prior games; first prediction is week 4.
    assert sorted(result.predictions["week"].unique()) == [4, 5, 6]


def test_no_splits_returns_empty_predictions_and_nan_metrics():
    schedule = make_schedule(seasons=(2021,), n_weeks=3)
    features, columns = _keyed_features(schedule)

    result = Backtester(min_games_before_train=10_000, random_state=7, logreg_c=1.0).run(
        features, columns
    )

    assert result.predictions.empty
    expected = {
        "model_accuracy",
        "model_log_loss",
        "model_brier",
        "baseline_home_rate_accuracy",
        "baseline_home_rate_log_loss",
        "baseline_home_rate_brier",
        "baseline_prior_season_accuracy",
        "baseline_prior_season_log_loss",
        "baseline_prior_season_brier",
    }
    assert set(result.metrics.keys()) == expected
    for val in result.metrics.values():
        assert np.isnan(val)


def test_end_to_end_with_real_features_and_model():
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=12, seed=3)
    features, columns = build_features(schedule, rolling_window=2)

    result = Backtester(min_games_before_train=8, random_state=7, logreg_c=0.05).run(features, columns)
    preds = result.predictions

    assert not preds.empty
    assert preds["game_id"].is_unique
    for column in ("home_win_prob", "baseline_home_rate", "baseline_prior_win_pct"):
        assert preds[column].between(0, 1).all(), column
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_summary_metrics_are_unweighted_mean_of_weekly_metrics():
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=12, seed=3)
    features, columns = build_features(schedule, rolling_window=2)
    result = Backtester(min_games_before_train=8, random_state=7, logreg_c=0.05).run(features, columns)

    weekly = [
        evaluate(group["home_win"].to_numpy(), group["home_win_prob"].to_numpy())
        for _, group in result.predictions.groupby(["season", "week"])
    ]
    expected = pd.DataFrame(weekly).mean()
    for metric in ("accuracy", "log_loss", "brier"):
        assert result.metrics[f"model_{metric}"] == pytest.approx(expected[metric])


def test_home_rate_baseline_uses_training_home_win_rate(spy):
    schedule = make_schedule(seasons=(2021,), n_weeks=6, seed=5)
    features, columns = _keyed_features(schedule)

    result = Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)

    for (season, week), group in result.predictions.groupby(["season", "week"]):
        prior = features[(features["season"] < season) | ((features["season"] == season) & (features["week"] < week))]
        np.testing.assert_allclose(group["baseline_home_rate"], prior["home_win"].mean())


# --------------------------------------------------------------------------
# Prior-season baseline helpers
# --------------------------------------------------------------------------

def test_team_season_win_pct():
    games = pd.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "home_team": ["AAA", "BBB", "AAA"],
            "away_team": ["BBB", "AAA", "CCC"],
            "home_score": [21, 10, 3],
            "away_score": [14, 20, 7],
        }
    )
    pct = _team_season_win_pct(games).set_index(["season", "team"])

    assert pct.loc[(2021, "AAA"), "win_pct"] == pytest.approx(2 / 3)
    assert pct.loc[(2021, "AAA"), "games"] == 3
    assert pct.loc[(2021, "BBB"), "win_pct"] == 0.0
    assert pct.loc[(2021, "CCC"), "win_pct"] == 1.0


def test_prior_season_probs_uses_previous_season_only():
    win_pcts = pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022],
            "team": ["AAA", "BBB", "AAA", "BBB"],
            "win_pct": [0.75, 0.25, 0.0, 1.0],
        }
    )
    test = pd.DataFrame({"season": [2022], "home_team": ["AAA"], "away_team": ["BBB"]})

    assert _prior_season_probs(test, win_pcts)[0] == pytest.approx(0.75)


def test_prior_season_probs_fall_back_to_half():
    win_pcts = pd.DataFrame({"season": [2021], "team": ["AAA"], "win_pct": [0.0]})
    test = pd.DataFrame(
        {"season": [2021, 2022, 2022], "home_team": ["AAA", "AAA", "ZZZ"], "away_team": ["BBB", "BBB", "YYY"]}
    )
    probs = _prior_season_probs(test, win_pcts)

    assert probs[0] == 0.5  # no 2020 data at all
    assert probs[1] == 0.0  # AAA was 0.0, BBB missing -> 0.5: 0 / (0 + 0.5)
    assert probs[2] == 0.5  # neither team found


def test_prior_season_probs_both_winless_is_half():
    win_pcts = pd.DataFrame({"season": [2021, 2021], "team": ["AAA", "BBB"], "win_pct": [0.0, 0.0]})
    test = pd.DataFrame({"season": [2022], "home_team": ["AAA"], "away_team": ["BBB"]})
    assert _prior_season_probs(test, win_pcts)[0] == 0.5


def test_market_baseline_is_vig_free_moneyline():
    schedule = make_schedule(seasons=(2021,), n_weeks=6)
    # Ensure variety of outcomes for logistic regression
    schedule.loc[::2, ["home_score", "away_score"]] = [24, 17]
    schedule.loc[1::2, ["home_score", "away_score"]] = [17, 24]

    # Use a constant moneyline for most games to ensure valid features for training
    schedule["home_moneyline"] = -150
    schedule["away_moneyline"] = 130
    
    # Prob(home) = (implied_home) / (implied_home + implied_away)
    # -150: 150/250 = 0.6
    # +130: 100/230 = 0.43478
    expected = 0.6 / (0.6 + 100 / 230)

    # Use window=1 so fewer games are skipped by missing features
    features, columns = build_features(schedule, rolling_window=1)
    result = Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)

    assert not result.predictions.empty
    assert result.predictions["baseline_market"].iloc[0] == pytest.approx(expected)
    assert np.allclose(result.predictions["baseline_market"], expected)
    assert "baseline_market_log_loss" in result.metrics


def test_market_baseline_absent_without_moneyline_columns():
    schedule = make_schedule(seasons=(2021,), n_weeks=6)
    # Ensure variety of outcomes
    schedule.loc[::2, ["home_score", "away_score"]] = [24, 17]
    schedule.loc[1::2, ["home_score", "away_score"]] = [17, 24]

    features, columns = build_features(schedule, rolling_window=2)
    result = Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)

    assert "baseline_market" not in result.predictions.columns
    assert "baseline_market_log_loss" not in result.metrics


def test_pooled_metrics_match_evaluate_over_all_predictions():
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=12, seed=3)
    features, columns = build_features(schedule, rolling_window=2)
    result = Backtester(min_games_before_train=8, random_state=7, logreg_c=0.05).run(features, columns)

    y_true = result.predictions["home_win"].to_numpy()
    y_prob = result.predictions["home_win_prob"].to_numpy()
    expected = evaluate(y_true, y_prob)

    assert result.pooled_metrics["model_accuracy"] == pytest.approx(expected["accuracy"])
    assert result.pooled_metrics["model_log_loss"] == pytest.approx(expected["log_loss"])


def test_coverage_counts_add_up():
    # 2 seasons, 4 weeks each, 2 games per week = 16 games total.
    # Teams are A, B, C, D.
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=4)
    # Ensure variety of outcomes for logistic regression
    schedule.loc[::2, ["home_score", "away_score"]] = [24, 17]
    schedule.loc[1::2, ["home_score", "away_score"]] = [17, 24]

    # 1. Unplayed game (NaN scores)
    schedule.loc[0, ["home_score", "away_score"]] = np.nan
    
    # 2. Missing features: build_features drops first 'rolling_window' games per team.
    # We want some games to have results but miss features.
    # We'll set a feature to NaN manually for one game late in the schedule (e.g. 2022).
    features, columns = build_features(schedule, rolling_window=1)
    
    # Season 2022 starts at index 8. Let's pick index 10.
    features.loc[10, columns[0]] = np.nan
    
    # 3. min_games_before_train skips games early in the backtest.
    # min_games=2 should skip at most the first week or two.
    result = Backtester(min_games_before_train=2, random_state=7, logreg_c=0.05).run(features, columns)
    
    cov = result.coverage
    assert cov["total_games"] == 16
    assert cov["games_with_result"] == 15
    assert cov["tested"] > 0
    assert cov["skipped_missing_features"] > 0
    assert cov["skipped_min_games"] > 0
    assert cov["tested"] + cov["skipped_missing_features"] + cov["skipped_min_games"] == 15


def test_unplayed_games_are_never_trained_or_tested(spy):
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=6)
    # Make one game unplayed in the middle of 2021
    # We'll use a specific game_id to track it.
    unplayed_idx = 5
    schedule.loc[unplayed_idx, ["home_score", "away_score"]] = np.nan
    unplayed_id = schedule.loc[unplayed_idx, "game_id"]
    unplayed_week_key = schedule.loc[unplayed_idx, "season"] * 100 + schedule.loc[unplayed_idx, "week"]
    
    features, columns = _keyed_features(schedule)
    # features["home_win"] is NaN for unplayed_idx.
    
    result = Backtester(min_games_before_train=1, random_state=7, logreg_c=1.0).run(features, columns)
    
    # 1. Check that the unplayed game is not in predictions
    assert unplayed_id not in result.predictions["game_id"].values
    
    # 2. Check that it was never in any training set
    assert spy.calls, "SpyModel was never called"
    for train_keys, test_keys in spy.calls:
        # train_keys are from X_train[:, 0] which are week_keys.
        # But wait, multiple games can have the same week_key.
        # We need to be careful. If unplayed_idx is in the train week, but filtered out...
        # The number of games in that week in train_keys should be one less.
        
        test_key = test_keys[0]
        if unplayed_week_key < test_key:
            # This week was part of training.
            actual_count = (train_keys == unplayed_week_key).sum()
            expected_count = (
                (features["week_key"] == unplayed_week_key) & features["home_win"].notna()
            ).sum()
            assert actual_count == expected_count

    # 3. Overall count check for every split
    for train_keys, test_keys in spy.calls:
        test_key = test_keys[0]
        expected_train_count = (
            (features["week_key"] < test_key) & features["home_win"].notna()
        ).sum()
        assert len(train_keys) == expected_train_count
