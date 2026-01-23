from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import LogisticRegressionModel
from .metrics import evaluate


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]


def _team_season_win_pct(features: pd.DataFrame) -> pd.DataFrame:
    home = features[["season", "home_team", "home_score", "away_score"]].copy()
    home.rename(columns={"home_team": "team"}, inplace=True)
    home["win"] = (home["home_score"] > home["away_score"]).astype(int)

    away = features[["season", "away_team", "home_score", "away_score"]].copy()
    away.rename(columns={"away_team": "team"}, inplace=True)
    away["win"] = (away["away_score"] > away["home_score"]).astype(int)

    team_games = pd.concat([home[["season", "team", "win"]], away[["season", "team", "win"]]])
    grouped = team_games.groupby(["season", "team"], sort=False)["win"].agg(["mean", "count"]).reset_index()
    grouped.rename(columns={"mean": "win_pct", "count": "games"}, inplace=True)
    return grouped


def _prior_season_probs(test: pd.DataFrame, win_pcts: pd.DataFrame) -> np.ndarray:
    lookup = {
        (row.season, row.team): row.win_pct for row in win_pcts.itertuples(index=False)
    }
    probs = []
    for row in test.itertuples(index=False):
        prev_season = int(row.season) - 1
        home_pct = lookup.get((prev_season, row.home_team), 0.5)
        away_pct = lookup.get((prev_season, row.away_team), 0.5)
        denom = home_pct + away_pct
        probs.append(home_pct / denom if denom > 0 else 0.5)
    return np.array(probs, dtype=float)


def _summarize_metrics(metrics_by_split: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not metrics_by_split:
        return {
            f"{prefix}_accuracy": float("nan"),
            f"{prefix}_log_loss": float("nan"),
            f"{prefix}_brier": float("nan"),
        }
    metrics_df = pd.DataFrame(metrics_by_split)
    summary = {}
    for metric, value in metrics_df.mean().to_dict().items():
        summary[f"{prefix}_{metric}"] = value
    return summary


class Backtester:
    def __init__(self, min_games_before_train: int, random_state: int, logreg_c: float) -> None:
        self.min_games_before_train = min_games_before_train
        self.random_state = random_state
        self.logreg_c = logreg_c

    def run(self, features: pd.DataFrame, feature_columns: list[str]) -> BacktestResult:
        features = features.copy()
        features.sort_values(["season", "week", "game_date", "game_id"], inplace=True)

        predictions = []
        model_metrics_by_split = []
        home_rate_metrics_by_split = []
        prior_season_metrics_by_split = []
        prior_season_win_pcts = _team_season_win_pct(features)

        split_keys = (
            features[["season", "week"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["season", "week"])
        )

        for season, week in split_keys.itertuples(index=False):
            train_mask = (features["season"] < season) | (
                (features["season"] == season) & (features["week"] < week)
            )
            test_mask = (features["season"] == season) & (features["week"] == week)

            train = features[train_mask].dropna(subset=feature_columns + ["home_win"])
            test = features[test_mask].dropna(subset=feature_columns + ["home_win"])

            if len(train) < self.min_games_before_train or test.empty:
                continue

            X_train = train[feature_columns].to_numpy(dtype=float)
            y_train = train["home_win"].to_numpy(dtype=int)
            X_test = test[feature_columns].to_numpy(dtype=float)
            y_test = test["home_win"].to_numpy(dtype=int)

            model = LogisticRegressionModel(
                random_state=self.random_state,
                c_value=self.logreg_c,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)

            model_metrics_by_split.append(evaluate(y_test, probs))

            home_rate = float(y_train.mean())
            home_rate_probs = np.full_like(y_test, fill_value=home_rate, dtype=float)
            home_rate_metrics_by_split.append(evaluate(y_test, home_rate_probs))

            prior_probs = _prior_season_probs(test, prior_season_win_pcts)
            prior_season_metrics_by_split.append(evaluate(y_test, prior_probs))

            pred_frame = test[[
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
            ]].copy()
            pred_frame["home_win_prob"] = probs
            pred_frame["baseline_home_rate"] = home_rate_probs
            pred_frame["baseline_prior_win_pct"] = prior_probs
            pred_frame["home_win"] = y_test
            predictions.append(pred_frame)

        if predictions:
            all_predictions = pd.concat(predictions, ignore_index=True)
        else:
            all_predictions = pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "home_win_prob",
                    "home_win",
                ]
            )

        summary = {}
        summary.update(_summarize_metrics(model_metrics_by_split, "model"))
        summary.update(_summarize_metrics(home_rate_metrics_by_split, "baseline_home_rate"))
        summary.update(_summarize_metrics(prior_season_metrics_by_split, "baseline_prior_season"))

        return BacktestResult(predictions=all_predictions, metrics=summary)
