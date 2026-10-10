from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import LogisticRegressionModel, MarketOffsetLogit
from .metrics import evaluate


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    pooled_metrics: dict[str, float]
    coverage: dict[str, int]


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
    def __init__(
        self,
        min_games_before_train: int,
        random_state: int,
        logreg_c: float,
        model: str = "logreg",
        alpha: float = 0.01,
    ) -> None:
        self.min_games_before_train = min_games_before_train
        self.random_state = random_state
        self.logreg_c = logreg_c
        self.model_type = model
        self.alpha = alpha

    def run(self, features: pd.DataFrame, feature_columns: list[str]) -> BacktestResult:
        features = features.copy()
        # Coverage tracking - initial counts
        total_games = len(features)
        games_with_result_initial = int(features["home_win"].notna().sum())

        # Handle MarketOffsetLogit special requirements
        active_features = feature_columns
        offset_col = None
        skipped_market_logit = 0
        if self.model_type == "market_offset":
            offset_col = "market_logit"
            # Exclude market-related columns from features
            exclude = {
                "spread_line",
                "home_ml_prob",
                "away_ml_prob",
                "ml_prob_diff",
                "market_prob",
                "market_logit",
            }
            active_features = [c for c in feature_columns if c not in exclude]
            # Games missing market_logit are considered "skipped_missing_features"
            # but they must have a result to be counted in that bucket.
            missing_logit = features[offset_col].isna() & features["home_win"].notna()
            skipped_market_logit = int(missing_logit.sum())
            # Drop rows with NaN market_logit for processing
            features = features.dropna(subset=[offset_col])

        features.sort_values(["season", "week", "game_date", "game_id"], inplace=True)

        predictions = []
        model_metrics_by_split = []
        home_rate_metrics_by_split = []
        prior_season_metrics_by_split = []
        market_metrics_by_split = []
        prior_season_win_pcts = _team_season_win_pct(features)

        # Coverage tracking
        games_with_result = int(features["home_win"].notna().sum())
        skipped_min_games = 0
        skipped_missing_features = skipped_market_logit
        tested = 0

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

            train = features[train_mask].dropna(subset=active_features + ["home_win"])
            test_all = features[test_mask]
            test_with_result = test_all.dropna(subset=["home_win"])
            test = test_with_result.dropna(subset=active_features)

            if test_with_result.empty:
                continue

            if len(train) < self.min_games_before_train:
                skipped_min_games += len(test_with_result)
                continue

            skipped_missing_features += len(test_with_result) - len(test)

            if test.empty:
                continue

            tested += len(test)

            X_train = train[active_features].to_numpy(dtype=float)
            y_train = train["home_win"].to_numpy(dtype=int)
            X_test = test[active_features].to_numpy(dtype=float)
            y_test = test["home_win"].to_numpy(dtype=int)

            if self.model_type == "market_offset":
                offset_train = train[offset_col].to_numpy(dtype=float)
                offset_test = test[offset_col].to_numpy(dtype=float)
                model = MarketOffsetLogit(alpha=self.alpha)
                model.fit(X_train, y_train, offset_train)
                probs = model.predict_proba(X_test, offset_test)
            else:
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

            market_probs = None
            if "home_ml_prob" in test.columns and "away_ml_prob" in test.columns:
                market_probs = test["home_ml_prob"] / (test["home_ml_prob"] + test["away_ml_prob"])
                market_probs = market_probs.to_numpy(dtype=float)
                market_metrics_by_split.append(evaluate(y_test, market_probs))

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
            if market_probs is not None:
                pred_frame["baseline_market"] = market_probs
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
        if market_metrics_by_split:
            summary.update(_summarize_metrics(market_metrics_by_split, "baseline_market"))

        pooled = {}
        if not all_predictions.empty:
            y_true = all_predictions["home_win"].to_numpy(dtype=int)
            pooled.update(_summarize_metrics([evaluate(y_true, all_predictions["home_win_prob"].to_numpy())], "model"))
            pooled.update(_summarize_metrics([evaluate(y_true, all_predictions["baseline_home_rate"].to_numpy())], "baseline_home_rate"))
            pooled.update(_summarize_metrics([evaluate(y_true, all_predictions["baseline_prior_win_pct"].to_numpy())], "baseline_prior_season"))
            if "baseline_market" in all_predictions.columns:
                pooled.update(_summarize_metrics([evaluate(y_true, all_predictions["baseline_market"].to_numpy())], "baseline_market"))

        coverage = {
            "total_games": total_games,
            "games_with_result": games_with_result_initial,
            "tested": tested,
            "skipped_missing_features": skipped_missing_features,
            "skipped_min_games": skipped_min_games,
        }

        return BacktestResult(
            predictions=all_predictions,
            metrics=summary,
            pooled_metrics=pooled,
            coverage=coverage,
        )
