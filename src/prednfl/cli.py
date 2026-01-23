from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

import pandas as pd

from prednfl.backtest import Backtester
from prednfl.config import BacktestConfig, DEFAULT_CONFIG
from prednfl.data import DataLoader
from prednfl.features import build_features
from prednfl.models import LogisticRegressionModel


def _parse_seasons(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Seasons cannot be empty.")
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    if "-" in value:
        start_str, end_str = [part.strip() for part in value.split("-", maxsplit=1)]
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise argparse.ArgumentTypeError("Season range must be start-end with end >= start.")
        return tuple(range(start, end + 1))
    return (int(value),)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prednfl", description="NFL win prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest.add_argument(
        "--seasons",
        type=_parse_seasons,
        help="Season list or range, e.g. 2021-2024 or 2021,2022,2023,2024",
    )
    backtest.add_argument("--rolling-window", type=int, help="Rolling games window")
    backtest.add_argument("--min-games", type=int, help="Min games before training")
    backtest.add_argument("--c", type=float, help="Logistic regression C value")

    predict = subparsers.add_parser("predict", help="Predict a single matchup")
    predict.add_argument("--season", type=int, required=True, help="Season year")
    predict.add_argument("--week", type=int, help="Week number")
    predict.add_argument("--game-id", help="Optional game_id override")
    predict.add_argument(
        "--teams",
        nargs=2,
        metavar=("TEAM1", "TEAM2"),
        help="Team abbreviations (e.g., BUF MIA). Auto-detects next matchup.",
    )
    predict.add_argument("--home", help="Home team abbreviation (e.g., BUF)")
    predict.add_argument("--away", help="Away team abbreviation (e.g., MIA)")
    predict.add_argument(
        "--allow-missing",
        action="store_true",
        help="Fill missing matchup features using training means.",
    )

    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    config = DEFAULT_CONFIG
    if getattr(args, "seasons", None):
        config = replace(config, seasons=tuple(args.seasons))
    if getattr(args, "rolling_window", None):
        config = replace(config, rolling_window_games=int(args.rolling_window))
    if getattr(args, "min_games", None):
        config = replace(config, min_games_before_train=int(args.min_games))
    if getattr(args, "c", None) is not None:
        config = replace(config, logreg_c=float(args.c))
    return config


def _load_features(config: BacktestConfig) -> tuple[pd.DataFrame, list[str]]:
    loader = DataLoader(seasons=config.seasons)
    schedules = loader.load_schedules()
    team_stats = loader.load_team_stats()
    return build_features(
        schedules,
        rolling_window=config.rolling_window_games,
        team_stats=team_stats,
        team_stats_metrics=config.team_stats_metrics,
    )


def _run_backtest(config: BacktestConfig) -> None:
    features, feature_columns = _load_features(config)
    backtester = Backtester(
        min_games_before_train=config.min_games_before_train,
        random_state=config.random_state,
        logreg_c=config.logreg_c,
    )
    result = backtester.run(features, feature_columns)

    ordered_prefixes = ("model_", "baseline_home_rate_", "baseline_prior_season_")
    print(f"Backtest summary metrics (C={config.logreg_c})")
    for prefix in ordered_prefixes:
        print(f"\n{prefix.rstrip('_')}")
        for key, value in result.metrics.items():
            if key.startswith(prefix):
                metric_name = key.replace(prefix, "")
                print(f"{metric_name}: {value:.4f}")

    print("\nSample predictions")
    print(result.predictions.head(10))


def _select_game(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.game_id:
        return features[features["game_id"] == args.game_id].copy()

    if args.teams:
        team_a, team_b = args.teams
        season_games = features[features["season"] == args.season].copy()
        candidates = season_games[
            (
                (season_games["home_team"] == team_a)
                & (season_games["away_team"] == team_b)
            )
            | (
                (season_games["home_team"] == team_b)
                & (season_games["away_team"] == team_a)
            )
        ].copy()

        if candidates.empty:
            raise SystemExit(
                "No matching game found for that season/team pair. "
                "Check abbreviations or try a different season."
            )

        candidates["game_date"] = candidates["game_date"].astype("datetime64[ns]")
        candidates.sort_values(["game_date", "week", "game_id"], inplace=True)
        today = pd.Timestamp.today().normalize()
        upcoming = candidates[candidates["game_date"] >= today]
        if not upcoming.empty:
            return upcoming.head(1).copy()

        print("No upcoming game found; using the most recent matchup in that season.")
        return candidates.tail(1).copy()

    if not args.home or not args.away:
        raise SystemExit("Provide --home and --away (or use --teams).")
    if args.week is None:
        raise SystemExit("Provide --week (or use --teams).")

    return features[
        (features["season"] == args.season)
        & (features["week"] == args.week)
        & (features["home_team"] == args.home)
        & (features["away_team"] == args.away)
    ].copy()


def _run_predict(config: BacktestConfig, args: argparse.Namespace) -> None:
    seasons = tuple(sorted(set(config.seasons + (args.season,))))
    predict_config = replace(config, seasons=seasons)
    features, feature_columns = _load_features(predict_config)
    game = _select_game(features, args)

    if game.empty:
        season_games = features[features["season"] == args.season]
        candidates = season_games[
            (
                (season_games["home_team"] == args.home)
                & (season_games["away_team"] == args.away)
            )
            | (
                (season_games["home_team"] == args.away)
                & (season_games["away_team"] == args.home)
            )
        ][["game_id", "week", "home_team", "away_team"]].copy()

        if candidates.empty:
            raise SystemExit(
                "No matching game found for that season/team pair. "
                "Check abbreviations or try a different week."
            )

        print("No exact match for that week. Available matchups this season:")
        print(candidates.sort_values(["week", "game_id"]).to_string(index=False))
        raise SystemExit(0)

    missing_mask = game[feature_columns].isna().iloc[0]
    missing_cols = game[feature_columns].columns[missing_mask].tolist()
    missing_count = len(missing_cols)
    total_count = len(feature_columns)

    if missing_count:
        print(f"Missing features: {missing_count}/{total_count}")

    season = int(game.iloc[0]["season"])
    week = int(game.iloc[0]["week"])
    home_team = str(game.iloc[0]["home_team"])
    away_team = str(game.iloc[0]["away_team"])

    train_mask = (features["season"] < season) | (
        (features["season"] == season) & (features["week"] < week)
    )
    train = features[train_mask].dropna(subset=["home_win"]).copy()

    if args.allow_missing:
        feature_means = train[feature_columns].mean(numeric_only=True)
        train.loc[:, feature_columns] = train[feature_columns].fillna(feature_means)
        game.loc[:, feature_columns] = game[feature_columns].fillna(feature_means)
    elif missing_count:
        raise SystemExit(
            "Missing features for this matchup (likely early season). "
            f"Missing: {', '.join(missing_cols)}. "
            "Use --allow-missing to fill from training means."
        )

    train = train.dropna(subset=feature_columns)
    if train.empty:
        raise SystemExit("Not enough prior games to train. Try a later week or add seasons.")

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["home_win"].to_numpy(dtype=int)

    model = LogisticRegressionModel(random_state=config.random_state, c_value=config.logreg_c)
    model.fit(X_train, y_train)

    X_game = game[feature_columns].to_numpy(dtype=float)
    home_prob = float(model.predict_proba(X_game)[0])
    away_prob = 1.0 - home_prob

    print("Matchup prediction")
    print(f"Season {season}, Week {week}")
    print(f"Home {home_team}: {home_prob:.3f}")
    print(f"Away {away_team}: {away_prob:.3f}")


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _build_config(args)

    if args.command == "backtest":
        _run_backtest(config)
    elif args.command == "predict":
        _run_predict(config, args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
