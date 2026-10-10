import pandas as pd
import numpy as np
from prednfl.backtest import Backtester
from prednfl.config import BacktestConfig
from prednfl.data import DataLoader
from prednfl.features import build_features
from prednfl.metrics import paired_bootstrap_diff

def run_ablation():
    config = BacktestConfig(
        seasons=(2021, 2022, 2023, 2024),
        model="market_offset"
    )
    
    loader = DataLoader(seasons=config.seasons)
    schedules = loader.load_schedules()
    team_stats = loader.load_team_stats()
    features, all_columns = build_features(
        schedules,
        rolling_window=config.rolling_window_games,
        team_stats=team_stats,
        team_stats_metrics=config.team_stats_metrics
    )
    
    groups = {
        "none": [],
        "points": ["pf_roll_diff", "pa_roll_diff"],
        "rest": ["rest_days_diff", "rest_sched_diff"],
        "offense_epa": ["ts_passing_epa_roll_diff", "ts_rushing_epa_roll_diff", "ts_receiving_epa_roll_diff"],
        "defense": ["ts_def_sacks_roll_diff", "ts_def_interceptions_roll_diff", "ts_def_tackles_for_loss_roll_diff"],
        "all": [
            "pf_roll_diff", "pa_roll_diff",
            "rest_days_diff", "rest_sched_diff",
            "ts_passing_epa_roll_diff", "ts_rushing_epa_roll_diff", "ts_receiving_epa_roll_diff",
            "ts_def_sacks_roll_diff", "ts_def_interceptions_roll_diff", "ts_def_tackles_for_loss_roll_diff"
        ]
    }
    
    # Required Change 2: Use common set of games for evaluation.
    # We find all games that are TESTED in the 'all' configuration.
    backtester_all = Backtester(
        min_games_before_train=config.min_games_before_train,
        random_state=config.random_state,
        logreg_c=config.logreg_c,
        model="market_offset",
        alpha=0.1 # Any alpha is fine for finding tested games
    )
    result_all = backtester_all.run(features, groups["all"])
    common_game_ids = set(result_all.predictions["game_id"])
    
    alphas = [0.001, 0.01, 0.1]
    
    n_common = len(common_game_ids)
    print(f"Ablation on {n_common} common games (2021-2024)")
    print(f"{'Group':<15} {'Alpha':<7} {'N':>5} {'Diff':>8} {'95% CI':<20}")
    print("-" * 60)
    
    for group_name, cols in groups.items():
        available_cols = [c for c in cols if c in all_columns]
        
        for alpha in alphas:
            backtester = Backtester(
                min_games_before_train=config.min_games_before_train,
                random_state=config.random_state,
                logreg_c=config.logreg_c,
                model="market_offset",
                alpha=alpha
            )
            
            result = backtester.run(features, available_cols)
            
            # Filter to common games
            preds = result.predictions[result.predictions["game_id"].isin(common_game_ids)]
            
            if preds.empty:
                print(f"{group_name:<15} {alpha:<7} {'0':>5} {'N/A':>8}")
                continue
            
            assert len(preds) == n_common, f"Expected {n_common} games, got {len(preds)} for {group_name}"
                
            y_true = preds["home_win"].to_numpy()
            y_model = preds["home_win_prob"].to_numpy()
            y_market = preds["baseline_market"].to_numpy()
            
            stats = paired_bootstrap_diff(y_true, y_model, y_market)
            ci_low, ci_high = stats["lo"], stats["hi"]
            diff = stats["mean"]
            
            print(f"{group_name:<15} {alpha:<7} {len(preds):>5} {diff:>8.4f} ({ci_low:>.4f} to {ci_high:>.4f})")

if __name__ == "__main__":
    run_ablation()
