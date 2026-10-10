from __future__ import annotations

import argparse

import pandas as pd
import pytest

import prednfl.cli as cli
from prednfl.config import DEFAULT_CONFIG
from prednfl.features import build_features

from conftest import make_schedule


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("2021-2024", (2021, 2022, 2023, 2024)),
        ("2021,2023", (2021, 2023)),
        (" 2021 , 2022 ,", (2021, 2022)),
        ("2022", (2022,)),
        ("2022-2022", (2022,)),
    ],
)
def test_parse_seasons(value, expected):
    assert cli._parse_seasons(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "2024-2021"])
def test_parse_seasons_rejects_bad_input(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_seasons(value)


def test_build_config_defaults():
    args = cli._parse_args(["backtest"])
    assert cli._build_config(args) == DEFAULT_CONFIG


def test_build_config_overrides():
    args = cli._parse_args(
        [
            "backtest", "--seasons", "2019-2020", "--rolling-window", "4",
            "--min-games", "20", "--c", "0.5", "--model", "market_offset", "--alpha", "0.1"
        ]
    )
    config = cli._build_config(args)
    assert config.seasons == (2019, 2020)
    assert config.rolling_window_games == 4
    assert config.min_games_before_train == 20
    assert config.logreg_c == 0.5
    assert config.model == "market_offset"
    assert config.alpha == 0.1


def test_build_config_allows_c_of_zero_to_reach_model():
    """--c uses an `is not None` check, so an explicit 0 is honored (unlike the others)."""
    config = cli._build_config(cli._parse_args(["backtest", "--c", "0"]))
    assert config.logreg_c == 0.0


# --------------------------------------------------------------------------
# Game selection
# --------------------------------------------------------------------------

def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2025, 2025, 2025, 2024],
            "week": [2, 9, 15, 3],
            "game_date": pd.to_datetime(["2000-09-14", "2000-11-02", "2099-12-14", "2000-09-21"]),
            "home_team": ["NE", "DEN", "NE", "NE"],
            "away_team": ["DEN", "NE", "DEN", "DEN"],
        }
    )


def _select(argv):
    return cli._select_game(_games(), cli._parse_args(["predict", *argv]))


def test_select_by_game_id():
    assert _select(["--season", "2025", "--game-id", "g2"])["game_id"].tolist() == ["g2"]


def test_select_by_home_away_week():
    assert _select(["--season", "2025", "--week", "9", "--home", "DEN", "--away", "NE"])[
        "game_id"
    ].tolist() == ["g2"]


def test_select_by_teams_prefers_next_upcoming_game():
    assert _select(["--season", "2025", "--teams", "DEN", "NE"])["game_id"].tolist() == ["g3"]


def test_select_by_teams_falls_back_to_most_recent(capsys):
    games = _games()
    games = games[games["game_id"] != "g3"]
    args = cli._parse_args(["predict", "--season", "2025", "--teams", "NE", "DEN"])
    assert cli._select_game(games, args)["game_id"].tolist() == ["g2"]
    assert "No upcoming game found" in capsys.readouterr().out


def test_select_by_teams_with_no_match_exits():
    with pytest.raises(SystemExit, match="No matching game"):
        _select(["--season", "2025", "--teams", "NE", "BUF"])


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--season", "2025", "--week", "9", "--home", "DEN"], "--home and --away"),
        (["--season", "2025", "--home", "DEN", "--away", "NE"], "--week"),
    ],
)
def test_select_requires_full_selector(argv, message):
    with pytest.raises(SystemExit, match=message):
        _select(argv)


# --------------------------------------------------------------------------
# End-to-end commands (data loading stubbed out; no network)
# --------------------------------------------------------------------------

@pytest.fixture
def offline_features(monkeypatch):
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=12, seed=3)

    def fake_load(config):
        return build_features(schedule, rolling_window=config.rolling_window_games)

    monkeypatch.setattr(cli, "_load_features", fake_load)
    return schedule


def test_backtest_command_prints_all_sections(offline_features, capsys):
    cli.main(["backtest", "--rolling-window", "2", "--min-games", "8"])
    out = capsys.readouterr().out
    for section in ("Tested", "Backtest summary metrics", "Acc", "per-week average", "Sample predictions"):
        assert section in out


def _game(schedule, season, week):
    return schedule[(schedule["season"] == season) & (schedule["week"] == week)].iloc[0]


def test_predict_command_outputs_probabilities_that_sum_to_one(offline_features, capsys):
    game = _game(offline_features, 2022, 8)
    cli.main(
        ["predict", "--season", "2022", "--week", "8", "--home", game.home_team, "--away", game.away_team]
    )
    lines = capsys.readouterr().out.splitlines()
    home = float(next(l for l in lines if l.startswith("Home")).split(":")[1])
    away = float(next(l for l in lines if l.startswith("Away")).split(":")[1])
    assert 0 <= home <= 1
    assert home + away == pytest.approx(1.0, abs=1e-3)


def test_predict_command_refuses_missing_features_without_flag(offline_features):
    game = _game(offline_features, 2021, 1)  # no rolling history yet
    with pytest.raises(SystemExit, match="--allow-missing"):
        cli.main(
            ["predict", "--season", "2021", "--week", "1", "--home", game.home_team, "--away", game.away_team]
        )


def test_predict_command_allow_missing_fills_from_training_means(monkeypatch, capsys):
    schedule = make_schedule(seasons=(2021, 2022), n_weeks=12, seed=3)
    game = _game(schedule, 2022, 8)

    def load_with_gap(config):
        features, columns = build_features(schedule, rolling_window=2)
        features.loc[features["game_id"] == game.game_id, "home_pf_roll"] = float("nan")
        return features, columns

    monkeypatch.setattr(cli, "_load_features", load_with_gap)

    with pytest.raises(SystemExit, match="home_pf_roll"):
        cli.main(["predict", "--season", "2022", "--week", "8", "--home", game.home_team, "--away", game.away_team])

    cli.main(
        [
            "predict", "--season", "2022", "--week", "8",
            "--home", game.home_team, "--away", game.away_team, "--allow-missing",
        ]
    )
    out = capsys.readouterr().out
    assert "Missing features" in out
    assert "Home" in out


def test_predict_command_lists_alternatives_for_wrong_week(offline_features, capsys):
    game = _game(offline_features, 2022, 8)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["predict", "--season", "2022", "--week", "99", "--home", game.home_team, "--away", game.away_team]
        )
    assert exc.value.code == 0
    assert "Available matchups" in capsys.readouterr().out
