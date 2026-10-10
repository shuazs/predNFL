from __future__ import annotations

import math

import numpy as np
import pytest

from prednfl.metrics import brier_score, evaluate, paired_bootstrap_diff


def test_brier_score_known_values():
    y = np.array([1, 0, 1, 0])
    assert brier_score(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert brier_score(y, np.array([0.0, 1.0, 0.0, 1.0])) == 1.0


def test_evaluate_keys_and_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.4, 0.6])
    result = evaluate(y, p)

    assert set(result) == {"accuracy", "log_loss", "brier"}
    assert result["accuracy"] == pytest.approx(0.5)
    expected_ll = -np.mean([math.log(0.9), math.log(0.8), math.log(0.4), math.log(0.4)])
    assert result["log_loss"] == pytest.approx(expected_ll)
    assert result["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_evaluate_threshold_half_predicts_home_win():
    assert evaluate(np.array([1]), np.array([0.5]))["accuracy"] == 1.0
    assert evaluate(np.array([0]), np.array([0.5]))["accuracy"] == 0.0


def test_evaluate_handles_single_class_week():
    """A week where every home team wins must not crash log_loss."""
    result = evaluate(np.array([1, 1, 1]), np.array([0.7, 0.6, 0.8]))
    assert np.isfinite(result["log_loss"])


def test_paired_bootstrap_identical_inputs_is_zero():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.6, 0.4])
    result = paired_bootstrap_diff(y, p, p)
    assert result["mean"] == 0.0
    assert result["lo"] == 0.0
    assert result["hi"] == 0.0


def test_paired_bootstrap_interval_contains_mean_and_is_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p_a = rng.uniform(0, 1, size=100)
    p_b = rng.uniform(0, 1, size=100)

    res1 = paired_bootstrap_diff(y, p_a, p_b, n_boot=100, seed=42)
    res2 = paired_bootstrap_diff(y, p_a, p_b, n_boot=100, seed=42)
    res3 = paired_bootstrap_diff(y, p_a, p_b, n_boot=100, seed=43)

    assert res1["mean"] == res2["mean"]
    assert res1["lo"] == res2["lo"]
    assert res1["mean"] == res3["mean"]
    # With 100 random diffs and 100 boot samples, seeds should almost certainly differ
    assert res1["lo"] != res3["lo"]
    assert res1["lo"] <= res1["mean"] <= res1["hi"]
