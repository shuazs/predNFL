from __future__ import annotations

import math

import numpy as np
import pytest

from prednfl.metrics import brier_score, evaluate


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
