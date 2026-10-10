from __future__ import annotations

import numpy as np
import pytest

from prednfl.models import LogisticRegressionModel, MarketOffsetLogit
from scipy.optimize import check_grad
from scipy.special import expit


def _toy_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Features on very different scales; only the first one carries signal.
    X = np.column_stack([rng.normal(0, 1, n), rng.normal(200, 50, n)])
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


def test_predict_proba_returns_positive_class_probabilities():
    X, y = _toy_data()
    model = LogisticRegressionModel(random_state=7, c_value=1.0)
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))
    # Higher signal -> higher home-win probability.
    assert probs[X[:, 0] > 1].mean() > 0.8
    assert probs[X[:, 0] < -1].mean() < 0.2


def test_smaller_c_shrinks_predictions_toward_base_rate():
    X, y = _toy_data()
    loose = LogisticRegressionModel(random_state=7, c_value=10.0)
    tight = LogisticRegressionModel(random_state=7, c_value=1e-4)
    loose.fit(X, y)
    tight.fit(X, y)

    assert np.std(tight.predict_proba(X)) < np.std(loose.predict_proba(X))


def test_is_deterministic():
    X, y = _toy_data()
    a = LogisticRegressionModel(random_state=7, c_value=0.05)
    b = LogisticRegressionModel(random_state=7, c_value=0.05)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_offset_model_gradient_matches_numerical():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 3))
    y = rng.integers(0, 2, 100)
    offset = rng.normal(0, 1, 100)
    model = MarketOffsetLogit(alpha=0.1)

    def func(p):
        return model._objective(p, X, y, offset)[0]

    def grad(p):
        return model._objective(p, X, y, offset)[1]

    initial_params = rng.normal(0, 1, 4)
    err = check_grad(func, grad, initial_params)
    assert err < 1e-6


def test_offset_model_without_features_returns_sigmoid_offset_plus_intercept():
    X = np.empty((100, 0))
    y = np.concatenate([np.ones(60), np.zeros(40)])  # 60% wins
    offset = np.zeros(100)
    
    model = MarketOffsetLogit(alpha=0.1)
    model.fit(X, y, offset)
    
    # Model: logit(p) = 0 + b.
    # We expect p ≈ 0.6, so b ≈ log(0.6/0.4) ≈ 0.405
    assert model.w.shape == (0,)
    assert model.b == pytest.approx(np.log(0.6 / 0.4), abs=1e-3)
    
    probs = model.predict_proba(X, offset)
    assert np.allclose(probs, 0.6, atol=1e-3)


def test_offset_model_recovers_known_weight():
    # simulate y ~ Bernoulli(sigmoid(offset + 0.8*x))
    n = 5000
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n, 1))
    offset = rng.normal(0, 1, n)
    z = offset + 0.8 * x.flatten()
    p = expit(z)
    y = rng.binomial(1, p)
    
    model = MarketOffsetLogit(alpha=1e-6) # tiny alpha
    model.fit(x, y, offset)
    
    # Weight should be within 0.15 of 0.8
    # Since x has mean 0 and std 1, StandardScaler shouldn't change it much.
    assert model.w[0] == pytest.approx(0.8, abs=0.15)


def test_offset_model_large_alpha_shrinks_weights_to_zero_but_not_intercept():
    n = 100
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n, 2))
    y = rng.binomial(1, 0.7, n)
    offset = np.zeros(n)
    
    model = MarketOffsetLogit(alpha=100.0) # Huge alpha
    model.fit(X, y, offset)
    
    assert model.w.all() == 0 or np.allclose(model.w, 0, atol=1e-2)
    assert model.b != 0
    assert model.b == pytest.approx(np.log(0.7 / 0.3), abs=0.5)
