from __future__ import annotations

import numpy as np

from prednfl.models import LogisticRegressionModel


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
