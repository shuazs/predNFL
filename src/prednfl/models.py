from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    """Wrapper for sklearn logistic regression."""

    def __init__(self, random_state: int, c_value: float) -> None:
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        C=c_value,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
