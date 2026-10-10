from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import expit


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


class MarketOffsetLogit:
    """Logistic regression with a fixed logit offset from the market.

    Model: logit(p) = market_logit + Xw + b
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.w = None
        self.b = None
        self.scaler = StandardScaler()

    def _objective(self, params, X, y, offset):
        b = params[0]
        w = params[1:]
        z = offset + X @ w + b
        
        # Mean log loss
        # log(1 + exp(z)) - y*z
        # Using np.logaddexp(0, z) for stability
        loss = np.mean(np.logaddexp(0, z) - y * z)
        
        # L2 penalty on w (not b)
        reg = self.alpha * np.sum(w**2)
        
        obj = loss + reg
        
        # Gradient
        # dloss/dz = sigmoid(z) - y
        p = expit(z)
        dz = p - y
        grad_b = np.mean(dz)
        grad_w = (X.T @ dz) / len(y) + 2 * self.alpha * w
        
        grad = np.concatenate([[grad_b], grad_w])
        return obj, grad

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> None:
        if X.shape[1] > 0:
            X_scaled = self.scaler.fit_transform(X)
            # Handle std=0: replace NaNs (from X/0) with 0
            # StandardScaler(with_std=True) sets scale to 1 if std=0
            # so we don't need to do much, it handles it.
        else:
            X_scaled = X

        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)
        
        res = minimize(
            fun=self._objective,
            x0=initial_params,
            args=(X_scaled, y, offset),
            method="L-BFGS-B",
            jac=True,
        )
        
        if not res.success:
            import warnings
            warnings.warn(f"MarketOffsetLogit optimization failed: {res.message}")
            
        self.b = res.x[0]
        self.w = res.x[1:]

    def predict_proba(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if X.shape[1] > 0:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        z = offset + X_scaled @ self.w + self.b
        return expit(z)
