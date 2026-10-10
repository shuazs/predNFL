from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": brier_score(y_true, y_prob),
    }


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    eps = 1e-6
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)

    ll_a = -(y * np.log(p_a) + (1 - y) * np.log(1 - p_a))
    ll_b = -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b))
    diffs = ll_a - ll_b

    boot_means = []
    for _ in range(n_boot):
        boot_diffs = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(float(np.mean(boot_diffs)))

    boot_means = np.array(boot_means)
    return {
        "mean": float(np.mean(diffs)),
        "lo": float(np.percentile(boot_means, 2.5)),
        "hi": float(np.percentile(boot_means, 97.5)),
    }
