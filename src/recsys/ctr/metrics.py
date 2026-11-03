"""Weighted CTR metrics (weights = bootstrap resample counts) and user-cluster bootstrap."""
from __future__ import annotations

import numpy as np


def auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted ROC AUC; tied scores count one half (Mann-Whitney)."""
    w = np.ones(len(y)) if w is None else w
    _, inv = np.unique(p, return_inverse=True)
    pos = np.bincount(inv, weights=w * y)
    neg = np.bincount(inv, weights=w * (1 - y))
    neg_below = np.cumsum(neg) - neg
    denom = pos.sum() * neg.sum()
    return float((pos * (neg_below + 0.5 * neg)).sum() / denom) if denom > 0 else float("nan")


def logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-7) -> float:
    w = np.ones(len(y)) if w is None else w
    p = np.clip(p, eps, 1 - eps)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def ece(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by bin mass."""
    w = np.ones(len(y)) if w is None else w
    b = np.minimum((p * bins).astype(int), bins - 1)
    W = np.bincount(b, weights=w, minlength=bins)
    P = np.bincount(b, weights=w * p, minlength=bins)
    Y = np.bincount(b, weights=w * y, minlength=bins)
    nz = W > 0
    return float((np.abs(P[nz] - Y[nz])).sum() / W.sum())


def calibration_ratio(y, p, w=None) -> float:
    """Mean prediction / observed positive rate (1.0 = calibrated on average)."""
    w = np.ones(len(y)) if w is None else w
    return float((w * p).sum() / (w * y).sum())


def cluster_weights(groups: np.ndarray, n_boot: int, seed: int = 42):
    """Yield row weights for a cluster bootstrap: resample groups (users) with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[inv].astype(np.float64)


def ci(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi)}


class Platt:
    """Logistic recalibration on the logit: p' = sigmoid(a * logit(p) + b), fit by Newton's method."""

    def fit(self, y, p, iters: int = 50) -> "Platt":
        z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        X = np.stack([z, np.ones_like(z)], 1)
        theta = np.array([1.0, 0.0])
        for _ in range(iters):
            q = 1 / (1 + np.exp(-X @ theta))
            grad = X.T @ (q - y)
            H = (X * (q * (1 - q))[:, None]).T @ X + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, grad)
            theta -= step
            if np.abs(step).max() < 1e-10:
                break
        self.a, self.b = float(theta[0]), float(theta[1])
        return self

    def transform(self, p):
        z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
        return 1 / (1 + np.exp(-(self.a * z + self.b)))
