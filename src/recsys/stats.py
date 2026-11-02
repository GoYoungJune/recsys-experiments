"""Paired bootstrap over users for comparing two models on per-user metrics."""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000
SEED = 42


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Compare mean(a) vs mean(b) where a[u], b[u] are metrics of the *same* user u.

    Users are resampled with replacement; both models keep their pairing, so per-user difficulty
    cancels out and only the model difference varies.

    Returns diff = mean(a) - mean(b), its 95% percentile CI, relative improvement mean(a)/mean(b) - 1,
    and a two-sided p-value from the null-centered bootstrap distribution.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same shape).")
    d = a - b
    observed = d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # H0: mean difference = 0. Shift the bootstrap distribution to be centered at 0 and ask how often
    # it is at least as extreme as what we observed. (+1)/(n+1) keeps p from being exactly 0.
    extreme = np.abs(boot - observed) >= abs(observed)
    p = (extreme.sum() + 1) / (n_boot + 1)
    mean_b = b.mean()
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(mean_b),
        "diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "rel_improvement": float(observed / mean_b) if mean_b > 0 else float("nan"),
        "p_value": float(min(p, 1.0)),
    }


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of x (used for error bars)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)
