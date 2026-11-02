"""Estimators and tests for two-arm experiments, vectorized over simulation replicates (axis 0)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def diff_in_means(y: np.ndarray, t: np.ndarray):
    """Welch z-statistic for mean(Y | T) - mean(Y | C). Returns (diff, standard error, z), each shape (reps,)."""
    n1, n0 = t.sum(axis=1), (~t).sum(axis=1)
    m1 = np.where(t, y, 0).sum(axis=1) / n1
    m0 = np.where(~t, y, 0).sum(axis=1) / n0
    v1 = np.where(t, (y - m1[:, None]) ** 2, 0).sum(axis=1) / (n1 - 1)
    v0 = np.where(~t, (y - m0[:, None]) ** 2, 0).sum(axis=1) / (n0 - 1)
    se = np.sqrt(v1 / n1 + v0 / n0)
    return m1 - m0, se, (m1 - m0) / se


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CUPED (Deng et al., 2013): Y - theta * (X - mean X), theta = cov(X, Y) / var(X) pooled over both arms.

    X is measured before assignment, so it is independent of T: the adjustment removes variance
    explained by X without biasing the treatment effect.
    """
    xc = x - x.mean(axis=1, keepdims=True)
    theta = (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)
    return y - theta[:, None] * xc


def sample_size_per_arm(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-sided z-test, equal arms: n = 2 (z_{1-a/2} + z_{power})^2 sd^2 / mde^2 per arm."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * z ** 2 * sd ** 2 / mde_abs ** 2))


def obrien_fleming_bounds(looks: int, alpha: float, rng: np.random.Generator, sims: int = 200_000) -> np.ndarray:
    """|z| boundaries c * sqrt(K / k) for equally spaced looks, c calibrated by Monte Carlo so that
    P(cross at any look | H0) = alpha. Uses the independent-increments structure of the z-process."""
    inc = rng.standard_normal((sims, looks))
    z_path = np.cumsum(inc, axis=1) / np.sqrt(np.arange(1, looks + 1))
    shape = np.sqrt(looks / np.arange(1, looks + 1))
    lo, hi = 1.0, 5.0
    for _ in range(60):  # bisection on c
        c = (lo + hi) / 2
        crossed = (np.abs(z_path) >= c * shape).any(axis=1).mean()
        lo, hi = (c, hi) if crossed > alpha else (lo, c)
    return hi * shape


def srm_pvalue(n_treat: np.ndarray, n_control: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """Chi-square goodness-of-fit test of the observed split against the designed ratio (1 df)."""
    n = n_treat + n_control
    e1, e0 = n * ratio, n * (1 - ratio)
    chi2 = (n_treat - e1) ** 2 / e1 + (n_control - e0) ** 2 / e0
    return stats.chi2.sf(chi2, df=1)
