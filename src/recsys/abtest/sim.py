"""Synthetic two-arm experiments on a heavy-tailed per-user engagement metric.

This is simulated data, not any company's logs. The shape imitates a marketplace metric such as
"chats started per user per week": most users do little, a few do a lot.
    (log X_u, log Y0_u) ~ bivariate normal, means mu, sds sigma, correlation corr
    pre-period X_u is known before assignment; outcome Y_u = Y0_u * (1 + lift * T_u)
So X and Y are log-normal (heavy-tailed) and correlated; `corr` is the correlation on the log scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    mu: float = 1.0        # log-scale mean of weekly activity
    sigma: float = 1.0     # log-scale spread (heavy tail)
    corr: float = 0.7      # log-scale correlation between pre-period and experiment-period activity


def draw(pop: Population, n: int, rng: np.random.Generator, lift: float = 0.0, reps: int = 1):
    """(reps, n) arrays X, Y, T with T ~ Bernoulli(0.5). Treatment multiplies Y by (1 + lift)."""
    z1 = rng.standard_normal((reps, n))
    z2 = pop.corr * z1 + np.sqrt(1 - pop.corr ** 2) * rng.standard_normal((reps, n))
    x = np.exp(pop.mu + pop.sigma * z1)
    y = np.exp(pop.mu + pop.sigma * z2)
    t = rng.random((reps, n)) < 0.5
    y = np.where(t, y * (1 + lift), y)
    return x, y, t
