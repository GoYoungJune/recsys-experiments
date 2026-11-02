from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender

DAY = 86400.0


class TimeDecay(Recommender):
    """Query-time recency weighting on top of an item-item model (EASE or ItemKNN).

    The base model is trained on the binary matrix exactly as before. Only the history used for scoring
    is reweighted:
        w_ui = exp(-(t_u_last - t_ui) / tau),   score(u, j) = sum_i w_ui * W_ij
    where W is EASE's B or ItemKNN's similarity. The user's latest interaction weighs 1, one that is
    `tau_days` older weighs 1/e. tau_days = inf gives back the base model.

    `times` holds a timestamp for every entry of the matrix passed to fit() (same sparsity pattern).
    """

    def __init__(self, base: Recommender, tau_days: float, times: sp.csr_matrix):
        self.base = base
        self.tau_days = tau_days
        self.times = times
        self.name = f"{base.name}+decay"

    def fit(self, train: sp.csr_matrix) -> "TimeDecay":
        times = self.times.tocsr().astype(np.float64)
        if times.shape != train.shape or times.nnz != train.nnz:
            raise ValueError("times must cover exactly the interactions in train.")
        self.base.fit(train)
        w = times.copy()
        if np.isfinite(self.tau_days):
            last = np.asarray(times.max(axis=1).todense()).ravel()
            rows = np.repeat(np.arange(times.shape[0]), np.diff(times.indptr))
            w.data = np.exp(-(last[rows] - times.data) / (self.tau_days * DAY))
        else:
            w.data[:] = 1.0
        self.Xw_ = w
        # EASE keeps its item matrix in B_, ItemKNN in sim_
        self.W_ = self.base.B_ if hasattr(self.base, "B_") else self.base.sim_
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.Xw_[users] @ self.W_)
