from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


class ItemKNN(Recommender):
    """Item-based kNN with shrunk cosine similarity.

    sim(i, j) = |U_i ∩ U_j| / (||x_i|| * ||x_j|| + shrink)
    The shrink term pulls similarities that rest on only a few co-occurrences toward 0.
    Only the top `neighbors` similarities are kept per column (item j), the rest set to 0.
    score(u, j) = sum_i x_ui * sim(i, j)
    """

    name = "ItemKNN"

    def __init__(self, neighbors: int = 100, shrink: float = 0.0):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, train: sp.csr_matrix) -> "ItemKNN":
        self.X_ = train.tocsr().astype(np.float64)
        co = (self.X_.T @ self.X_).toarray()          # item-item co-occurrence (dense: catalog is small)
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0.0)
        n_items = sim.shape[0]
        if self.neighbors < n_items:
            # zero everything except the `neighbors` largest entries in each column
            drop = np.argpartition(sim, n_items - self.neighbors, axis=0)[: n_items - self.neighbors]
            np.put_along_axis(sim, drop, 0.0, axis=0)
        self.sim_ = sim
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.sim_)
