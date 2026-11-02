from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


class Popularity(Recommender):
    """Non-personalized baseline: score = number of positives an item received in train."""

    name = "Popularity"

    def fit(self, train: sp.csr_matrix) -> "Popularity":
        self.item_counts_ = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.item_counts_, (len(users), 1))
