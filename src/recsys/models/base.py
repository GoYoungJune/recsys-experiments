"""Common interface for all recommenders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

SEED = 42


def mask_seen(scores: np.ndarray, seen: sp.csr_matrix, users: np.ndarray) -> np.ndarray:
    """Set scores of items each user already interacted with to -inf (in place) and return them."""
    rows = seen[users]
    r, c = rows.nonzero()
    scores[r, c] = -np.inf
    return scores


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, sorted from best to worst."""
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


class Recommender(ABC):
    """fit(train) learns from a (n_users, n_items) 0/1 csr matrix; score(users) returns dense scores."""

    name: str = "Recommender"

    @abstractmethod
    def fit(self, train: sp.csr_matrix) -> "Recommender":
        ...

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Dense (len(users), n_items) score matrix; higher = more recommended."""

    def recommend(self, users: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
        """Top-k item indices per user, never including items in `seen`."""
        users = np.asarray(users)
        scores = mask_seen(self.score(users).astype(np.float64), seen, users)
        return top_k(scores, k)
