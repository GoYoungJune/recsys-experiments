from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


class EASE(Recommender):
    """Embarrassingly Shallow Autoencoder (Steck, 2019).

    Learns an item-item weight matrix B that reconstructs X from X itself:
        min_B ||X - XB||^2 + l2 * ||B||^2   s.t. diag(B) = 0
    Closed form: P = (X^T X + l2 I)^-1,  B_ij = -P_ij / P_jj,  B_jj = 0.
    The diag constraint stops the trivial solution "predict an item from itself".
    """

    name = "EASE"

    def __init__(self, l2: float = 500.0):
        self.l2 = l2

    def fit(self, train: sp.csr_matrix) -> "EASE":
        self.X_ = train.tocsr().astype(np.float64)
        G = (self.X_.T @ self.X_).toarray()
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)        # divides column j by P_jj
        np.fill_diagonal(B, 0.0)
        self.B_ = B
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.B_)
