"""Full-ranking evaluation (no negative sampling)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .metrics import coverage_at_k, ndcg_at_k, recall_at_k, target_ranks
from .models.base import Recommender, mask_seen, top_k


def evaluate(
    model: Recommender,
    seen: sp.csr_matrix,
    targets: np.ndarray,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 1024,
) -> dict:
    """Rank *all* items for every user, after masking already-seen items with -inf.

    Sampled evaluation (target vs 100 random negatives) is avoided on purpose: it can change the
    ordering of models (Krichene & Rendle, 2020).

    Returns mean metrics plus per-user arrays (needed for paired bootstrap and segment analysis).
    """
    n_users = seen.shape[0]
    ranks = np.empty(n_users, dtype=np.int64)
    topk_all = np.empty((n_users, max(ks)), dtype=np.int64)
    for start in range(0, n_users, batch_size):
        users = np.arange(start, min(start + batch_size, n_users))
        scores = mask_seen(model.score(users).astype(np.float64), seen, users)
        ranks[users] = target_ranks(scores, targets[users])
        topk_all[users] = top_k(scores, max(ks))

    out: dict = {"ranks": ranks, "topk": topk_all, "per_user": {}}
    for k in ks:
        r, n = recall_at_k(ranks, k), ndcg_at_k(ranks, k)
        out["per_user"][f"Recall@{k}"] = r
        out["per_user"][f"NDCG@{k}"] = n
        out[f"Recall@{k}"] = float(r.mean())
        out[f"NDCG@{k}"] = float(n.mean())
        out[f"Coverage@{k}"] = coverage_at_k(topk_all[:, :k], seen.shape[1])
    return out
