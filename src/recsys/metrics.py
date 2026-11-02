"""Top-K metrics for the single-held-out-item (leave-one-out) setting."""
from __future__ import annotations

import numpy as np


def target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each user's target item, with pessimistic tie handling.

    rank = #{items with score >= target score}; ties are counted as ranked *above* the target,
    so a model that gives every item the same score gets the worst rank, not a lucky one.
    """
    target_scores = scores[np.arange(len(targets)), targets][:, None]
    return (scores >= target_scores).sum(axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-user Recall@K. With one held-out item this equals HitRate@K."""
    return (ranks <= k).astype(np.float64)


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-user NDCG@K. With one relevant item the ideal DCG is 1, so NDCG = 1/log2(rank+1)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    return np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)


def coverage_at_k(topk: np.ndarray, n_items: int) -> float:
    """Share of the catalog that appears in at least one user's Top-K list."""
    return len(np.unique(topk)) / n_items
