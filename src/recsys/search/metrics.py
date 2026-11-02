"""Graded NDCG for reranking a fixed candidate list per query."""
from __future__ import annotations

import numpy as np


def ranked_gains(scores: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Gains in ranked order. Ties are broken pessimistically: among equal scores the lowest gain comes first."""
    order = np.lexsort((gains, -scores))
    return gains[order]


def dcg(gains: np.ndarray, k: int) -> float:
    """Linear gains as in the ESCI Task 1 metric (E 1.0, S 0.1, C 0.01, I 0)."""
    g = gains[:k]
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 10) -> float:
    """NDCG@k over the query's judged candidates. Queries with no positive gain get 0 (they carry no signal)."""
    ideal = dcg(np.sort(gains)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg(ranked_gains(scores, gains), k) / ideal


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009): sum of 1 / (k + rank) over the input rankings."""
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out
