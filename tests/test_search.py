import numpy as np

from recsys.search.metrics import ndcg_at_k, rrf
from recsys.search.rankers import BM25


def test_ndcg_perfect_and_worst():
    gains = np.array([0.0, 1.0, 0.1, 0.01])
    assert np.isclose(ndcg_at_k(gains, gains), 1.0)          # scores = gains -> ideal order
    assert ndcg_at_k(-gains, gains) < 1.0
    assert ndcg_at_k(np.ones(3), np.zeros(3)) == 0.0          # no positive gain -> 0, not nan


def test_ndcg_ties_are_pessimistic():
    gains = np.array([1.0, 0.0])
    tied = ndcg_at_k(np.zeros(2), gains)                      # constant scores must not win by luck
    assert np.isclose(tied, ndcg_at_k(np.array([0.0, 1.0]), gains))


def test_rrf_prefers_items_ranked_high_by_both():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    fused = rrf([a, b])
    assert fused.argmax() == 1                                # rank 2 and 1 beats rank 1 and 3


def test_bm25_rewards_matching_rare_terms():
    corpus = ["red shoe", "blue shoe", "red hat", "shoe rack", "green shoe"]
    bm = BM25(corpus)
    s = bm.score("red shoe", ["red shoe", "blue shoe", "green hat"])
    assert s[0] > s[1] > s[2]
    assert s[2] == 0.0
