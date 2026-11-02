import numpy as np
import pytest

from recsys.metrics import coverage_at_k, ndcg_at_k, recall_at_k, target_ranks


def test_ranks_and_values():
    scores = np.array([[0.9, 0.5, 0.1, 0.3],
                       [0.1, 0.2, 0.3, 0.4]])
    ranks = target_ranks(scores, np.array([1, 0]))
    assert ranks.tolist() == [2, 4]
    assert recall_at_k(ranks, 2).tolist() == [1.0, 0.0]
    assert ndcg_at_k(ranks, 2) == pytest.approx([1 / np.log2(3), 0.0])
    assert ndcg_at_k(np.array([1]), 10) == pytest.approx([1.0])


def test_ties_are_pessimistic():
    # all-equal scores must give the worst rank, not rank 1
    scores = np.zeros((1, 5))
    assert target_ranks(scores, np.array([2])).tolist() == [5]
    # tie with one other item -> ranked below it
    scores = np.array([[0.7, 0.7, 0.1]])
    assert target_ranks(scores, np.array([0])).tolist() == [2]


def test_masked_items_do_not_count():
    scores = np.array([[np.inf * -1, 0.2, 0.1]])
    assert target_ranks(scores, np.array([2])).tolist() == [2]


def test_coverage():
    topk = np.array([[0, 1], [1, 2]])
    assert coverage_at_k(topk, 10) == pytest.approx(0.3)
