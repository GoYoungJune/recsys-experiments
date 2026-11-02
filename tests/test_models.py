import numpy as np
import pytest

from recsys.data import iterative_filter, leave_one_out
from recsys.evaluate import evaluate
from recsys.models import BPRMF, EASE, ItemKNN, Popularity, TimeDecay


@pytest.fixture
def split(toy_df):
    return leave_one_out(iterative_filter(toy_df, min_count=5))


MODELS = [
    lambda: Popularity(),
    lambda: ItemKNN(neighbors=10, shrink=10),
    lambda: EASE(l2=10),
    lambda: BPRMF(dim=8, epochs=3),
]


@pytest.mark.parametrize("make", MODELS)
def test_never_recommends_seen_items(split, make):
    model = make().fit(split.train)
    users = np.arange(split.n_users)
    k = 10
    recs = model.recommend(users, split.train, k)
    assert recs.shape == (split.n_users, k)
    seen = split.train.toarray()
    assert (seen[users[:, None], recs] == 0).all()
    assert all(len(set(r)) == k for r in recs)


def test_ease_closed_form(split):
    m = EASE(l2=5).fit(split.train)
    assert np.allclose(np.diag(m.B_), 0)
    X = split.train.toarray()
    P = np.linalg.inv(X.T @ X + 5 * np.eye(X.shape[1]))
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)
    assert np.allclose(m.B_, B)


def test_itemknn_keeps_top_neighbors(split):
    m = ItemKNN(neighbors=3).fit(split.train)
    assert ((m.sim_ != 0).sum(axis=0) <= 3).all()


def test_bpr_early_stopping_restores_best(split):
    calls = iter([0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    m = BPRMF(dim=8, epochs=100, eval_every=1, patience=4).fit(split.train, eval_fn=lambda _: next(calls))
    assert m.best_epoch_ == 2
    assert len(m.history_) == 6  # best at eval 2, then 4 non-improving evals


def test_bpr_is_reproducible(split):
    a = BPRMF(dim=8, epochs=2).fit(split.train).score(np.arange(5))
    b = BPRMF(dim=8, epochs=2).fit(split.train).score(np.arange(5))
    assert np.allclose(a, b)


def test_evaluate_keys(split):
    res = evaluate(Popularity().fit(split.train), split.train, split.valid_items)
    for key in ["Recall@10", "NDCG@10", "Coverage@10", "Recall@20", "NDCG@20"]:
        assert 0.0 <= res[key] <= 1.0
    assert res["per_user"]["NDCG@10"].shape == (split.n_users,)


def test_time_decay_inf_equals_base(split):
    base = EASE(l2=10).fit(split.train).score(np.arange(split.n_users))
    decayed = TimeDecay(EASE(l2=10), np.inf, split.train_times).fit(split.train).score(np.arange(split.n_users))
    assert np.allclose(base, decayed)


def test_time_decay_weights_recent_history_more(split):
    m = TimeDecay(ItemKNN(neighbors=10), 1e-6, split.train_times).fit(split.train)  # ~only the latest counts
    w = m.Xw_.toarray()
    assert np.allclose(w.max(axis=1), 1.0)                       # latest interaction weighs 1
    assert ((w > 0) <= (split.train.toarray() > 0)).all()        # never adds unseen items
