import numpy as np
import pandas as pd

from recsys.data import iterative_filter, leave_one_out


def test_filter_conditions_hold(toy_df):
    out = iterative_filter(toy_df, min_count=5)
    assert len(out) > 0
    assert out["userId"].value_counts().min() >= 5
    assert out["movieId"].value_counts().min() >= 5


def test_filter_needs_multiple_passes():
    # Dense core: users 0-5 x items 0-5. User 6 has items 0,1,2 + item 100 + rare item 101 (5 items).
    # Item 100 is liked by users 0-3 and 6 (5 users).
    # Pass 1 drops item 101 -> user 6 falls to 4 -> pass 2 drops user 6 -> item 100 falls to 4 -> pass 3 drops it.
    # A single simultaneous pass would wrongly keep user 6 and item 100.
    rows = [(u, i, 0) for u in range(6) for i in range(6)]
    rows += [(6, i, 0) for i in (0, 1, 2, 100, 101)]
    rows += [(u, 100, 0) for u in range(4)]
    df = pd.DataFrame(rows, columns=["userId", "movieId", "timestamp"])
    out = iterative_filter(df, min_count=5)
    assert set(out["userId"]) == set(range(6))
    assert set(out["movieId"]) == set(range(6))


def test_split_has_no_leakage(toy_df):
    s = leave_one_out(iterative_filter(toy_df, min_count=5))
    tr = s.train.toarray()
    users = np.arange(s.n_users)
    assert (tr[users, s.test_items] == 0).all(), "test item leaked into train"
    assert (tr[users, s.valid_items] == 0).all(), "valid item leaked into train"
    assert (s.test_items != s.valid_items).all()


def test_split_is_temporal_and_deterministic(toy_df):
    df = iterative_filter(toy_df, min_count=5)
    s1 = leave_one_out(df)
    s2 = leave_one_out(df.sample(frac=1.0, random_state=1))  # shuffled input -> same split
    assert (s1.test_items == s2.test_items).all()
    assert (s1.valid_items == s2.valid_items).all()
    assert (s1.train != s2.train).nnz == 0
    # test item is the last one after sorting by (timestamp, movieId)
    for u_idx, uid in enumerate(s1.user_ids):
        last = df[df.userId == uid].sort_values(["timestamp", "movieId"]).iloc[-1]["movieId"]
        assert s1.item_ids[s1.test_items[u_idx]] == last


def test_train_plus_valid_adds_exactly_one_per_user(toy_df):
    s = leave_one_out(iterative_filter(toy_df, min_count=5))
    tv = s.train_plus_valid()
    added = np.asarray(tv.sum(axis=1)).ravel() - np.asarray(s.train.sum(axis=1)).ravel()
    assert (added == 1).all()
    assert set(np.unique(tv.data)) == {1.0}
    assert (tv.toarray()[np.arange(s.n_users), s.valid_items] == 1).all()
    assert (tv.toarray()[np.arange(s.n_users), s.test_items] == 0).all()


def test_train_times_match_train(toy_df):
    s = leave_one_out(iterative_filter(toy_df, min_count=5))
    assert s.train_times.nnz == s.train.nnz
    tvt = s.train_plus_valid_times()
    assert tvt.nnz == s.train_plus_valid().nnz
    assert np.allclose(tvt.toarray()[np.arange(s.n_users), s.valid_items], s.valid_times)
