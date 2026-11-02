"""Loading, filtering and temporal leave-one-out splitting of MovieLens implicit feedback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

POSITIVE_THRESHOLD = 4.0
MIN_COUNT = 5


def load_positives(ratings_path: str | Path, threshold: float = POSITIVE_THRESHOLD) -> pd.DataFrame:
    """Keep only ratings >= threshold as positive (implicit) interactions."""
    df = pd.read_csv(ratings_path)
    df = df[df["rating"] >= threshold]
    return df[["userId", "movieId", "timestamp"]].reset_index(drop=True)


def iterative_filter(df: pd.DataFrame, min_count: int = MIN_COUNT) -> pd.DataFrame:
    """Repeatedly drop users and items with < min_count positives until both conditions hold.

    One pass is not enough: dropping sparse items can push a user below the threshold and vice versa.
    """
    while True:
        item_cnt = df["movieId"].map(df["movieId"].value_counts())
        user_cnt = df["userId"].map(df["userId"].value_counts())
        keep = (item_cnt >= min_count) & (user_cnt >= min_count)
        if keep.all():
            return df.reset_index(drop=True)
        df = df[keep]


@dataclass
class Split:
    """Temporal leave-one-out split. Row u of `train` and entries u of valid/test refer to the same user."""

    train: sp.csr_matrix          # (n_users, n_items) 0/1
    valid_items: np.ndarray       # (n_users,) item index held out for validation
    test_items: np.ndarray        # (n_users,) item index held out for test
    user_ids: np.ndarray          # index -> original userId
    item_ids: np.ndarray          # index -> original movieId

    @property
    def n_users(self) -> int:
        return self.train.shape[0]

    @property
    def n_items(self) -> int:
        return self.train.shape[1]

    def train_plus_valid(self) -> sp.csr_matrix:
        """Train matrix with each user's validation item added back (used for the final refit)."""
        add = sp.csr_matrix(
            (np.ones(self.n_users, dtype=np.float32), (np.arange(self.n_users), self.valid_items)),
            shape=self.train.shape,
        )
        out = (self.train + add).tocsr()
        out.data[:] = 1.0
        return out


def leave_one_out(df: pd.DataFrame) -> Split:
    """Per user, sorted by (userId, timestamp, movieId): last -> test, second-to-last -> valid, rest -> train.

    movieId is a tie-breaker so that equal timestamps always give the same split.
    """
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)
    user_ids, u_idx = np.unique(df["userId"].to_numpy(), return_inverse=True)
    item_ids, i_idx = np.unique(df["movieId"].to_numpy(), return_inverse=True)

    # position from the end within each user (0 = last interaction)
    pos_from_end = df.groupby("userId").cumcount(ascending=False).to_numpy()
    if (df.groupby("userId").size() < 3).any():
        raise ValueError("Every user needs at least 3 interactions for train/valid/test.")

    n_users, n_items = len(user_ids), len(item_ids)
    test_items = np.empty(n_users, dtype=np.int64)
    valid_items = np.empty(n_users, dtype=np.int64)
    test_items[u_idx[pos_from_end == 0]] = i_idx[pos_from_end == 0]
    valid_items[u_idx[pos_from_end == 1]] = i_idx[pos_from_end == 1]

    tr = pos_from_end >= 2
    train = sp.csr_matrix(
        (np.ones(tr.sum(), dtype=np.float32), (u_idx[tr], i_idx[tr])), shape=(n_users, n_items)
    )
    train.sum_duplicates()
    train.data[:] = 1.0
    return Split(train, valid_items, test_items, user_ids, item_ids)


def load_split(raw_dir: str | Path) -> tuple[Split, pd.DataFrame]:
    """Full pipeline: positives -> iterative filter -> leave-one-out. Returns (split, filtered dataframe)."""
    df = iterative_filter(load_positives(Path(raw_dir) / "ratings.csv"))
    return leave_one_out(df), df
