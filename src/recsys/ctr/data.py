"""ml-1m as a CTR task: temporal split, train-only vocabularies, categorical + multi-valued (genre) fields."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SINGLE_FIELDS = ["user", "movie", "gender", "age", "occupation", "zip3", "year5"]
ID_FIELDS = ["user", "movie"]
MAX_GENRES = 6


def load_ml1m(raw: str | Path) -> pd.DataFrame:
    """Join ratings with user and movie attributes. Returns one row per rating (rating 3 kept here)."""
    raw = Path(raw)
    kw = dict(sep="::", engine="python", encoding="latin-1")
    r = pd.read_csv(raw / "ratings.dat", names=["user", "movie", "rating", "ts"], **kw)
    u = pd.read_csv(raw / "users.dat", names=["user", "gender", "age", "occupation", "zip"], **kw)
    m = pd.read_csv(raw / "movies.dat", names=["movie", "title", "genres"], **kw)
    u["zip3"] = u["zip"].astype(str).str[:3]
    year = m["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    m["year5"] = (year // 5 * 5).fillna(-1).astype(int).astype(str)
    df = r.merge(u[["user", "gender", "age", "occupation", "zip3"]], on="user").merge(
        m[["movie", "genres", "year5"]], on="movie")
    return df


def temporal_split(df: pd.DataFrame, fracs=(0.8, 0.1)) -> dict[str, pd.DataFrame]:
    """Global time split on (ts, user, movie); rating-3 rows are removed AFTER splitting, labels 4-5 -> 1, 1-2 -> 0."""
    df = df.sort_values(["ts", "user", "movie"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    c1, c2 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    parts = {"train": df.iloc[:c1], "valid": df.iloc[c1:c2], "test": df.iloc[c2:]}
    out = {}
    for k, p in parts.items():
        p = p[p["rating"] != 3].copy()
        p["label"] = (p["rating"] >= 4).astype(np.float32)
        out[k] = p.reset_index(drop=True)
    return out


@dataclass
class Encoded:
    single: np.ndarray      # (n, n_fields) int64, index 0 of every field = unknown
    genres: np.ndarray      # (n, MAX_GENRES) int64, 0 = padding
    label: np.ndarray       # (n,) float32
    user: np.ndarray        # (n,) original user id, for cluster bootstrap and cold/warm segments


class Encoder:
    """Vocabularies built on train only; values never seen in train map to 0 ("unknown")."""

    def __init__(self, fields: list[str] = SINGLE_FIELDS):
        self.fields = list(fields)

    def fit(self, train: pd.DataFrame) -> "Encoder":
        self.vocab = {f: {v: i + 1 for i, v in enumerate(sorted(train[f].astype(str).unique()))} for f in self.fields}
        genres = sorted({g for s in train["genres"] for g in s.split("|")})
        self.genre_vocab = {g: i + 1 for i, g in enumerate(genres)}
        return self

    @property
    def field_sizes(self) -> list[int]:
        return [len(self.vocab[f]) + 1 for f in self.fields]

    @property
    def n_genres(self) -> int:
        return len(self.genre_vocab) + 1

    def transform(self, df: pd.DataFrame) -> Encoded:
        single = np.stack([df[f].astype(str).map(self.vocab[f]).fillna(0).to_numpy(np.int64) for f in self.fields], 1)
        genres = np.zeros((len(df), MAX_GENRES), dtype=np.int64)
        for row, s in enumerate(df["genres"].to_numpy()):
            ids = [self.genre_vocab.get(g, 0) for g in s.split("|")][:MAX_GENRES]
            genres[row, :len(ids)] = ids
        return Encoded(single, genres, df["label"].to_numpy(np.float32), df["user"].to_numpy())
