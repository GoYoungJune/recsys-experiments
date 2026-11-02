"""Amazon ESCI (Shopping Queries Dataset): US small-version test queries as a reranking task."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# official ESCI gains: Exact 1, Substitute 0.1, Complement 0.01, Irrelevant 0
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def load_examples(esci_dir: str | Path) -> pd.DataFrame:
    """US, small version (the Task 1 reranking set). Columns: query_id, query, product_id, esci_label, split, gain."""
    ex = pd.read_parquet(Path(esci_dir) / "examples.parquet")
    ex = ex[(ex["product_locale"] == "us") & (ex["small_version"] == 1)]
    ex = ex[["query_id", "query", "product_id", "esci_label", "split"]].reset_index(drop=True)
    ex["gain"] = ex["esci_label"].map(GAINS)
    return ex


def sample_queries(ex: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n test queries drawn once with a fixed seed; all their judged candidates are kept."""
    ids = np.sort(ex.loc[ex["split"] == "test", "query_id"].unique())
    keep = np.random.default_rng(seed).choice(ids, size=min(n, len(ids)), replace=False)
    out = ex[ex["query_id"].isin(keep)]
    return out.sort_values(["query_id", "product_id"], kind="mergesort").reset_index(drop=True)


def load_product_text(esci_dir: str | Path, product_ids: set[str] | None = None) -> pd.Series:
    """product_id -> "title (brand)" for US products. Same text goes to every ranker."""
    # filter inside pyarrow so only the needed rows (and no descriptions) are ever loaded into memory
    cond = ds.field("product_locale") == "us"
    if product_ids is not None:
        cond = cond & ds.field("product_id").isin(sorted(product_ids))
    table = ds.dataset(Path(esci_dir) / "products.parquet").to_table(
        columns=["product_id", "product_title", "product_brand"], filter=cond
    )
    df = table.to_pandas()
    df = df.drop_duplicates("product_id")
    title = df["product_title"].fillna("").str.strip()
    brand = df["product_brand"].fillna("").str.strip()
    text = np.where(brand != "", title + " (" + brand + ")", title)
    return pd.Series(text, index=df["product_id"].to_numpy())


def groups(df: pd.DataFrame) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """[(query_id, query, row indices, gains)] in query_id order."""
    out = []
    for qid, g in df.groupby("query_id", sort=True):
        out.append((int(qid), g["query"].iloc[0], g.index.to_numpy(), g["gain"].to_numpy()))
    return out
