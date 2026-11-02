import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_df():
    """Random implicit-feedback log with some sparse users/items and duplicate timestamps."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        n = rng.integers(2, 25)
        items = rng.choice(80, size=n, replace=False)
        ts = rng.integers(0, 10, size=n)  # few distinct timestamps -> many ties
        rows += [(u, int(i), int(t)) for i, t in zip(items, ts)]
    return pd.DataFrame(rows, columns=["userId", "movieId", "timestamp"])
