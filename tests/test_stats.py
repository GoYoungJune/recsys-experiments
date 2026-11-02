import numpy as np

from recsys.stats import bootstrap_ci, paired_bootstrap


def test_detects_clear_difference():
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 0.5, size=500)
    a = b + 0.1 + rng.normal(0, 0.05, size=500)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["diff"] > 0
    assert r["ci_low"] > 0
    assert r["p_value"] < 0.01
    assert abs(r["rel_improvement"] - r["diff"] / b.mean()) < 1e-12


def test_no_difference_not_significant():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, size=500)
    b = a + rng.normal(0, 0.1, size=500)
    b = b - (b.mean() - a.mean())  # identical means
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["p_value"] > 0.5


def test_ci_contains_mean():
    x = np.random.default_rng(3).uniform(size=300)
    lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < x.mean() < hi
