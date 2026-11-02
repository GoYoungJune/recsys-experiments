import numpy as np

from recsys.abtest.analysis import cuped, diff_in_means, obrien_fleming_bounds, sample_size_per_arm, srm_pvalue
from recsys.abtest.sim import Population, draw


def test_diff_in_means_recovers_lift():
    rng = np.random.default_rng(0)
    x, y, t = draw(Population(), 200_000, rng, lift=0.10)
    d, se, z = diff_in_means(y, t)
    control_mean = y[~t].mean()
    assert abs(d[0] / control_mean - 0.10) < 3 * se[0] / control_mean + 0.01


def test_cuped_is_unbiased_and_reduces_variance():
    rng = np.random.default_rng(1)
    x, y, t = draw(Population(corr=0.8), 5_000, rng, lift=0.0, reps=300)
    d_raw = diff_in_means(y, t)[0]
    d_cup = diff_in_means(cuped(y, x), t)[0]
    assert abs(d_cup.mean()) < 3 * d_cup.std() / np.sqrt(300)
    assert d_cup.var() < 0.8 * d_raw.var()


def test_sample_size_formula_matches_textbook():
    # sd 1, mde 0.1, alpha 0.05, power 0.8 -> 2 * (1.96 + 0.8416)^2 / 0.01 = 1570
    assert sample_size_per_arm(1.0, 0.1) == 1570


def test_obrien_fleming_bounds_shape():
    b = obrien_fleming_bounds(5, 0.05, np.random.default_rng(0), sims=50_000)
    assert np.all(np.diff(b) < 0)               # strict early, close to 2 at the end
    assert 1.96 < b[-1] < 2.2


def test_srm_flags_imbalance_only():
    assert srm_pvalue(np.array([5000]), np.array([5000]))[0] > 0.9
    assert srm_pvalue(np.array([5000]), np.array([5300]))[0] < 0.01
