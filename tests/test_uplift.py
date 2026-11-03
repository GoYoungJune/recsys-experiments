import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from recsys.uplift.analysis import (
    CONTROL, MENS, WOMENS, TLearner, best_arm_policy, bootstrap_policy_values, budget_policy,
    diff_in_means, holm, interaction_test, policy_value, smd, srm_test, stratified_split,
)


def synthetic(n=30_000, seed=0):
    """Randomized 3-arm data: Mens e-mail helps only g=1 (+0.10), Womens helps only g=0 (+0.10)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    arm = rng.integers(0, 3, n)
    base = 0.10
    p = {CONTROL: np.full(n, base), MENS: base + 0.10 * g, WOMENS: base + 0.10 * (1 - g)}
    u = rng.uniform(size=n)
    pot = {a: (u < p[a]).astype(float) for a in p}  # potential outcomes share the same noise
    y = np.choose(arm, [pot[CONTROL], pot[MENS], pot[WOMENS]])
    return g, arm, y, p


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]).tolist() == [0.5]


def test_srm():
    assert srm_test([21307, 21387, 21306])[1] > 0.5
    assert srm_test([10000, 10000, 9000])[1] < 1e-6


def test_smd_and_diff():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.5, 1, 5000), rng.normal(0, 1, 5000)
    assert smd(a, b) == pytest.approx(0.5, abs=0.05)
    r = diff_in_means(a, b)
    assert r["ci_low"] < 0.5 < r["ci_high"] and r["p"] < 1e-10


def test_interaction_detects_planted_effect():
    g, arm, y, _ = synthetic()
    r = interaction_test(y, arm, g, MENS)
    assert r["diff"] == pytest.approx(0.10, abs=0.03) and r["p"] < 1e-6
    # no interaction for an e-mail whose effect does not depend on a random, irrelevant group
    noise_group = np.random.default_rng(5).integers(0, 2, len(y))
    assert interaction_test(y, arm, noise_group, MENS)["p"] > 0.01


def test_policy_value_is_unbiased_on_randomized_data():
    g, arm, y, p = synthetic()
    oracle = np.where(g == 1, MENS, WOMENS)
    true_value = np.mean(np.choose(oracle, [p[CONTROL], p[MENS], p[WOMENS]]))  # 0.20
    assert policy_value(y, arm, oracle) == pytest.approx(true_value, abs=0.01)
    assert policy_value(y, arm, np.full(len(y), CONTROL)) == pytest.approx(0.10, abs=0.01)


def test_bootstrap_is_paired_and_reproducible():
    g, arm, y, _ = synthetic(n=6000)
    pols = {"oracle": np.where(g == 1, MENS, WOMENS), "mens": np.full(len(y), MENS)}
    a = bootstrap_policy_values(y, arm, pols, n_boot=300)
    b = bootstrap_policy_values(y, arm, pols, n_boot=300)
    assert np.allclose(a["oracle"], b["oracle"])
    assert np.isclose(np.median(a["oracle"]), policy_value(y, arm, pols["oracle"]), atol=0.01)
    assert (a["oracle"] - a["mens"] > 0).mean() > 0.99


def test_tlearner_recovers_heterogeneity_and_policies():
    g, arm, y, _ = synthetic()
    X = g[:, None].astype(float)
    up = TLearner(lambda: LogisticRegression()).fit(X, arm, y).predict_uplift(X)
    pol = best_arm_policy(up)
    assert (pol == np.where(g == 1, MENS, WOMENS)).mean() > 0.99
    half = budget_policy(up, 0.5)
    assert (half != CONTROL).sum() == len(y) // 2


def test_stratified_split_disjoint_and_balanced():
    arm = np.repeat([0, 1, 2], [1001, 1000, 999])
    tr, te = stratified_split(arm)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(arm)
    for a in range(3):
        assert abs((arm[tr] == a).sum() - (arm[te] == a).sum()) <= 1
