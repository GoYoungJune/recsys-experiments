import numpy as np
import pandas as pd
import pytest
import torch

from recsys.ctr.data import Encoder, temporal_split
from recsys.ctr.metrics import Platt, auc, calibration_ratio, cluster_weights, ece, logloss
from recsys.ctr.models import CTRModel, fit, predict


def toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user": rng.integers(1, 60, n), "movie": rng.integers(1, 40, n), "ts": rng.permutation(n),
        "gender": rng.choice(["F", "M"], n), "age": rng.choice([18, 25, 35], n), "occupation": rng.integers(0, 5, n),
        "zip3": rng.choice(["100", "200"], n), "year5": rng.choice(["1990", "1995"], n),
        "genres": rng.choice(["Drama", "Comedy|Drama", "Action|Sci-Fi|Thriller"], n)})
    # liking depends on an interaction: women like Drama, men like Action
    like = ((df.gender == "F") & df.genres.str.contains("Drama")) | ((df.gender == "M") & df.genres.str.contains("Action"))
    df["rating"] = np.where(like, 5, 1)
    df.loc[rng.random(n) < 0.1, "rating"] = 3
    return df


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200).astype(float)
    p = np.round(rng.random(200), 1)  # many ties
    pos, neg = p[y == 1], p[y == 0]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert auc(y, p) == pytest.approx(brute)
    w = rng.integers(0, 3, 200).astype(float)  # integer weights == duplicated rows
    rep = np.repeat(np.arange(200), w.astype(int))
    assert auc(y, p, w) == pytest.approx(auc(y[rep], p[rep]))


def test_logloss_ece_calibration():
    y = np.array([1, 0, 1, 0.])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert logloss(y, p) == pytest.approx(-np.mean(np.log([0.8, 0.8, 0.6, 0.6])))
    assert ece(np.array([1, 0.]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert ece(np.array([1, 1.]), np.array([0.25, 0.25])) == pytest.approx(0.75)
    assert calibration_ratio(y, p) == pytest.approx(1.0)


def test_platt_fixes_shifted_probabilities_without_changing_order():
    rng = np.random.default_rng(2)
    true = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < true).astype(float)
    z = np.log(true / (1 - true))
    biased = 1 / (1 + np.exp(-(2 * z + 1)))  # over-confident and shifted
    cal = Platt().fit(y, biased)
    fixed = cal.transform(biased)
    assert ece(y, fixed) < ece(y, biased) / 3
    assert auc(y, fixed) == pytest.approx(auc(y, biased))


def test_cluster_weights_resample_whole_users():
    groups = np.array([1, 1, 1, 2, 3, 3])
    for w in cluster_weights(groups, 20, seed=0):
        assert w[0] == w[1] == w[2] and w[4] == w[5]
        assert w[[0, 3, 4]].sum() == 3


def test_split_is_temporal_and_vocab_train_only():
    parts = temporal_split(toy())
    assert parts["train"].ts.max() < parts["valid"].ts.min() <= parts["valid"].ts.max() < parts["test"].ts.min()
    assert all((p.rating != 3).all() for p in parts.values())
    enc = Encoder().fit(parts["train"])
    test = parts["test"].copy()
    test.loc[test.index[0], "user"] = 99999  # never seen in train
    e = enc.transform(test)
    assert e.single[0, 0] == 0 and (e.genres >= 0).all()


@pytest.mark.parametrize("kind", ["lr", "fm", "deepfm", "dcnv2"])
def test_models_train_and_interaction_models_learn_interaction(kind):
    parts = temporal_split(toy(6000))
    enc = Encoder().fit(parts["train"])
    tr, va = enc.transform(parts["train"]), enc.transform(parts["valid"])
    m = CTRModel(kind, enc.field_sizes, enc.n_genres, dim=8)
    info = fit(m, tr, va, lr=1e-2, batch=512, epochs=15, log=lambda *_: None)
    p = predict(m, va)
    assert p.shape == (len(va.label),) and np.all((p >= 0) & (p <= 1))
    a = auc(va.label, p)
    if kind == "lr":
        assert a < 0.75  # gender x genre interaction is not linearly separable by additive effects
    else:
        assert a > 0.95
    assert info["best_epoch"] >= 1


def test_training_is_reproducible():
    parts = temporal_split(toy())
    enc = Encoder().fit(parts["train"])
    tr, va = enc.transform(parts["train"]), enc.transform(parts["valid"])
    ps = []
    for _ in range(2):
        m = CTRModel("deepfm", enc.field_sizes, enc.n_genres, dim=8)
        fit(m, tr, va, epochs=2, batch=512, log=lambda *_: None)
        ps.append(predict(m, va))
    assert np.allclose(ps[0], ps[1])
