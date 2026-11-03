"""Analysis of a real multi-arm randomized email experiment (Hillstrom 2008): tests, uplift, policy value.

Arms are coded 0 = Control (no e-mail), 1 = Mens e-mail, 2 = Womens e-mail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

CONTROL, MENS, WOMENS = 0, 1, 2
ARM_NAMES = {CONTROL: "Control", MENS: "Mens", WOMENS: "Womens"}
_SEGMENT_TO_ARM = {"No E-Mail": CONTROL, "Mens E-Mail": MENS, "Womens E-Mail": WOMENS}
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]


# --- data ---------------------------------------------------------------------------------------
def load_hillstrom(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arm"] = df["segment"].map(_SEGMENT_TO_ARM).astype(int)
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    """Pre-treatment features only. history_segment is a binned copy of history, so it is left out."""
    cats = pd.get_dummies(df[CATEGORICAL], dtype=float)
    return np.hstack([df[NUMERIC].to_numpy(dtype=float), cats.to_numpy()])


def stratified_split(arm: np.ndarray, frac: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split indices 50:50 within each arm so both halves keep the 1:1:1 design."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for a in np.unique(arm):
        idx = rng.permutation(np.flatnonzero(arm == a))
        cut = int(round(len(idx) * frac))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


# --- checks and tests ---------------------------------------------------------------------------
def srm_test(counts: np.ndarray, ratios: np.ndarray | None = None) -> tuple[float, float]:
    """Chi-square goodness of fit of arm sizes against the designed ratios. Returns (chi2, p)."""
    counts = np.asarray(counts, dtype=float)
    ratios = np.full(len(counts), 1 / len(counts)) if ratios is None else np.asarray(ratios, dtype=float)
    expected = counts.sum() * ratios / ratios.sum()
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return chi2, float(stats.chi2.sf(chi2, df=len(counts) - 1))


def smd(x_treat: np.ndarray, x_control: np.ndarray) -> float:
    """Standardized mean difference with the pooled SD; |SMD| < 0.1 is the usual balance threshold."""
    pooled = np.sqrt((np.var(x_treat, ddof=1) + np.var(x_control, ddof=1)) / 2)
    return float((np.mean(x_treat) - np.mean(x_control)) / pooled) if pooled > 0 else 0.0


def diff_in_means(y1: np.ndarray, y0: np.ndarray, alpha: float = 0.05) -> dict:
    """mean(y1) - mean(y0) with Welch standard error, z test and normal CI (n is in the thousands)."""
    y1, y0 = np.asarray(y1, dtype=float), np.asarray(y0, dtype=float)
    m1, m0 = y1.mean(), y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = (m1 - m0) / se
    q = stats.norm.ppf(1 - alpha / 2)
    return {"mean_t": float(m1), "mean_c": float(m0), "diff": float(m1 - m0), "se": float(se), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))), "ci_low": float(m1 - m0 - q * se), "ci_high": float(m1 - m0 + q * se),
            "rel": float((m1 - m0) / m0) if m0 > 0 else float("nan"), "n_t": len(y1), "n_c": len(y0)}


def interaction_test(y: np.ndarray, arm: np.ndarray, group: np.ndarray, treat: int) -> dict:
    """Effect of `treat` vs Control in group == 1 minus the same effect in group == 0.

    The four cells are independent samples, so Var(difference of effects) is the sum of the four
    cell variances of the means.
    """
    out = {}
    for g in (1, 0):
        out[g] = diff_in_means(y[(arm == treat) & (group == g)], y[(arm == CONTROL) & (group == g)])
    d = out[1]["diff"] - out[0]["diff"]
    se = np.sqrt(out[1]["se"] ** 2 + out[0]["se"] ** 2)
    z = d / se
    return {"effect_g1": out[1], "effect_g0": out[0], "diff": float(d), "se": float(se), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))), "ci_low": float(d - 1.959964 * se), "ci_high": float(d + 1.959964 * se)}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# --- uplift model -------------------------------------------------------------------------------
class TLearner:
    """One outcome model per arm; uplift of arm a = p(y | a, x) - p(y | Control, x)."""

    def __init__(self, make_model: Callable[[], object]):
        self.make_model = make_model

    def fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> "TLearner":
        self.models_ = {a: self.make_model().fit(X[arm == a], y[arm == a]) for a in np.unique(arm)}
        return self

    def predict_uplift(self, X: np.ndarray) -> dict[int, np.ndarray]:
        p = {a: m.predict_proba(X)[:, 1] for a, m in self.models_.items()}
        return {a: p[a] - p[CONTROL] for a in p if a != CONTROL}


def best_arm_policy(uplift: dict[int, np.ndarray]) -> np.ndarray:
    """Send each customer the e-mail with the larger predicted uplift (ties -> Mens)."""
    return np.where(uplift[WOMENS] > uplift[MENS], WOMENS, MENS)


def budget_policy(uplift: dict[int, np.ndarray], frac: float) -> np.ndarray:
    """Top `frac` customers by max predicted uplift get their better e-mail; the rest get nothing."""
    score = np.maximum(uplift[MENS], uplift[WOMENS])
    n_send = int(round(frac * len(score)))
    policy = np.full(len(score), CONTROL)
    if n_send > 0:
        top = np.argsort(-score, kind="stable")[:n_send]
        policy[top] = best_arm_policy(uplift)[top]
    return policy


# --- policy value -------------------------------------------------------------------------------
def policy_value(y: np.ndarray, arm: np.ndarray, policy: np.ndarray, w: np.ndarray | None = None) -> float:
    """Stratified Hajek estimate of E[Y] if everyone were treated by `policy`.

    V = sum_a P(policy = a) * mean(Y | T = a, policy = a). Valid because T was randomized
    independently of x. Optional `w` are bootstrap counts per row.
    """
    w = np.ones(len(y)) if w is None else w
    total = w.sum()
    v = 0.0
    for a in np.unique(policy):
        want = policy == a
        match = want & (arm == a)
        if w[match].sum() == 0:
            return float("nan")
        v += w[want].sum() / total * (w[match] * y[match]).sum() / w[match].sum()
    return float(v)


def bootstrap_policy_values(y: np.ndarray, arm: np.ndarray, policies: dict[str, np.ndarray],
                            n_boot: int = 10_000, seed: int = 42, chunk: int = 500) -> dict[str, np.ndarray]:
    """Bootstrap distribution of every policy's value using the SAME resampled customers (paired).

    Each replicate is a vector of resample counts per customer; values are computed with matrix
    products so 10,000 replicates over ~32,000 customers stay fast and memory-bounded.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    # per policy and arm: indicator of "policy wants a", and of "wants a and got a", and y on matches
    cols, layout = [], {}
    for name, pol in policies.items():
        layout[name] = []
        for a in np.unique(pol):
            want = (pol == a).astype(float)
            match = want * (arm == a)
            layout[name].append(len(cols))
            cols += [want, match, match * y]
    M = np.stack(cols, axis=1)                                    # (n, 3 * #(policy, arm))
    out = {name: np.empty(n_boot) for name in policies}
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(b, n))
        W = np.zeros((b, n))
        for r in range(b):
            W[r] = np.bincount(idx[r], minlength=n)
        S = W @ M                                                 # (b, columns)
        for name, starts in layout.items():
            v = np.zeros(b)
            for c in starts:
                with np.errstate(invalid="ignore", divide="ignore"):
                    v += S[:, c] / n * S[:, c + 2] / S[:, c + 1]
            out[name][s:s + b] = v
    return out


def summarize_boot(x: np.ndarray, null: float = 0.0) -> dict:
    """Point-free summary of a bootstrap distribution: 95% percentile CI and a two-sided p for `null`."""
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    p = 2 * min((x <= null).mean(), (x >= null).mean())
    return {"ci_low": float(lo), "ci_high": float(hi), "p_boot": float(min(1.0, max(p, 1 / (len(x) + 1)))),
            "n_valid": int(len(x))}
