"""A/B testing study by simulation: power, CUPED, peeking, sample ratio mismatch (A1-A4).

Simulated data only (see src/recsys/abtest/sim.py). Every rate below is a Monte Carlo estimate with
a 95% Wilson interval. Outputs go to results/abtest/: results.md, results.json, peeking.png, cuped.png

    python scripts/run_abtest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.abtest.analysis import (  # noqa: E402
    cuped, diff_in_means, obrien_fleming_bounds, sample_size_per_arm, srm_pvalue,
)
from recsys.abtest.sim import Population, draw  # noqa: E402

SEED = 42
REPS = 4000            # simulated experiments per scenario
CHUNK = 250            # replicates drawn at once (memory)
ALPHA = 0.05
MDE_REL = 0.03         # minimum detectable effect: +3% of the control mean
LOOKS = 14             # daily looks over a two-week test
SRM_ALPHA = 0.001      # common practice: alarm only on very small p, since the check runs on every test
OUT = ROOT / "results" / "abtest"
Z = stats.norm.ppf(1 - ALPHA / 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a Monte Carlo rate k / n."""
    p, z = k / n, 1.959964
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(c - h), float(c + h)


def rate(hits: np.ndarray) -> dict:
    k, n = int(hits.sum()), len(hits)
    lo, hi = wilson(k, n)
    return {"rate": k / n, "ci": [lo, hi], "n": n}


def chunks(pop, n, rng, lift):
    for _ in range(REPS // CHUNK):
        yield draw(pop, n, rng, lift=lift, reps=CHUNK)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pop = Population()

    # population facts from one very large draw (the "historical data" an analyst would use)
    x_big, y_big, _ = draw(pop, 2_000_000, rng)
    mean_y, sd_y = float(y_big.mean()), float(y_big.std())
    rho = float(np.corrcoef(x_big[0], y_big[0])[0, 1])
    del x_big, y_big
    mde_abs = MDE_REL * mean_y
    n_raw = sample_size_per_arm(sd_y, mde_abs, ALPHA, 0.8)
    n_cuped = sample_size_per_arm(sd_y * np.sqrt(1 - rho ** 2), mde_abs, ALPHA, 0.8)
    facts = {"mean_Y": mean_y, "sd_Y": sd_y, "corr_X_Y_raw_scale": rho, "mde_abs": mde_abs,
             "n_per_arm_formula": n_raw, "n_per_arm_formula_cuped": n_cuped,
             "cuped_sample_saving": 1 - n_cuped / n_raw}
    log(f"Population: {facts}")
    n_total = 2 * n_raw

    log("A1/A2) power and type I error, raw vs CUPED, at the raw-formula sample size")
    res = {}
    for label, lift in [("null", 0.0), ("effect", MDE_REL)]:
        rej_raw, rej_cup, d_raw, d_cup = [], [], [], []
        for x, y, t in chunks(pop, n_total, rng, lift):
            dr, _, zr = diff_in_means(y, t)
            dc, _, zc = diff_in_means(cuped(y, x), t)
            rej_raw.append(np.abs(zr) > Z)
            rej_cup.append(np.abs(zc) > Z)
            d_raw.append(dr)
            d_cup.append(dc)
        d_raw, d_cup = np.concatenate(d_raw), np.concatenate(d_cup)
        res[label] = {"raw": rate(np.concatenate(rej_raw)), "cuped": rate(np.concatenate(rej_cup)),
                      "var_ratio_cuped_over_raw": float(d_cup.var() / d_raw.var()),
                      "mean_diff_raw": float(d_raw.mean()), "mean_diff_cuped": float(d_cup.mean())}
        log(f"   {label}: reject raw={res[label]['raw']['rate']:.3f} cuped={res[label]['cuped']['rate']:.3f} "
            f"var ratio={res[label]['var_ratio_cuped_over_raw']:.3f}")
    # CUPED at its own (smaller) formula sample size should again give ~80% power
    hits = []
    for x, y, t in chunks(pop, 2 * n_cuped, rng, MDE_REL):
        hits.append(np.abs(diff_in_means(cuped(y, x), t)[2]) > Z)
    res["cuped_at_cuped_n"] = rate(np.concatenate(hits))
    log(f"   CUPED power at n={n_cuped}/arm: {res['cuped_at_cuped_n']['rate']:.3f}")

    log("A3) peeking: 14 daily looks, users arrive uniformly")
    bounds = obrien_fleming_bounds(LOOKS, ALPHA, rng)
    peek, crossed_by_look = {}, np.zeros(LOOKS)
    for label, lift in [("null", 0.0), ("effect", MDE_REL)]:
        any_naive, any_obf, final_only = [], [], []
        for x, y, t in chunks(pop, n_total, rng, lift):
            zs = []
            for k in range(1, LOOKS + 1):
                m = int(n_total * k / LOOKS)
                zs.append(diff_in_means(y[:, :m], t[:, :m])[2])
            zs = np.abs(np.stack(zs, axis=1))
            if label == "null":  # share of A/A tests that have crossed at least once by look k
                crossed_by_look += np.maximum.accumulate(zs > Z, axis=1).sum(axis=0)
            any_naive.append((zs > Z).any(axis=1))
            any_obf.append((zs >= bounds).any(axis=1))
            final_only.append(zs[:, -1] > Z)
        peek[label] = {"fixed_horizon": rate(np.concatenate(final_only)),
                       "naive_peeking": rate(np.concatenate(any_naive)),
                       "obrien_fleming": rate(np.concatenate(any_obf))}
        log(f"   {label}: fixed={peek[label]['fixed_horizon']['rate']:.3f} naive={peek[label]['naive_peeking']['rate']:.3f} "
            f"OBF={peek[label]['obrien_fleming']['rate']:.3f}")
    cum_naive = (crossed_by_look / REPS).tolist()

    log("A4) sample ratio mismatch: treatment silently loses heavy users")
    srm = {}
    for label, loss in [("clean", 0.0), ("bug", 0.2)]:
        est, alarm, false_pos = [], [], []
        for x, y, t in chunks(pop, n_total, rng, 0.0):
            # bug: 20% of treatment users in the top 10% of pre-period activity drop out of the logs
            heavy = x > np.quantile(x, 0.9, axis=1, keepdims=True)
            lost = t & heavy & (rng.random(t.shape) < loss)
            keep = ~lost
            y2 = np.where(keep, y, np.nan)
            n1, n0 = (t & keep).sum(axis=1), (~t).sum(axis=1)
            m1 = np.nansum(np.where(t, y2, np.nan), axis=1) / n1
            m0 = np.nansum(np.where(~t, y2, np.nan), axis=1) / n0
            v1 = np.nansum(np.where(t, (y2 - m1[:, None]) ** 2, np.nan), axis=1) / (n1 - 1)
            v0 = np.nansum(np.where(~t, (y2 - m0[:, None]) ** 2, np.nan), axis=1) / (n0 - 1)
            z = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
            est.append(m1 / m0 - 1)
            false_pos.append(np.abs(z) > Z)
            alarm.append(srm_pvalue(n1, n0) < SRM_ALPHA)
        est = np.concatenate(est)
        srm[label] = {"mean_relative_lift": float(est.mean()), "lift_ci": [float(np.percentile(est, 2.5)),
                      float(np.percentile(est, 97.5))], "false_positive": rate(np.concatenate(false_pos)),
                      "srm_alarm": rate(np.concatenate(alarm)),
                      "share_of_treatment_lost": loss * 0.1}
        log(f"   {label}: lift={srm[label]['mean_relative_lift']:+.4f} FP={srm[label]['false_positive']['rate']:.3f} "
            f"SRM alarm={srm[label]['srm_alarm']['rate']:.3f}")

    log("Writing results")
    plot_peeking(cum_naive, peek, OUT / "peeking.png")
    plot_cuped(res, facts, OUT / "cuped.png")
    out = {"settings": {"reps": REPS, "alpha": ALPHA, "mde_rel": MDE_REL, "looks": LOOKS, "srm_alpha": SRM_ALPHA,
                        "seed": SEED, "population": vars(pop)},
           "facts": facts, "power_cuped": res, "peeking": peek, "obf_bounds": bounds.tolist(),
           "peeking_curve_null": cum_naive, "srm": srm, "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(out)
    log(f"Done in {time.time() - t0:.0f}s")


def plot_peeking(curve, peek, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(range(1, LOOKS + 1), curve, marker="o", color="#D55E00", label="look every day, stop at first p < 0.05")
    ax.axhline(ALPHA, color="#555555", ls="--", lw=1, label="nominal 5%")
    ax.axhline(peek["null"]["obrien_fleming"]["rate"], color="#0072B2", lw=1.5,
               label=f"O'Brien-Fleming, 14 looks ({peek['null']['obrien_fleming']['rate']:.1%})")
    ax.set_xlabel("Day of the test (look number)")
    ax.set_ylabel("False positive rate so far (no true effect)")
    ax.set_ylim(0, max(curve) * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Peeking inflates false positives (simulated A/A tests)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cuped(res, facts, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["No effect: false positives", "+3% effect: power"]
    raw = [res["null"]["raw"]["rate"], res["effect"]["raw"]["rate"]]
    cup = [res["null"]["cuped"]["rate"], res["effect"]["cuped"]["rate"]]
    x = np.arange(2)
    ax.bar(x - 0.18, raw, 0.36, color="#999999", label="difference in means")
    ax.bar(x + 0.18, cup, 0.36, color="#0072B2", label="CUPED")
    for i in range(2):
        ax.text(x[i] - 0.18, raw[i] + 0.01, f"{raw[i]:.1%}", ha="center", fontsize=9)
        ax.text(x[i] + 0.18, cup[i] + 0.01, f"{cup[i]:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Same sample size ({facts['n_per_arm_formula']:,} per arm): CUPED keeps 5%, raises power")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(o):
    f, r, p, s = o["facts"], o["power_cuped"], o["peeking"], o["srm"]
    pct = lambda d: f"{d['rate']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
    L = ["# A/B testing simulation results", "",
         f"Auto-generated by `scripts/run_abtest.py`. Simulated data, {o['settings']['reps']:,} experiments per scenario, "
         f"seed {o['settings']['seed']}. Rates: Monte Carlo estimate [95% Wilson CI].", ""]
    L += ["## Population and sample size", "",
          tabulate([[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in f.items()],
                   headers=["fact", "value"], tablefmt="github", disable_numparse=True), ""]
    rows = [["false positive rate (no effect)", pct(r["null"]["raw"]), pct(r["null"]["cuped"])],
            [f"power (+{o['settings']['mde_rel']:.0%} effect)", pct(r["effect"]["raw"]), pct(r["effect"]["cuped"])],
            ["variance of the estimate, CUPED / raw", "1.000", f"{r['effect']['var_ratio_cuped_over_raw']:.3f}"]]
    L += [f"## A1/A2: raw difference in means vs CUPED (n = {f['n_per_arm_formula']:,} per arm)", "",
          tabulate(rows, headers=["", "difference in means", "CUPED"], tablefmt="github", disable_numparse=True), "",
          f"CUPED at its own formula size ({f['n_per_arm_formula_cuped']:,} per arm): power {pct(r['cuped_at_cuped_n'])}.", ""]
    rows = [[sc, pct(p[sc]["fixed_horizon"]), pct(p[sc]["naive_peeking"]), pct(p[sc]["obrien_fleming"])]
            for sc in ["null", "effect"]]
    L += [f"## A3: peeking ({o['settings']['looks']} equally spaced looks)", "",
          tabulate(rows, headers=["scenario", "look once at the end", "stop at first p < 0.05", "O'Brien-Fleming bounds"],
                   tablefmt="github", disable_numparse=True), "",
          "Row `null` = false positive rate; row `effect` = power. O'Brien-Fleming |z| bounds by look: "
          + ", ".join(f"{b:.2f}" for b in o["obf_bounds"]), ""]
    rows = [[sc, f"{s[sc]['share_of_treatment_lost']:.0%}", f"{s[sc]['mean_relative_lift']:+.4f}",
             f"[{s[sc]['lift_ci'][0]:+.4f}, {s[sc]['lift_ci'][1]:+.4f}]", pct(s[sc]["false_positive"]), pct(s[sc]["srm_alarm"])]
            for sc in ["clean", "bug"]]
    L += ["## A4: sample ratio mismatch (true effect = 0)", "",
          tabulate(rows, headers=["scenario", "treatment users lost", "mean estimated lift", "middle 95% of estimates",
                                  "false positive rate", f"SRM alarm rate (p < {o['settings']['srm_alpha']})"],
                   tablefmt="github", disable_numparse=True), ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
