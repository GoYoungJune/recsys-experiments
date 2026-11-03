"""Part 5: CTR prediction on ml-1m following ctr/PREREG.md (LR, FM, DeepFM, DCN-v2; calibration; cold users).

Outputs go to results/ctr/: results.md, results.json, calibration.png, cold_warm.png, run.log

    python scripts/download_data.py ml-1m
    python scripts/run_ctr.py
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
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.ctr.data import ID_FIELDS, SINGLE_FIELDS, Encoder, load_ml1m, temporal_split  # noqa: E402
from recsys.ctr.metrics import Platt, calibration_ratio, ci, cluster_weights, ece, logloss  # noqa: E402
from recsys.ctr.models import CTRModel, fit, predict  # noqa: E402

SEED = 42
N_BOOT = 1000
KINDS = ["lr", "fm", "deepfm", "dcnv2"]
NAMES = {"lr": "LR", "fm": "FM", "deepfm": "DeepFM", "dcnv2": "DCN-v2"}
OUT = ROOT / "results" / "ctr"
torch.set_num_threads(8)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc_fast(inv: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUC given precomputed unique-score inverse indices (same formula as metrics.auc)."""
    pos = np.bincount(inv, weights=w * y)
    neg = np.bincount(inv, weights=w * (1 - y))
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def inv_of(p):
    return np.unique(p, return_inverse=True)[1]


def train_kind(kind, enc, tr, va):
    t = time.time()
    m = CTRModel(kind, enc.field_sizes, enc.n_genres, dim=16, seed=SEED)
    info = fit(m, tr, va, lr=1e-3, batch=4096, epochs=30, patience=2, seed=SEED, log=log)
    info["train_sec"] = round(time.time() - t, 1)
    info["n_params"] = int(sum(p.numel() for p in m.parameters()))
    return m, info


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = temporal_split(load_ml1m(ROOT / "data" / "ml-1m"))
    train_users = set(parts["train"]["user"])
    stats = {k: {"rows": len(v), "pos_rate": float(v["label"].mean()),
                 "cold_row_share": float((~v["user"].isin(train_users)).mean()),
                 "start": str(np.datetime64(int(v["ts"].min()), "s")), "end": str(np.datetime64(int(v["ts"].max()), "s"))}
             for k, v in parts.items()}
    log(f"Data: {stats}")

    enc = Encoder(SINGLE_FIELDS).fit(parts["train"])
    tr, va, te = (enc.transform(parts[k]) for k in ("train", "valid", "test"))
    cold = ~np.isin(te.user, list(train_users))

    log("1) Train four models (fixed hyper-parameters, early stopping on valid logloss)")
    models, infos, p_va, p_te = {}, {}, {}, {}
    for kind in KINDS:
        log(f"   {NAMES[kind]}")
        models[kind], infos[kind] = train_kind(kind, enc, tr, va)
        p_va[kind], p_te[kind] = predict(models[kind], va), predict(models[kind], te)
    best_deep = min(("deepfm", "dcnv2"), key=lambda k: infos[k]["best_valid_logloss"])
    log(f"   best deep model by valid logloss: {NAMES[best_deep]}")

    platt = Platt().fit(va.label, p_va[best_deep])
    p_cal = platt.transform(p_te[best_deep])
    log(f"   Platt on valid: a={platt.a:.3f} b={platt.b:.3f}")
    # post-hoc (not pre-registered): recalibrate on the valid rows of returning users only, which match the
    # test mix better (test is 80% returning users, valid only 23%)
    va_warm = np.isin(va.user, list(train_users))
    platt_w = Platt().fit(va.label[va_warm], p_va[best_deep][va_warm])
    p_cal_w = platt_w.transform(p_te[best_deep])
    log(f"   Platt on valid returning users only (post-hoc): a={platt_w.a:.3f} b={platt_w.b:.3f}")

    log("2) Exploratory: best deep model without ID fields")
    noid_fields = [f for f in SINGLE_FIELDS if f not in ID_FIELDS]
    enc_n = Encoder(noid_fields).fit(parts["train"])
    tr_n, va_n, te_n = (enc_n.transform(parts[k]) for k in ("train", "valid", "test"))
    m_n = CTRModel(best_deep, enc_n.field_sizes, enc_n.n_genres, dim=16, seed=SEED)
    info_n = fit(m_n, tr_n, va_n, seed=SEED, log=log)
    p_noid = predict(m_n, te_n)

    log(f"3) User-cluster bootstrap on test ({N_BOOT} resamples)")
    y = te.label.astype(np.float64)
    preds = {**p_te, "platt": p_cal, "platt_warm": p_cal_w, "noid": p_noid}
    invs = {k: inv_of(v) for k, v in preds.items()}
    inv_cold = {k: inv_of(preds[k][cold]) for k in (best_deep, "noid")}
    inv_warm = {k: inv_of(preds[k][~cold]) for k in (best_deep, "noid")}
    boot = {f"auc_{k}": [] for k in preds} | {f"ll_{k}": [] for k in preds} | {f"ece_{k}": [] for k in preds}
    boot |= {f"auc_cold_{k}": [] for k in inv_cold} | {f"auc_warm_{k}": [] for k in inv_warm}
    for w in cluster_weights(te.user, N_BOOT, seed=SEED):
        for k, p in preds.items():
            boot[f"auc_{k}"].append(auc_fast(invs[k], y, w))
            boot[f"ll_{k}"].append(logloss(y, p, w))
            boot[f"ece_{k}"].append(ece(y, p, w))
        for k in inv_cold:
            boot[f"auc_cold_{k}"].append(auc_fast(inv_cold[k], y[cold], w[cold]))
            boot[f"auc_warm_{k}"].append(auc_fast(inv_warm[k], y[~cold], w[~cold]))
    boot = {k: np.array(v) for k, v in boot.items()}
    ones = np.ones(len(y))

    def point(k):
        p = preds[k]
        return {"AUC": auc_fast(invs[k], y, ones), "LogLoss": logloss(y, p), "ECE": ece(y, p),
                "calib_ratio": calibration_ratio(y, p), "AUC_ci": ci(boot[f"auc_{k}"]),
                "LogLoss_ci": ci(boot[f"ll_{k}"]), "ECE_ci": ci(boot[f"ece_{k}"])}

    test = {k: point(k) for k in preds}

    def diff(a_key, b_key, metric="auc"):
        d = boot[f"{metric}_{a_key}"] - boot[f"{metric}_{b_key}"]
        pt = test[a_key]["AUC" if metric == "auc" else metric.upper()] - test[b_key]["AUC" if metric == "auc" else metric.upper()]
        return {"diff": pt, **ci(d), "share_boot_gt0": float((d > 0).mean())}

    C1 = diff("fm", "lr")
    C2 = diff(best_deep, "fm")
    C3 = diff("platt", best_deep, "ece")
    C3_warm = diff("platt_warm", best_deep, "ece")
    va_stats = {"valid_pos_rate_new": float(va.label[~va_warm].mean()), "valid_pos_rate_returning": float(va.label[va_warm].mean()),
                "valid_ece_new": ece(va.label[~va_warm].astype(float), p_va[best_deep][~va_warm]),
                "valid_ece_returning": ece(va.label[va_warm].astype(float), p_va[best_deep][va_warm]),
                "valid_calib_ratio_new": calibration_ratio(va.label[~va_warm], p_va[best_deep][~va_warm]),
                "valid_calib_ratio_returning": calibration_ratio(va.label[va_warm], p_va[best_deep][va_warm])}
    c4_pt = {k: {"cold": auc_fast(inv_cold[k], y[cold], ones[cold]), "warm": auc_fast(inv_warm[k], y[~cold], ones[~cold])}
             for k in inv_cold}
    C4 = {"cold": c4_pt[best_deep]["cold"], "warm": c4_pt[best_deep]["warm"],
          "diff": c4_pt[best_deep]["cold"] - c4_pt[best_deep]["warm"],
          **ci(boot[f"auc_cold_{best_deep}"] - boot[f"auc_warm_{best_deep}"])}
    noid = {"cold": c4_pt["noid"]["cold"], "warm": c4_pt["noid"]["warm"],
            "cold_minus_full": c4_pt["noid"]["cold"] - c4_pt[best_deep]["cold"],
            "cold_minus_full_ci": ci(boot["auc_cold_noid"] - boot[f"auc_cold_{best_deep}"]),
            "warm_minus_full": c4_pt["noid"]["warm"] - c4_pt[best_deep]["warm"],
            "warm_minus_full_ci": ci(boot["auc_warm_noid"] - boot[f"auc_warm_{best_deep}"]),
            "valid_logloss": info_n["best_valid_logloss"]}
    for name, c in (("C1 FM-LR AUC", C1), ("C2 deep-FM AUC", C2), ("C3 ECE platt-raw", C3), ("C4 AUC cold-warm", C4)):
        log(f"   {name}: {c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")

    log("4) Writing results")
    plot_calibration(y, {NAMES[best_deep]: p_te[best_deep], f"{NAMES[best_deep]} + Platt (fit on valid)": p_cal,
                         "LR": p_te["lr"]}, OUT / "calibration.png")
    plot_cold_warm(c4_pt, boot, best_deep, cold.sum(), (~cold).sum(), OUT / "cold_warm.png")
    res = {"data": stats, "cold_rows_test": int(cold.sum()), "train": infos, "best_deep": best_deep,
           "platt": {"a": platt.a, "b": platt.b}, "platt_warm": {"a": platt_w.a, "b": platt_w.b},
           "test": test, "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C3_warm_posthoc": C3_warm, "valid_by_user_type": va_stats,
           "noid": noid, "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_markdown(res)
    log(f"Done in {time.time() - t0:.0f}s")


def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    mp = np.bincount(b, weights=p, minlength=bins) / np.maximum(n, 1)
    my = np.bincount(b, weights=y, minlength=bins) / np.maximum(n, 1)
    keep = n >= 50
    return mp[keep], my[keep]


def plot_calibration(y, series, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], ":", color="#999999", label="Perfect calibration")
    styles = [("#0072B2", "o-"), ("#D55E00", "s-"), ("#777777", "^--")]
    for (name, p), (c, st) in zip(series.items(), styles):
        mp, my = reliability(y, p)
        ax.plot(mp, my, st, color=c, label=f"{name} (ECE {ece(y, p):.3f})", ms=4)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed like rate (bin)")
    ax.set_title("Calibration on the test period")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cold_warm(pt, boot, best_deep, n_cold, n_warm, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(2)
    for j, (k, label, color) in enumerate(((best_deep, f"{NAMES[best_deep]} (with user/movie IDs)", "#0072B2"),
                                           ("noid", f"{NAMES[best_deep]} without IDs", "#E69F00"))):
        m = np.array([pt[k]["cold"], pt[k]["warm"]])
        lo = np.array([np.percentile(boot[f"auc_cold_{k}"], 2.5), np.percentile(boot[f"auc_warm_{k}"], 2.5)])
        hi = np.array([np.percentile(boot[f"auc_cold_{k}"], 97.5), np.percentile(boot[f"auc_warm_{k}"], 97.5)])
        ax.bar(x + (j - 0.5) * 0.36, m, 0.36, yerr=[m - lo, hi - m], capsize=4, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"New users in test\n(not in train, {n_cold:,} rows)", f"Returning users\n({n_warm:,} rows)"])
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, None)
    ax.set_title("AUC for new vs returning users (95% user-cluster CI)", pad=26)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(r):
    t = lambda rows, h: tabulate(rows, headers=h, tablefmt="github", disable_numparse=True)
    f = lambda c: f"{c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
    L = ["# Results: ml-1m CTR prediction (part 5)", "",
         "Auto-generated by `scripts/run_ctr.py`, following `ctr/PREREG.md`. Seed 42, CPU, user-cluster bootstrap 1,000.", "",
         "## Data (temporal split; rating 3 removed after splitting)", "",
         t([[k, v["rows"], f"{v['pos_rate']:.4f}", f"{v['cold_row_share']:.1%}", v["start"][:10], v["end"][:10]]
            for k, v in r["data"].items()], ["split", "rows", "positive rate", "rows of users not in train", "from", "to"]), "",
         "## Training", "",
         t([[NAMES[k], i["n_params"], i["best_epoch"], len(i["history"]), f"{i['best_valid_logloss']:.4f}", i["train_sec"]]
            for k, i in r["train"].items()], ["model", "parameters", "best epoch", "epochs run", "valid logloss", "seconds"]), "",
         f"Best deep model by valid logloss: **{NAMES[r['best_deep']]}**. "
         f"Platt scaling fit on valid: a = {r['platt']['a']:.3f}, b = {r['platt']['b']:.3f}.", "",
         "## Test results (95% user-cluster bootstrap CI)", ""]
    rows = []
    labels = {**NAMES, "platt": f"{NAMES[r['best_deep']]} + Platt", "noid": f"{NAMES[r['best_deep']]} without IDs (exploratory)",
              "platt_warm": f"{NAMES[r['best_deep']]} + Platt on returning users of valid (post-hoc)"}
    for k in ["lr", "fm", "deepfm", "dcnv2", "platt", "platt_warm", "noid"]:
        m = r["test"][k]
        rows.append([labels[k], f"{m['AUC']:.4f}", f"[{m['AUC_ci']['ci_low']:.4f}, {m['AUC_ci']['ci_high']:.4f}]",
                     f"{m['LogLoss']:.4f}", f"{m['ECE']:.4f}", f"[{m['ECE_ci']['ci_low']:.4f}, {m['ECE_ci']['ci_high']:.4f}]",
                     f"{m['calib_ratio']:.3f}"])
    L += [t(rows, ["model", "AUC", "AUC 95% CI", "LogLoss", "ECE", "ECE 95% CI", "mean pred / positive rate"]), ""]
    L += ["## Hypotheses", "",
          t([["C1 FM - LR (AUC)", f(r["C1"]), "CI low > 0"],
             [f"C2 {NAMES[r['best_deep']]} - FM (AUC)", f(r["C2"]), "CI low > 0"],
             [f"C3 ECE after Platt - before ({NAMES[r['best_deep']]})", f(r["C3"]), "CI high < 0"],
             [f"C4 AUC new users - returning ({NAMES[r['best_deep']]})",
              f"{r['C4']['diff']:+.4f} [{r['C4']['ci_low']:+.4f}, {r['C4']['ci_high']:+.4f}] "
              f"(new {r['C4']['cold']:.4f}, returning {r['C4']['warm']:.4f})", "CI high < 0"]],
            ["hypothesis", "estimate [95% CI]", "confirmed if"]), "",
          "## Exploratory: without user/movie ID fields", "",
          t([["new users", f"{r['noid']['cold']:.4f}", f"{r['noid']['cold_minus_full']:+.4f}",
              f"[{r['noid']['cold_minus_full_ci']['ci_low']:+.4f}, {r['noid']['cold_minus_full_ci']['ci_high']:+.4f}]"],
             ["returning users", f"{r['noid']['warm']:.4f}", f"{r['noid']['warm_minus_full']:+.4f}",
              f"[{r['noid']['warm_minus_full_ci']['ci_low']:+.4f}, {r['noid']['warm_minus_full_ci']['ci_high']:+.4f}]"]],
            ["test rows", "AUC without IDs", "minus model with IDs", "95% CI"]), ""]
    v = r["valid_by_user_type"]
    L += ["## Post-hoc (NOT pre-registered): why did Platt scaling hurt?", "",
          t([["new users (not in train)", f"{v['valid_pos_rate_new']:.4f}", f"{v['valid_ece_new']:.4f}", f"{v['valid_calib_ratio_new']:.3f}"],
             ["returning users", f"{v['valid_pos_rate_returning']:.4f}", f"{v['valid_ece_returning']:.4f}",
              f"{v['valid_calib_ratio_returning']:.3f}"]],
            [f"valid rows ({NAMES[r['best_deep']]})", "positive rate", "ECE", "mean pred / positive rate"]), "",
          f"Platt fit on valid returning users only: a = {r['platt_warm']['a']:.3f}, b = {r['platt_warm']['b']:.3f}; "
          f"test ECE after - before: {f(r['C3_warm_posthoc'])}", ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
