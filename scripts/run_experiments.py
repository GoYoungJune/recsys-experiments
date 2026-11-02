"""Run the full experiment: grid search -> refit on train+valid -> test -> bootstrap -> segments -> report.

Outputs go to results/: results.md, results.json, ndcg_overall.png, ndcg_by_segment.png
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allow running without `pip install -e .`

from recsys.data import load_split  # noqa: E402
from recsys.evaluate import evaluate  # noqa: E402
from recsys.models import BPRMF, EASE, ItemKNN, Popularity  # noqa: E402
from recsys.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

SEED = 42
K = 10
METRICS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "Coverage@10"]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
MODEL_ORDER = ["Popularity", "ItemKNN", "EASE", "BPR-MF"]
COLORS = {"Popularity": "#999999", "ItemKNN": "#E69F00", "EASE": "#0072B2", "BPR-MF": "#009E73"}

GRIDS = {
    "Popularity": [{}],
    "ItemKNN": [{"neighbors": n, "shrink": s} for n, s in itertools.product([50, 100, 200, 400], [0, 10, 50])],
    "EASE": [{"l2": v} for v in [50, 100, 250, 500, 1000, 2000]],
    "BPR-MF": [{"dim": d, "weight_decay": w} for d, w in itertools.product([32, 64, 128], [1e-5, 1e-4, 1e-3, 1e-2])],
}
FACTORY = {"Popularity": Popularity, "ItemKNN": ItemKNN, "EASE": EASE, "BPR-MF": BPRMF}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------------------------------
def grid_search(split):
    """Pick hyper-parameters by validation NDCG@10 (train -> valid)."""
    ev = lambda m: evaluate(m, split.train, split.valid_items, ks=(K,))[f"NDCG@{K}"]
    grid_rows, best = [], {}
    for name, grid in GRIDS.items():
        for params in grid:
            t = time.time()
            model = FACTORY[name](**params)
            if name == "BPR-MF":
                model.fit(split.train, eval_fn=ev)
                score, extra = model.best_score_, {"best_epoch": model.best_epoch_}
            else:
                model.fit(split.train)
                score, extra = ev(model), {}
            grid_rows.append({"model": name, "params": params, "valid_NDCG@10": score, **extra})
            log(f"  {name:10s} {params} valid NDCG@10={score:.4f} {extra} ({time.time() - t:.1f}s)")
            if name not in best or score > best[name]["valid_NDCG@10"]:
                best[name] = {"params": params, "valid_NDCG@10": score, **extra}
    return grid_rows, best


def final_test(split, best):
    """Refit each model's best config on train+valid and evaluate on test (seen = train+valid)."""
    tv = split.train_plus_valid()
    results = {}
    for name in MODEL_ORDER:
        params = best[name]["params"]
        if name == "BPR-MF":
            model = BPRMF(**params, epochs=best[name]["best_epoch"]).fit(tv)  # fixed epochs, no peeking
        else:
            model = FACTORY[name](**params).fit(tv)
        res = evaluate(model, tv, split.test_items, ks=(10, 20))
        res["model"] = model
        res["ndcg_ci"] = bootstrap_ci(res["per_user"]["NDCG@10"])
        results[name] = res
        log(f"  {name:10s} test " + " ".join(f"{m}={res[m]:.4f}" for m in METRICS))
    return tv, results


def segment_analysis(split, tv, test_res, best_name, df):
    # Timing diagnostics per user (user order = sorted userId, same as split.user_ids)
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")
    ts = df.groupby("userId")["timestamp"]
    t_test = ts.transform("last")
    same_day_share = ((t_test - df["timestamp"]) < 86400).groupby(df["userId"]).mean().to_numpy()
    span_days = ((ts.max() - ts.min()) / 86400).to_numpy()
    hist_len = np.asarray(tv.sum(axis=1)).ravel().astype(int)
    # equal-sized quartiles by history length (rank first so tied lengths don't break qcut)
    seg = pd.qcut(pd.Series(hist_len).rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).to_numpy()
    item_pop = np.asarray(tv.sum(axis=0)).ravel()
    pop_rank = pd.Series(-item_pop).rank(method="min").to_numpy().astype(int)  # 1 = most popular
    target_pop_rank = pop_rank[split.test_items]

    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        m = seg == q
        row = {
            "segment": q,
            "n_users": int(m.sum()),
            "hist_min": int(hist_len[m].min()),
            "hist_max": int(hist_len[m].max()),
            "hist_median": float(np.median(hist_len[m])),
            "target_pop_rank_median": float(np.median(target_pop_rank[m])),
            # share of the user's positives logged within 24h before the test item (1.0 = one session)
            "same_day_share_median": float(np.median(same_day_share[m])),
            "history_span_days_median": float(np.median(span_days[m])),  # first -> last positive
        }
        for name in MODEL_ORDER:
            x = test_res[name]["per_user"]["NDCG@10"][m]
            row[f"{name}_NDCG@10"] = float(x.mean())
            row[f"{name}_ci"] = bootstrap_ci(x)
        bs = paired_bootstrap(
            test_res[best_name]["per_user"]["NDCG@10"][m], test_res["Popularity"]["per_user"]["NDCG@10"][m]
        )
        row["best_vs_pop"] = bs
        rows.append(row)
    return rows, hist_len, seg, target_pop_rank


def qualitative(split, tv, test_res, hist_len, movies):
    """EASE Top-10 for the user whose history length is closest to the median."""
    med = np.median(hist_len)
    u = int(np.argmin(np.abs(hist_len - med)))  # argmin -> smallest index on ties (deterministic)
    ease = test_res["EASE"]["model"]
    recs = ease.recommend(np.array([u]), tv, K)[0]
    title = movies.set_index("movieId")["title"]
    target = int(split.test_items[u])
    return {
        "user_index": u,
        "userId": int(split.user_ids[u]),
        "history_length": int(hist_len[u]),
        "test_movie": title[split.item_ids[target]],
        "test_rank_ease": int(test_res["EASE"]["ranks"][u]),
        "test_in_top10": bool(target in recs),
        "top10": [title[split.item_ids[i]] for i in recs],
    }


# ---------------------------------------------------------------------------------------------------
def plot_overall(test_res, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    means = [test_res[n]["NDCG@10"] for n in MODEL_ORDER]
    cis = [test_res[n]["ndcg_ci"] for n in MODEL_ORDER]
    err = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
    bars = ax.bar(MODEL_ORDER, means, yerr=err, capsize=6, color=[COLORS[n] for n in MODEL_ORDER])
    for b, m, (lo, hi) in zip(bars, means, cis):
        ax.text(b.get_x() + b.get_width() / 2, hi + 0.002, f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Test NDCG@10")
    ax.set_title("Test NDCG@10 by model (error bars: 95% bootstrap CI)")
    ax.set_ylim(0, max(hi for _, hi in cis) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_segments(seg_rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(seg_rows))
    w = 0.2
    top = 0.0
    for k, name in enumerate(MODEL_ORDER):
        means = np.array([r[f"{name}_NDCG@10"] for r in seg_rows])
        cis = np.array([r[f"{name}_ci"] for r in seg_rows])
        err = np.vstack([means - cis[:, 0], cis[:, 1] - means])
        ax.bar(x + (k - 1.5) * w, means, w, yerr=err, capsize=3, color=COLORS[name], label=name)
        top = max(top, cis[:, 1].max())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['segment']}\n{r['hist_min']}-{r['hist_max']} items\n(n={r['n_users']})" for r in seg_rows])
    ax.set_xlabel("User segment by training-history length (quartiles)")
    ax.set_ylabel("Test NDCG@10")
    ax.set_ylim(0, top * 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    # legend above the axes so it never covers bars
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("Test NDCG@10 by history-length segment (95% bootstrap CI)", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    split, df = load_split(RAW)
    movies = pd.read_csv(RAW / "movies.csv")
    data_stats = {
        "n_users": split.n_users,
        "n_items": split.n_items,
        "n_positives": int(len(df)),
        "n_train": int(split.train.nnz),
        "density": len(df) / (split.n_users * split.n_items),
    }
    log(f"Data: {data_stats}")

    log("1) Grid search (validation NDCG@10)")
    grid_rows, best = grid_search(split)
    best_name = max(MODEL_ORDER, key=lambda n: best[n]["valid_NDCG@10"])
    log(f"   best on validation: {best_name}")

    log("2) Refit on train+valid, evaluate on test")
    tv, test_res = final_test(split, best)

    log(f"3) Paired bootstrap: {best_name} vs others (per-user NDCG@10, 10,000 resamples)")
    pairs = [(best_name, n) for n in MODEL_ORDER if n != best_name]
    # extra pairs needed to judge the hypotheses: H1 (every personalized model vs Popularity), H2
    for extra in [("ItemKNN", "Popularity"), ("EASE", "Popularity"), ("BPR-MF", "Popularity"), ("BPR-MF", "ItemKNN")]:
        if extra not in pairs and extra[::-1] not in pairs:
            pairs.append(extra)
    boot = []
    for a, b in pairs:
        r = paired_bootstrap(test_res[a]["per_user"]["NDCG@10"], test_res[b]["per_user"]["NDCG@10"])
        boot.append({"a": a, "b": b, **r})
        log(f"   {a} - {b}: diff={r['diff']:+.4f} CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={fmt_p(r['p_value'])}")

    log("4) Segment analysis")
    seg_rows, hist_len, seg, target_pop_rank = segment_analysis(split, tv, test_res, best_name, df)

    log("5) Qualitative example (EASE)")
    qual = qualitative(split, tv, test_res, hist_len, movies)
    title = movies.set_index("movieId")["title"]
    # most recent 5 liked movies before the test item (by timestamp)
    u_df = df[df.userId == qual["userId"]].sort_values(["timestamp", "movieId"])
    qual["recent_history"] = [title[m] for m in u_df["movieId"].iloc[-6:-1]]

    log("6) Writing results")
    plot_overall(test_res, OUT / "ndcg_overall.png")
    plot_segments(seg_rows, OUT / "ndcg_by_segment.png")
    write_markdown(data_stats, grid_rows, best, best_name, test_res, boot, seg_rows, qual)
    json_out = {
        "data": data_stats,
        "grid": grid_rows,
        "best_params": best,
        "best_model_by_validation": best_name,
        "test": {
            n: {**{m: test_res[n][m] for m in METRICS}, "NDCG@10_ci95": test_res[n]["ndcg_ci"]} for n in MODEL_ORDER
        },
        "bootstrap": boot,
        "segments": seg_rows,
        "qualitative": qual,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Done in {time.time() - t0:.0f}s")


def write_markdown(data_stats, grid_rows, best, best_name, test_res, boot, seg_rows, qual):
    L = ["# Results", "", "Auto-generated by `scripts/run_experiments.py`. All numbers are from a single run (seed 42).", ""]
    L += ["## Data", "", tabulate([[k, f"{v:.4%}" if k == "density" else v] for k, v in data_stats.items()],
                                  headers=["stat", "value"], tablefmt="github"), ""]

    L += ["## Best hyper-parameters (validation NDCG@10)", ""]
    rows = [[n, json.dumps(best[n]["params"]), f"{best[n]['valid_NDCG@10']:.4f}", best[n].get("best_epoch", "")]
            for n in MODEL_ORDER]
    L += [tabulate(rows, headers=["model", "params", "valid NDCG@10", "best epoch"], tablefmt="github"), ""]
    L += [f"Best model on validation: **{best_name}**", ""]

    L += ["## Test results (refit on train+valid, full ranking over all unseen items)", ""]
    rows = [[n] + [f"{test_res[n][m]:.4f}" for m in METRICS]
            + [f"[{test_res[n]['ndcg_ci'][0]:.4f}, {test_res[n]['ndcg_ci'][1]:.4f}]"] for n in MODEL_ORDER]
    L += [tabulate(rows, headers=["model"] + METRICS + ["NDCG@10 95% CI"], tablefmt="github"), ""]

    L += ["## Paired bootstrap on per-user NDCG@10 (10,000 resamples)", ""]
    rows = [[f"{r['a']} vs {r['b']}", f"{r['diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
             f"{r['rel_improvement']:+.1%}", fmt_p(r["p_value"]), "yes" if r["p_value"] < 0.05 else "no"] for r in boot]
    L += [tabulate(rows, headers=["comparison", "diff", "95% CI", "relative", "p-value", "significant (0.05)"],
                   tablefmt="github", disable_numparse=True), ""]

    L += ["## Segments by training-history length (train+valid, quartiles)", ""]
    rows = [[r["segment"], f"{r['hist_min']}-{r['hist_max']}", r["n_users"], f"{r['target_pop_rank_median']:.0f}",
             f"{r['same_day_share_median']:.0%}", f"{r['history_span_days_median']:.1f}"]
            + [f"{r[f'{n}_NDCG@10']:.4f}" for n in MODEL_ORDER]
            + [f"{r['best_vs_pop']['diff']:+.4f} [{r['best_vs_pop']['ci_low']:+.4f}, {r['best_vs_pop']['ci_high']:+.4f}]"]
            for r in seg_rows]
    L += [tabulate(rows, headers=["segment", "history", "users", "median pop. rank of test item",
                                  "median share of history within 24h of test", "median history span (days)"]
                   + [f"{n} NDCG@10" for n in MODEL_ORDER] + [f"{best_name} - Popularity (95% CI)"],
                   tablefmt="github", disable_numparse=True), ""]
    L += ["Popularity rank 1 = most popular item in train+valid.", ""]

    L += ["## Qualitative example: EASE Top-10", ""]
    L += [f"- user {qual['userId']} (history length {qual['history_length']}, closest to median)",
          f"- recent liked movies: {'; '.join(qual['recent_history'])}",
          f"- held-out test movie: **{qual['test_movie']}** -> EASE rank {qual['test_rank_ease']}"
          f" ({'in' if qual['test_in_top10'] else 'not in'} Top-10)", ""]
    L += [tabulate([[i + 1, t] for i, t in enumerate(qual["top10"])], headers=["rank", "title"], tablefmt="github"), ""]

    L += ["## Full grid", ""]
    rows = [[g["model"], json.dumps(g["params"]), f"{g['valid_NDCG@10']:.4f}", g.get("best_epoch", "")] for g in grid_rows]
    L += [tabulate(rows, headers=["model", "params", "valid NDCG@10", "best epoch"], tablefmt="github"), ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
