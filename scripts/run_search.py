"""Search reranking on Amazon ESCI: BM25 -> dense -> hybrid -> cross-encoder -> zero-shot LLM.

Quality: graded NDCG@10 over each query's judged candidates, paired bootstrap over queries.
Cost: online latency per query (index building excluded), measured on a fixed subset.
Outputs go to results/search/: results.md, results.json, quality_vs_latency.png, ndcg_by_query_length.png

    python scripts/run_search.py
"""
from __future__ import annotations

import gc
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

from recsys.search.data import groups, load_examples, load_product_text, sample_queries  # noqa: E402
from recsys.search.metrics import ndcg_at_k, rrf  # noqa: E402
from recsys.search.rankers import BM25, CrossEncoder, Dense, PointwiseLLM  # noqa: E402
from recsys.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

SEED = 42
N_QUERIES = 2000          # LLM scoring cost bounds the sample; drawn once from the 8,956 US test queries
N_LATENCY = 200           # queries timed per ranker (after warm-up)
ESCI = ROOT / "data" / "esci"
OUT = ROOT / "results" / "search"
ORDER = ["BM25", "Dense", "Hybrid (RRF)", "Cross-encoder", "LLM"]
COLORS = {"BM25": "#999999", "Dense": "#0072B2", "Hybrid (RRF)": "#56B4E9", "Cross-encoder": "#E69F00", "LLM": "#CC79A7"}
# pre-registered comparisons (a, b): is a better than b?
HYPOTHESES = [
    ("S1", "Dense", "BM25", "dense retrieval beats lexical BM25"),
    ("S2", "Hybrid (RRF)", "BM25", "hybrid beats BM25"),
    ("S2", "Hybrid (RRF)", "Dense", "hybrid beats dense"),
    ("S3", "Cross-encoder", "Dense", "cross-encoder beats dense"),
    ("S4", "LLM", "Cross-encoder", "zero-shot 3B LLM does NOT beat the trained cross-encoder (expected diff <= 0)"),
]
LENGTH_BINS = [(1, 2, "1-2 words"), (3, 4, "3-4 words"), (5, 99, "5+ words")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def score_all(name, ranker, qgroups, df, text, use_ids=False):
    """Scores for every candidate row, one query at a time (the same call the latency benchmark times)."""
    scores = np.zeros(len(df))
    t = time.time()
    for k, (_, query, rows, _) in enumerate(qgroups):
        ids = df.loc[rows, "product_id"].tolist()
        scores[rows] = ranker.score(query, ids if use_ids else [text[i] for i in ids])
        if (k + 1) % 500 == 0:
            log(f"   {name}: {k + 1}/{len(qgroups)} queries ({time.time() - t:.0f}s)")
    return scores


def latency(ranker, qgroups, df, text, use_ids=False):
    """Online milliseconds per query on N_LATENCY queries, after 10 warm-up queries."""
    sub = qgroups[: N_LATENCY + 10]
    times = []
    for k, (_, query, rows, _) in enumerate(sub):
        ids = df.loc[rows, "product_id"].tolist()
        arg = ids if use_ids else [text[i] for i in ids]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        ranker.score(query, arg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if k >= 10:
            times.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ex = load_examples(ESCI)
    df = sample_queries(ex, N_QUERIES, SEED)
    corpus_ids = set(ex["product_id"])
    del ex
    corpus = load_product_text(ESCI, corpus_ids)
    df = df[df["product_id"].isin(corpus.index)].reset_index(drop=True)
    qgroups = groups(df)
    # queries whose candidates are all Irrelevant have ideal DCG 0 and carry no signal
    qgroups = [g for g in qgroups if g[3].max() > 0]
    data_stats = {"queries": len(qgroups), "candidates": int(sum(len(g[2]) for g in qgroups)),
                  "corpus_products_for_bm25": len(corpus),
                  "label_share": df["esci_label"].value_counts(normalize=True).round(4).to_dict()}
    log(f"Data: {data_stats}")

    log("1) BM25 (corpus statistics from all US small-version products)")
    bm25 = BM25(corpus.tolist())
    text = corpus[df["product_id"].unique()]      # only the candidates are needed from here on
    del corpus
    gc.collect()
    scores = {"BM25": score_all("BM25", bm25, qgroups, df, text)}
    lat = {"BM25": latency(bm25, qgroups, df, text)}

    log("2) Dense (bge-m3): index candidate documents offline, then score")
    dense = Dense()
    cand_ids = df["product_id"].unique().tolist()
    dense.index(cand_ids, [text[i] for i in cand_ids])
    scores["Dense"] = score_all("Dense", dense, qgroups, df, text, use_ids=True)
    lat["Dense"] = latency(dense, qgroups, df, text, use_ids=True)
    scores["Hybrid (RRF)"] = np.zeros(len(df))
    for _, _, rows, _ in qgroups:
        scores["Hybrid (RRF)"][rows] = rrf([scores["BM25"][rows], scores["Dense"][rows]])
    lat["Hybrid (RRF)"] = {k: lat["BM25"][k] + lat["Dense"][k] for k in lat["BM25"]}  # both run per query
    del dense
    torch.cuda.empty_cache()

    log("3) Cross-encoder (bge-reranker-v2-m3)")
    ce = CrossEncoder()
    scores["Cross-encoder"] = score_all("Cross-encoder", ce, qgroups, df, text)
    lat["Cross-encoder"] = latency(ce, qgroups, df, text)
    del ce
    torch.cuda.empty_cache()

    log("4) Zero-shot pointwise LLM (Qwen2.5-3B-Instruct)")
    llm = PointwiseLLM()
    scores["LLM"] = score_all("LLM", llm, qgroups, df, text)
    lat["LLM"] = latency(llm, qgroups, df, text)
    del llm
    torch.cuda.empty_cache()

    log("5) Metrics, bootstrap, segments")
    per_query = {n: np.array([ndcg_at_k(scores[n][rows], gains) for _, _, rows, gains in qgroups]) for n in ORDER}
    overall = {n: {"NDCG@10": float(per_query[n].mean()), "ci": bootstrap_ci(per_query[n]), **lat[n]} for n in ORDER}
    for n in ORDER:
        log(f"   {n:14s} NDCG@10={overall[n]['NDCG@10']:.4f} p50={overall[n]['p50_ms']:.1f}ms")
    tests = []
    for hid, a, b, desc in HYPOTHESES:
        r = paired_bootstrap(per_query[a], per_query[b])
        tests.append({"id": hid, "a": a, "b": b, "hypothesis": desc, **r})
        log(f"   {hid} {a} - {b}: {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={fmt_p(r['p_value'])}")
    qlen = np.array([len(q.split()) for _, q, _, _ in qgroups])
    segs = []
    for lo, hi, label in LENGTH_BINS:
        m = (qlen >= lo) & (qlen <= hi)
        row = {"segment": label, "queries": int(m.sum()), **{n: float(per_query[n][m].mean()) for n in ORDER}}
        row["LLM_vs_CE"] = paired_bootstrap(per_query["LLM"][m], per_query["Cross-encoder"][m])
        segs.append(row)

    log("6) Writing results")
    plot_quality_latency(overall, OUT / "quality_vs_latency.png")
    plot_segments(segs, OUT / "ndcg_by_query_length.png")
    write_markdown(data_stats, overall, tests, segs)
    (OUT / "results.json").write_text(json.dumps({
        "data": data_stats, "n_queries_sampled": N_QUERIES, "n_latency_queries": N_LATENCY, "seed": SEED,
        "overall": overall, "tests": tests, "segments": segs,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runtime_sec": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Done in {time.time() - t0:.0f}s")


def plot_quality_latency(overall, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for n in ORDER:
        o = overall[n]
        ax.errorbar(o["p50_ms"], o["NDCG@10"], yerr=[[o["NDCG@10"] - o["ci"][0]], [o["ci"][1] - o["NDCG@10"]]],
                    fmt="o", color=COLORS[n], capsize=4, markersize=8)
        ax.annotate(n, (o["p50_ms"], o["NDCG@10"]), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Online latency per query, p50 (ms, log scale)")
    ax.set_ylabel("NDCG@10 (ESCI gains)")
    ax.set_title("Quality vs latency (error bars: 95% bootstrap CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_segments(segs, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(segs))
    w = 0.8 / len(ORDER)
    for k, n in enumerate(ORDER):
        ax.bar(x + (k - (len(ORDER) - 1) / 2) * w, [s[n] for s in segs], w, color=COLORS[n], label=n)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['segment']}\n(n={s['queries']})" for s in segs])
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(min(s[n] for s in segs for n in ORDER) * 0.95, max(s[n] for s in segs for n in ORDER) * 1.03)
    ax.legend(ncol=len(ORDER), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    ax.set_title("NDCG@10 by query length (y axis does not start at 0)", pad=26)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(data_stats, overall, tests, segs):
    L = ["# Search reranking results (Amazon ESCI, US small version, test)", "",
         "Auto-generated by `scripts/run_search.py`. Single run, seed 42.", ""]
    L += ["## Data", "", tabulate([[k, v] for k, v in data_stats.items()], headers=["stat", "value"], tablefmt="github"), ""]
    rows = [[n, f"{overall[n]['NDCG@10']:.4f}", f"[{overall[n]['ci'][0]:.4f}, {overall[n]['ci'][1]:.4f}]",
             f"{overall[n]['p50_ms']:.1f}", f"{overall[n]['p95_ms']:.1f}"] for n in ORDER]
    L += ["## Quality and online latency", "",
          tabulate(rows, headers=["ranker", "NDCG@10", "95% CI", "p50 ms/query", "p95 ms/query"], tablefmt="github",
                   disable_numparse=True), "",
          "Hybrid latency = BM25 + Dense (both run per query). Index building is excluded.", ""]
    rows = [[t["id"], f"{t['a']} vs {t['b']}", f"{t['diff']:+.4f}", f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]",
             fmt_p(t["p_value"]), "yes" if t["p_value"] < 0.05 else "no", t["hypothesis"]] for t in tests]
    L += ["## Pre-registered comparisons (paired bootstrap over queries, 10,000 resamples)", "",
          tabulate(rows, headers=["id", "comparison", "diff", "95% CI", "p-value", "significant (0.05)", "hypothesis"],
                   tablefmt="github", disable_numparse=True), ""]
    rows = [[s["segment"], s["queries"]] + [f"{s[n]:.4f}" for n in ORDER]
            + [f"{s['LLM_vs_CE']['diff']:+.4f} (p={fmt_p(s['LLM_vs_CE']['p_value'])})"] for s in segs]
    L += ["## By query length", "",
          tabulate(rows, headers=["segment", "queries"] + ORDER + ["LLM - Cross-encoder"], tablefmt="github",
                   disable_numparse=True), ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
