"""H8: serve SASRec on CPU. Quality and single-request latency for PyTorch, ONNX fp32 and ONNX int8.

Refits SASRec on ml-1m train+valid with the configuration chosen in results/ml-1m/results.json, exports it,
quantizes it, then evaluates every variant on the same test users. EASE is timed as a reference.
Outputs go to results/ml-1m/serving/: results.md, results.json, sasrec_fp32.onnx, sasrec_int8.onnx

    python scripts/benchmark_serving.py      # run after: python scripts/run_experiments.py --dataset ml-1m
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data import load_split  # noqa: E402
from recsys.evaluate import evaluate  # noqa: E402
from recsys.models import EASE, SASRec  # noqa: E402
from recsys.serving import OnnxSASRec, export_onnx, quantize_int8  # noqa: E402
from recsys.stats import paired_bootstrap  # noqa: E402

SEED = 42
THREADS = 4             # same CPU budget for every variant
N_REQUESTS = 1000       # single-user requests timed (after 50 warm-up requests)
RAW = ROOT / "data" / "ml-1m"
RES = ROOT / "results" / "ml-1m"
OUT = RES / "serving"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def time_requests(fn, users):
    """Milliseconds per single-user request (top-10 included), after warm-up."""
    for u in users[:50]:
        fn(u)
    t = []
    for u in users[50:]:
        s = time.perf_counter()
        fn(u)
        t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    t0 = time.time()
    prior = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    cfg = prior["best_params"]["SASRec"]
    split, _ = load_split(RAW)
    tv, tv_times = split.train_plus_valid(), split.train_plus_valid_times()

    log(f"Refit SASRec {cfg['params']} for {cfg['best_epoch']} epochs on train+valid")
    sas = SASRec(tv_times, **cfg["params"], epochs=cfg["best_epoch"]).fit(tv)
    sas.net.to("cpu")
    sas.device = "cpu"

    log("Export and quantize")
    fp32 = export_onnx(sas, OUT / "sasrec_fp32.onnx")
    int8 = quantize_int8(fp32, OUT / "sasrec_int8.onnx")
    variants = {
        "PyTorch fp32": sas,
        "ONNX Runtime fp32": OnnxSASRec(fp32, sas.seqs_, THREADS, "ONNX fp32"),
        "ONNX Runtime int8": OnnxSASRec(int8, sas.seqs_, THREADS, "ONNX int8"),
    }
    torch_bytes = sum(p.numel() * p.element_size() for p in sas.net.parameters())
    sizes = {"PyTorch fp32": torch_bytes, "ONNX Runtime fp32": fp32.stat().st_size,
             "ONNX Runtime int8": int8.stat().st_size}

    log("Quality on test (same users, full ranking)")
    quality = {}
    for name, m in variants.items():
        r = evaluate(m, tv, split.test_items, ks=(10,), batch_size=256)
        quality[name] = r
        log(f"   {name:18s} NDCG@10={r['NDCG@10']:.4f}")
    ease_cfg = prior["best_params"]["EASE"]["params"]
    ease = EASE(**ease_cfg).fit(tv)
    ease.B_ = ease.B_.astype(np.float32)
    quality["EASE (reference)"] = evaluate(ease, tv, split.test_items, ks=(10,))
    tests = {
        "int8 vs fp32 (ONNX)": paired_bootstrap(quality["ONNX Runtime int8"]["per_user"]["NDCG@10"],
                                                quality["ONNX Runtime fp32"]["per_user"]["NDCG@10"]),
        "ONNX fp32 vs PyTorch": paired_bootstrap(quality["ONNX Runtime fp32"]["per_user"]["NDCG@10"],
                                                 quality["PyTorch fp32"]["per_user"]["NDCG@10"]),
    }
    max_abs = float(np.abs(variants["ONNX Runtime fp32"].score(np.arange(64)) - sas.score(np.arange(64))).max())

    log(f"Latency: {N_REQUESTS} single-user requests, {THREADS} CPU threads")
    rng = np.random.default_rng(SEED)
    users = rng.integers(0, split.n_users, size=N_REQUESTS + 50)
    seen = tv.tocsr()

    def topk(scores, u, k=10):
        scores = scores.astype(np.float64)
        scores[seen.indices[seen.indptr[u]:seen.indptr[u + 1]]] = -np.inf
        return np.argpartition(-scores, k)[:k]

    lat = {}
    for name, m in variants.items():
        lat[name] = time_requests(lambda u, m=m: topk(m.score(np.array([u]))[0], u), users)
        log(f"   {name:18s} p50={lat[name]['p50_ms']:.2f}ms p99={lat[name]['p99_ms']:.2f}ms")
    xv = tv.astype(np.float32).tocsr()
    lat["EASE (reference)"] = time_requests(lambda u: topk(np.asarray(xv[u] @ ease.B_).ravel(), u), users)
    sizes["EASE (reference)"] = ease.B_.nbytes

    rows = []
    for name in list(variants) + ["EASE (reference)"]:
        rows.append({"variant": name, "NDCG@10": quality[name]["NDCG@10"], "size_mb": sizes[name] / 1e6, **lat[name]})
    out = {"sasrec_config": cfg, "threads": THREADS, "n_requests": N_REQUESTS, "rows": rows, "tests": tests,
           "onnx_fp32_max_abs_score_diff_vs_pytorch": max_abs,
           "cpu_note": "single machine; numbers are for relative comparison", "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    L = ["# H8: serving SASRec on CPU (ml-1m test users)", "",
         f"Auto-generated by `scripts/benchmark_serving.py`. {N_REQUESTS} single-user requests, {THREADS} CPU threads, "
         "latency includes masking seen items and top-10 selection.", ""]
    L += [tabulate([[r["variant"], f"{r['NDCG@10']:.4f}", f"{r['size_mb']:.1f}", f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}",
                     f"{r['p99_ms']:.2f}"] for r in rows],
                   headers=["variant", "test NDCG@10", "model size (MB)", "p50 ms", "p95 ms", "p99 ms"],
                   tablefmt="github", disable_numparse=True), ""]
    L += [tabulate([[k, f"{v['diff']:+.4f}", f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]", fmt_p(v["p_value"])]
                    for k, v in tests.items()], headers=["comparison", "diff", "95% CI", "p-value"],
                   tablefmt="github", disable_numparse=True), "",
          f"ONNX fp32 vs PyTorch: max |score difference| on 64 users = {max_abs:.2e}.", ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
