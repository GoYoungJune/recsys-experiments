"""Minimal recommendation API: SASRec (ONNX int8) on CPU.

    python scripts/serve.py                         # after scripts/benchmark_serving.py
    curl http://127.0.0.1:8000/recommend/42?k=10    # user index -> top-k movieIds and titles
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, HTTPException  # noqa: E402

from recsys.data import load_split  # noqa: E402
from recsys.models import SASRec  # noqa: E402
from recsys.serving import OnnxSASRec  # noqa: E402
from run_experiments import load_titles  # noqa: E402

RAW = ROOT / "data" / "ml-1m"
MODEL = ROOT / "results" / "ml-1m" / "serving" / "sasrec_int8.onnx"


def build_app(model_path: Path = MODEL, raw: Path = RAW) -> FastAPI:
    split, _ = load_split(raw)
    tv, tv_times = split.train_plus_valid(), split.train_plus_valid_times()
    seqs = SASRec(tv_times)._sequences(tv)          # the same input the benchmark used
    model = OnnxSASRec(model_path, seqs, threads=4)
    seen = tv.tocsr()
    titles = load_titles(raw)
    app = FastAPI(title="SASRec recommender")

    @app.get("/recommend/{user}")
    def recommend(user: int, k: int = 10):
        if not 0 <= user < split.n_users or not 1 <= k <= 100:
            raise HTTPException(status_code=404, detail="unknown user index or k outside 1..100")
        scores = model.score(np.array([user]))[0].astype(np.float64)
        scores[seen.indices[seen.indptr[user]:seen.indptr[user + 1]]] = -np.inf
        top = np.argsort(-scores)[:k]
        return {"user": user, "items": [{"movieId": int(split.item_ids[i]), "title": titles[split.item_ids[i]],
                                         "score": float(scores[i])} for i in top]}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
