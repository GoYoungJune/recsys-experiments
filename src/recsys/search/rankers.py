"""Rankers for one query's candidate list. Each score(query, docs) returns one float per doc, higher = better.

Offline vs online: BM25 statistics and dense document embeddings can be built before a query arrives
(an index). `score` is the online part only, which is what the latency benchmark times.
"""
from __future__ import annotations

import re

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 with IDF and average length taken from a whole product corpus (not just the candidates)."""

    name = "BM25"

    def __init__(self, corpus: list[str], k1: float = 1.2, b: float = 0.75):
        vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, binary=True)
        df = np.asarray(vec.fit_transform(corpus).sum(axis=0)).ravel()
        n = len(corpus)
        self.idf = dict(zip(vec.get_feature_names_out(), np.log(1 + (n - df + 0.5) / (df + 0.5))))
        self.avgdl = float(np.mean([len(tokenize(t)) for t in corpus]))
        self.k1, self.b = k1, b

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(docs))
        for i, d in enumerate(docs):
            toks = tokenize(d)
            dl = len(toks)
            for t in q:
                tf = toks.count(t)
                if tf:
                    out[i] += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


class Dense:
    """Bi-encoder (bge-m3): cosine of query and document embeddings. Documents are encoded once, offline."""

    name = "Dense (bge-m3)"

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model.half()
        self.doc_emb: dict[str, np.ndarray] = {}

    def index(self, doc_ids: list[str], texts: list[str], batch_size: int = 128) -> None:
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        self.doc_emb.update(zip(doc_ids, emb))

    def score(self, query: str, doc_ids: list[str]) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        return np.stack([self.doc_emb[d] for d in doc_ids]) @ q


class CrossEncoder:
    """Cross-encoder reranker (bge-reranker-v2-m3): reads query and document together."""

    name = "Cross-encoder (bge-reranker-v2-m3)"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda", max_length: int = 256):
        from sentence_transformers import CrossEncoder as STCrossEncoder

        self.model = STCrossEncoder(model_name, device=device, max_length=max_length)
        self.model.model.half()

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, d) for d in docs], batch_size=64, show_progress_bar=False))


class PointwiseLLM:
    """Zero-shot LLM judge: score = logit("Yes") - logit("No") for "does this product match the query?".

    No fine-tuning and no examples in the prompt. Pointwise rather than listwise because a 3B model
    follows a yes/no format far more reliably than it reorders a list of 20 items.
    """

    name = "LLM pointwise (Qwen2.5-3B)"
    PROMPT = (
        "Shopping query: {query}\n"
        "Product: {doc}\n"
        "Is this product exactly what the shopper is looking for? Answer Yes or No."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "cuda", max_doc_chars: int = 300):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
        # the document is cut instead of the tokenized prompt, so the answer cue at the end is never truncated
        self.device, self.max_doc_chars = device, max_doc_chars
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no = self.tok.encode("No", add_special_tokens=False)[0]

    def _prompts(self, query: str, docs: list[str]) -> list[str]:
        return [
            self.tok.apply_chat_template(
                [{"role": "user", "content": self.PROMPT.format(query=query, doc=d[: self.max_doc_chars])}],
                tokenize=False, add_generation_prompt=True,
            )
            for d in docs
        ]

    @torch.no_grad()
    def score(self, query: str, docs: list[str]) -> np.ndarray:
        enc = self.tok(self._prompts(query, docs), return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**enc).logits[:, -1, :].float()
        return (logits[:, self.yes] - logits[:, self.no]).cpu().numpy()
