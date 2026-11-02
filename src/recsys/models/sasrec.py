from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from .base import SEED, Recommender


class SASRecNet(nn.Module):
    """Causal Transformer over a user's item sequence (Kang & McAuley, 2018).

    Item index 0 is padding; real items are shifted by +1. The hidden state at position t is used to
    score the item at position t+1 by a dot product with the (shared) item embeddings.
    """

    def __init__(self, n_items: int, dim: int, n_blocks: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim_feedforward=dim, dropout=dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        # small init as in the reference implementation; nn.Embedding's N(0, 1) times sqrt(dim) saturates the logits
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        causal =torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """(batch, max_len) item ids, left-padded with 0 -> (batch, max_len, dim) hidden states."""
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) * self.item_emb.embedding_dim ** 0.5 + self.pos_emb(positions)
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        x = self.blocks(x, mask=self.causal[: seq.shape[1], : seq.shape[1]])
        return self.norm(x)

    def last_scores(self, seq: torch.Tensor) -> torch.Tensor:
        """Scores over all real items from the last position: (batch, n_items)."""
        h = self.forward(seq)[:, -1]
        return h @ self.item_emb.weight[1:].T


class SASRec(Recommender):
    """Self-attentive sequential recommendation on the time-ordered positive history.

    Training: for every position, predict the next item against one uniformly sampled negative
    (binary cross-entropy), as in the paper. Scoring: the hidden state after the user's last
    `max_len` items. `times` gives the order and must cover exactly the interactions passed to fit()
    (ties broken by item index, i.e. by movieId, the same as the split).

    Early stopping follows BPRMF: validate every `eval_every` epochs, stop after `patience`
    evaluations without improvement, restore the best weights, record `best_epoch_`.
    """

    name = "SASRec"

    def __init__(
        self,
        times: sp.csr_matrix,
        dim: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        max_len: int = 200,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 200,
        eval_every: int = 5,
        patience: int = 4,
        seed: int = SEED,
        device: str | None = None,
    ):
        self.times = times
        self.dim = dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_every = eval_every
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # -- data -----------------------------------------------------------------------------------
    def _sequences(self, train: sp.csr_matrix) -> np.ndarray:
        """(n_users, max_len) left-padded item ids (+1), oldest -> newest, last max_len kept."""
        times = self.times.tocsr()
        if times.shape != train.shape or times.nnz != train.nnz:
            raise ValueError("times must cover exactly the interactions in train.")
        seqs = np.zeros((train.shape[0], self.max_len), dtype=np.int64)
        for u in range(train.shape[0]):
            lo, hi = times.indptr[u], times.indptr[u + 1]
            items, ts = times.indices[lo:hi], times.data[lo:hi]
            order = np.lexsort((items, ts))           # by time, then by item index
            s = items[order][-self.max_len :] + 1
            seqs[u, self.max_len - len(s) :] = s
        return seqs

    def _negatives(self, targets: np.ndarray, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One uniform negative per position that is not a positive of that user (0 where target is pad)."""
        neg = rng.integers(1, self.n_items_ + 1, size=targets.shape)
        for _ in range(10):  # resample collisions; the catalog is large so this converges fast
            bad = np.isin(users[:, None] * (self.n_items_ + 1) + neg, self.pos_keys_)
            if not bad.any():
                break
            neg[bad] = rng.integers(1, self.n_items_ + 1, size=bad.sum())
        neg[targets == 0] = 0
        return neg

    # -- training -------------------------------------------------------------------------------
    def fit(self, train: sp.csr_matrix, eval_fn: Callable[["SASRec"], float] | None = None) -> "SASRec":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_users_, self.n_items_ = train.shape
        coo = train.tocoo()
        self.pos_keys_ = np.sort(coo.row.astype(np.int64) * (self.n_items_ + 1) + coo.col + 1)
        self.seqs_ = self._sequences(train)
        # training input = sequence shifted by one: positions 0..L-2 predict 1..L-1
        full = self._sequences_with_next(train)

        dev = torch.device(self.device)
        self.net = SASRecNet(self.n_items_, self.dim, self.n_blocks, self.n_heads, self.max_len, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.98))
        trainable = np.flatnonzero((full[:, -2] > 0))   # users with at least 2 items

        best_score, best_state, bad_rounds = -np.inf, None, 0
        self.best_epoch_ = self.epochs
        self.history_: list[tuple[int, float]] = []
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = rng.permutation(trainable)
            for s in range(0, len(perm), self.batch_size):
                users = perm[s : s + self.batch_size]
                inp, pos = full[users, :-1], full[users, 1:]
                neg = self._negatives(pos, users, rng)
                inp_t = torch.from_numpy(inp).to(dev)
                pos_t, neg_t = torch.from_numpy(pos).to(dev), torch.from_numpy(neg).to(dev)
                h = self.net(inp_t)
                pos_logit = (h * self.net.item_emb(pos_t)).sum(-1)
                neg_logit = (h * self.net.item_emb(neg_t)).sum(-1)
                mask = pos_t > 0
                loss = (
                    F.binary_cross_entropy_with_logits(pos_logit[mask], torch.ones_like(pos_logit[mask]))
                    + F.binary_cross_entropy_with_logits(neg_logit[mask], torch.zeros_like(neg_logit[mask]))
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

            if eval_fn is not None and epoch % self.eval_every == 0:
                self.net.eval()
                score = eval_fn(self)
                self.history_.append((epoch, score))
                if score > best_score:
                    best_score, bad_rounds, self.best_epoch_ = score, 0, epoch
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad_rounds += 1
                    if bad_rounds >= self.patience:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_score_ = best_score
        return self

    def _sequences_with_next(self, train: sp.csr_matrix) -> np.ndarray:
        """Like _sequences but max_len + 1 long, so input and next-item targets both have max_len."""
        keep = self.max_len
        self.max_len = keep + 1
        try:
            return self._sequences(train)
        finally:
            self.max_len = keep

    # -- inference ------------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        seq = torch.from_numpy(self.seqs_[np.asarray(users)]).to(next(self.net.parameters()).device)
        return self.net.last_scores(seq).float().cpu().numpy()
