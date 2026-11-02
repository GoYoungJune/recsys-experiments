from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .base import SEED, Recommender


class BPRMF(Recommender):
    """Matrix factorization trained with Bayesian Personalized Ranking (Rendle et al., 2009).

    x_ui = <p_u, q_i> + b_i
    loss = -mean(log sigmoid(x_ui - x_uj)) + weight_decay * mean(||p_u||^2 + ||q_i||^2 + ||q_j||^2 + b_i^2 + b_j^2)
    for observed positive i and sampled item j that the user has NOT interacted with.

    If `eval_fn` is given to fit(), the model is validated every `eval_every` epochs; training stops
    after `patience` evaluations without improvement and the best weights are restored.
    `best_epoch_` records the epoch of the best validation score (used to refit on train+valid).
    """

    name = "BPR-MF"

    def __init__(
        self,
        dim: int = 64,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 200,
        eval_every: int = 5,
        patience: int = 4,
        seed: int = SEED,
        device: str = "cpu",
    ):
        self.dim = dim
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_every = eval_every
        self.patience = patience
        self.seed = seed
        self.device = device

    # -- training -------------------------------------------------------------------------------
    def _sample_negatives(self, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Uniform negatives, resampled until none of them is a positive of that user."""
        neg = rng.integers(0, self.n_items_, size=len(users))
        bad = np.isin(users * self.n_items_ + neg, self.pos_keys_, assume_unique=False)
        while bad.any():
            neg[bad] = rng.integers(0, self.n_items_, size=bad.sum())
            bad[bad] = np.isin(users[bad] * self.n_items_ + neg[bad], self.pos_keys_)
        return neg

    def fit(
        self,
        train: sp.csr_matrix,
        eval_fn: Callable[["BPRMF"], float] | None = None,
    ) -> "BPRMF":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        train = train.tocoo()
        self.n_users_, self.n_items_ = train.shape
        users, items = train.row.astype(np.int64), train.col.astype(np.int64)
        self.pos_keys_ = np.sort(users * self.n_items_ + items)

        dev = torch.device(self.device)
        self.user_emb = torch.nn.Embedding(self.n_users_, self.dim).to(dev)
        self.item_emb = torch.nn.Embedding(self.n_items_, self.dim).to(dev)
        self.item_bias = torch.nn.Embedding(self.n_items_, 1).to(dev)
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
        torch.nn.init.zeros_(self.item_bias.weight)
        params = [self.user_emb.weight, self.item_emb.weight, self.item_bias.weight]
        opt = torch.optim.Adam(params, lr=self.lr)

        best_score, best_state, bad_rounds = -np.inf, None, 0
        self.best_epoch_ = self.epochs
        self.history_: list[tuple[int, float]] = []
        n = len(users)
        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n)
            neg_all = self._sample_negatives(users[perm], rng)
            for s in range(0, n, self.batch_size):
                b = perm[s : s + self.batch_size]
                u = torch.from_numpy(users[b]).to(dev)
                i = torch.from_numpy(items[b]).to(dev)
                j = torch.from_numpy(neg_all[s : s + self.batch_size]).to(dev)
                pu, qi, qj = self.user_emb(u), self.item_emb(i), self.item_emb(j)
                bi, bj = self.item_bias(i).squeeze(1), self.item_bias(j).squeeze(1)
                x_uij = (pu * qi).sum(1) + bi - (pu * qj).sum(1) - bj
                reg = (pu.pow(2).sum(1) + qi.pow(2).sum(1) + qj.pow(2).sum(1) + bi.pow(2) + bj.pow(2)).mean()
                loss = -F.logsigmoid(x_uij).mean() + self.weight_decay * reg
                opt.zero_grad()
                loss.backward()
                opt.step()

            if eval_fn is not None and epoch % self.eval_every == 0:
                score = eval_fn(self)
                self.history_.append((epoch, score))
                if score > best_score:
                    best_score, bad_rounds, self.best_epoch_ = score, 0, epoch
                    best_state = [p.detach().clone() for p in params]
                else:
                    bad_rounds += 1
                    if bad_rounds >= self.patience:
                        break

        if best_state is not None:
            with torch.no_grad():
                for p, best in zip(params, best_state):
                    p.copy_(best)
        self.best_score_ = best_score
        return self

    # -- inference ------------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.user_emb.weight.device)
        s = self.user_emb(u) @ self.item_emb.weight.T + self.item_bias.weight.T
        return s.cpu().numpy()
