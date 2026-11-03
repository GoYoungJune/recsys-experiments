"""CTR models over categorical fields: LR, FM, DeepFM, DCN-v2. All return logits of shape (batch,)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Encoded


class FieldEmbedding(nn.Module):
    """One table per single-valued field (index 0 = unknown) plus a genre table averaged over non-pad genres.

    Returns embeddings (B, n_fields + 1, d) and first-order weights (B, n_fields + 1).
    """

    def __init__(self, field_sizes: list[int], n_genres: int, dim: int):
        super().__init__()
        self.offsets = torch.tensor(np.concatenate([[0], np.cumsum(field_sizes)[:-1]]), dtype=torch.long)
        total = int(sum(field_sizes))
        self.emb = nn.Embedding(total, dim)
        self.w = nn.Embedding(total, 1)
        self.g_emb = nn.Embedding(n_genres, dim, padding_idx=0)
        self.g_w = nn.Embedding(n_genres, 1, padding_idx=0)
        for e in (self.emb, self.g_emb):
            nn.init.normal_(e.weight, std=0.01)
        for e in (self.w, self.g_w):
            nn.init.zeros_(e.weight)
        with torch.no_grad():
            self.g_emb.weight[0].zero_()

    def forward(self, single: torch.Tensor, genres: torch.Tensor):
        idx = single + self.offsets.to(single.device)
        mask = (genres > 0).float().unsqueeze(-1)                       # (B, G, 1)
        cnt = mask.sum(1).clamp(min=1)
        g_e = (self.g_emb(genres) * mask).sum(1) / cnt                  # (B, d)
        g_w = (self.g_w(genres) * mask).sum(1) / cnt                    # (B, 1)
        e = torch.cat([self.emb(idx), g_e.unsqueeze(1)], 1)
        w = torch.cat([self.w(idx).squeeze(-1), g_w], 1)
        return e, w


def mlp(d_in: int, hidden=(256, 128), dropout=0.2) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    return nn.Sequential(*layers)


class CTRModel(nn.Module):
    """kind in {"lr", "fm", "deepfm", "dcnv2"}."""

    def __init__(self, kind: str, field_sizes: list[int], n_genres: int, dim: int = 16, n_cross: int = 3,
                 seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)  # weight init happens here, before fit() seeds the batch order
        self.kind = kind
        self.embed = FieldEmbedding(field_sizes, n_genres, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        d_flat = (len(field_sizes) + 1) * dim
        if kind == "deepfm":
            self.deep = mlp(d_flat)
            self.deep_out = nn.Linear(128, 1)
        if kind == "dcnv2":
            self.cross = nn.ModuleList([nn.Linear(d_flat, d_flat) for _ in range(n_cross)])
            self.deep = mlp(d_flat)
            self.out = nn.Linear(d_flat + 128, 1)

    def forward(self, single, genres):
        e, w = self.embed(single, genres)
        linear = w.sum(1) + self.bias
        if self.kind == "lr":
            return linear
        if self.kind in ("fm", "deepfm"):
            s = e.sum(1)
            fm = 0.5 * (s.pow(2) - e.pow(2).sum(1)).sum(1)
            out = linear + fm
            if self.kind == "deepfm":
                out = out + self.deep_out(self.deep(e.flatten(1))).squeeze(-1)
            return out
        x0 = e.flatten(1)                                                # DCN-v2: x_{l+1} = x0 * (W x_l + b) + x_l
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.out(torch.cat([x, self.deep(x0)], 1)).squeeze(-1)


def _tensors(d: Encoded):
    return torch.from_numpy(d.single), torch.from_numpy(d.genres), torch.from_numpy(d.label)


@torch.no_grad()
def predict(model: CTRModel, d: Encoded, batch: int = 65536) -> np.ndarray:
    """Predicted probabilities."""
    model.eval()
    s, g, _ = _tensors(d)
    out = [torch.sigmoid(model(s[i:i + batch], g[i:i + batch])) for i in range(0, len(s), batch)]
    return torch.cat(out).numpy().astype(np.float64)


def fit(model: CTRModel, train: Encoded, valid: Encoded, lr=1e-3, batch=4096, epochs=30, patience=2,
        seed=42, log=print) -> dict:
    """Adam + BCE; early stopping on valid logloss; best weights restored. Returns training history."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    s, g, y = _tensors(train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(model(s[b], g[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        p = np.clip(predict(model, valid), 1e-7, 1 - 1e-7)
        vll = float(-np.mean(valid.label * np.log(p) + (1 - valid.label) * np.log(1 - p)))
        hist.append((ep, vll))
        log(f"      epoch {ep:2d} valid logloss {vll:.4f}")
        if vll < best - 1e-5:
            best, best_state, bad = vll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"history": hist, "best_epoch": min(hist, key=lambda h: h[1])[0], "best_valid_logloss": best}
