"""Export SASRec to ONNX, quantize it, and serve it with ONNX Runtime on CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models.base import Recommender
from .models.sasrec import SASRec


class _LastScores(torch.nn.Module):
    """ONNX graph = sequence (batch, max_len) int64 -> scores over all items (batch, n_items)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, seq):
        return self.net.last_scores(seq)


def export_onnx(model: SASRec, path: str | Path) -> Path:
    """The wrapper must be in eval mode: torch.onnx.export restores the exported module's original mode
    afterwards, and a fresh wrapper starts in train mode, which silently switched the wrapped SASRec to
    train mode (dropout on) after export. Found by tests/test_serving.py."""
    net = model.net.to("cpu").eval()
    example = torch.from_numpy(model.seqs_[:2])
    torch.onnx.export(
        _LastScores(net).eval(), (example,), str(path), input_names=["seq"], output_names=["scores"],
        dynamic_axes={"seq": {0: "batch"}, "scores": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    net.eval()
    return Path(path)


def quantize_int8(src: str | Path, dst: str | Path) -> Path:
    """Dynamic int8 quantization of the weights (MatMul / Gemm); activations stay float at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return Path(dst)


class OnnxSASRec(Recommender):
    """Same scoring as SASRec.score, but through an ONNX Runtime session on CPU."""

    def __init__(self, path: str | Path, seqs: np.ndarray, threads: int = 4, name: str = "SASRec (ONNX)"):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.seqs = seqs
        self.name = name

    def fit(self, train):  # already trained
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.session.run(["scores"], {"seq": self.seqs[np.asarray(users)]})[0]
