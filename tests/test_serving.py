import numpy as np
import pytest

from recsys.data import iterative_filter, leave_one_out


@pytest.fixture
def split(toy_df):
    return leave_one_out(iterative_filter(toy_df, min_count=5))


def test_onnx_export_matches_pytorch_and_int8_runs(split, tmp_path):
    from recsys.models import SASRec
    from recsys.serving import OnnxSASRec, export_onnx, quantize_int8

    m = SASRec(split.train_times, dim=16, max_len=20, epochs=2, device="cpu").fit(split.train)
    users = np.arange(split.n_users)
    before = m.score(users)
    fp32 = export_onnx(m, tmp_path / "m.onnx")
    assert np.allclose(m.score(users), before, atol=1e-6)      # exporting must not change the model
    onnx_scores = OnnxSASRec(fp32, m.seqs_, threads=1).score(users)
    assert np.allclose(onnx_scores, before, atol=1e-4)
    int8 = quantize_int8(fp32, tmp_path / "m_int8.onnx")
    q = OnnxSASRec(int8, m.seqs_, threads=1).score(users)
    assert q.shape == onnx_scores.shape and np.isfinite(q).all()
