"""分段与划分：窗口计数、重叠步长、分组划分无泄漏。"""
import numpy as np

from cwru_diag.segmentation import WindowSet, segment_array, split_by_time_block


def test_step_and_count():
    x = np.arange(1000, dtype=float)
    X, starts = segment_array(x, window=100, overlap=0.5)
    assert X.shape == (19, 100)          # step=50, 0..900
    assert starts[1] - starts[0] == 50
    assert np.allclose(X[0], x[:100])
    assert np.allclose(X[-1], x[900:1000])


def test_short_signal_returns_empty():
    X, starts = segment_array(np.zeros(50), window=100, overlap=0.5)
    assert X.shape == (0, 100) and len(starts) == 0


def test_no_overlap_full_coverage():
    X, starts = segment_array(np.arange(400, dtype=float), window=100, overlap=0.0)
    assert X.shape == (4, 100)
    assert starts.tolist() == [0, 100, 200, 300]


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        segment_array(np.zeros(500), window=100, overlap=1.0)


def test_grouped_split_has_no_block_leakage():
    n = 400
    ws = WindowSet(
        X=np.zeros((n, 8)),
        file_id=np.repeat([1, 2], n // 2),
        block=np.concatenate([np.arange(n // 2) // 4, np.arange(n // 2) // 4]),
        start=np.arange(n),
    )
    tr, te = split_by_time_block(ws, test_frac=0.3, guard_blocks=1)
    assert tr.sum() + te.sum() < n            # 隔离带丢掉了窗口
    assert not set(ws.block[tr]) & set(ws.block[te])
    # 每个文件在训练与测试两侧都有样本
    assert set(ws.file_id[tr]) == {1, 2}
    assert set(ws.file_id[te]) == {1, 2}
