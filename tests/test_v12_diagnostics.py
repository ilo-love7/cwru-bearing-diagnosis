"""v1.2 新增纯函数：序数/分层指标与异常件诊断。"""
import numpy as np
import pytest

from cwru_diag.diagnostics import feature_zscores, prediction_breakdown
from cwru_diag.ordinal import (CLASSES10, SIZE_LEVEL, hierarchical_scores,
                               size_level, to_labels10)


class _WS:
    """最小 WindowSet 替身，只带 to_labels10 需要的字段。"""
    def __init__(self, label, severity):
        self.label = np.asarray(label)
        self.severity = np.asarray(severity, dtype=float)

    def __len__(self):
        return len(self.label)


def test_classes10_layout():
    assert len(CLASSES10) == 10
    assert CLASSES10[0] == "Normal"
    assert "IR007" in CLASSES10 and "OR014" in CLASSES10 and "B021" in CLASSES10
    assert set(SIZE_LEVEL.values()) == {0, 1, 2, 3}


def test_size_level_mapping():
    assert size_level("Normal") == 0
    assert size_level("IR007") == 1
    assert size_level("OR014") == 2
    assert size_level("B021") == 3


def test_to_labels10():
    ws = _WS(label=[0, 1, 2, 3], severity=[0.0, 0.007, 0.014, 0.021])
    y = to_labels10(ws)
    assert [CLASSES10[i] for i in y] == ["Normal", "IR007", "OR014", "B021"]


def test_hierarchical_perfect():
    y = np.array([0, 1, 5, 9])
    s = hierarchical_scores(y, y)
    assert s["type_accuracy"] == 1.0
    assert s["size_acc_given_type"] == 1.0
    assert s["size_within1_given_type"] == 1.0
    assert s["mean_abs_level_err_given_type"] == 0.0
    assert s["size_level_accuracy"] == 1.0


def test_hierarchical_size_off_by_one():
    # 真值 IR007，预测 IR014：部位正确、尺寸差 1 档
    y_true = np.array([CLASSES10.index("IR007")])
    y_pred = np.array([CLASSES10.index("IR014")])
    s = hierarchical_scores(y_true, y_pred)
    assert s["type_accuracy"] == 1.0
    assert s["size_acc_given_type"] == 0.0
    assert s["size_within1_given_type"] == 1.0
    assert s["mean_abs_level_err_given_type"] == pytest.approx(1.0)
    assert s["size_level_accuracy"] == 0.0


def test_hierarchical_type_error_excluded_from_size_conditional():
    # IR007 → B007：部位错、尺寸档相同；条件尺寸指标只统计部位正确的样本
    y_true = np.array([CLASSES10.index("IR007")] * 3 + [CLASSES10.index("OR021")])
    y_pred = np.array([CLASSES10.index("B007")] * 3 + [CLASSES10.index("OR021")])
    s = hierarchical_scores(y_true, y_pred)
    assert s["type_accuracy"] == pytest.approx(0.25)
    assert s["size_acc_given_type"] == 1.0          # 只剩 OR021 那个样本
    assert s["size_level_accuracy"] == pytest.approx(1.0)  # 档位都对（3 个 B007 档位 1 = 真值 1）


def test_feature_zscores_ordering():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 3))
    tgt = ref.copy()
    tgt[:, 1] += 5.0          # 特征 1 显著偏移
    rows = feature_zscores(ref, ["a", "b", "c"],
                           np.ones(200, bool), np.ones(200, bool))
    # ref == target 时 z 全 0；改用不同掩码验证排序
    X = np.vstack([ref, tgt])
    m_ref = np.zeros(400, bool); m_ref[:200] = True
    m_tgt = ~m_ref
    rows = feature_zscores(X, ["a", "b", "c"], m_ref, m_tgt)
    assert rows[0]["name"] == "b"
    assert rows[0]["z"] == pytest.approx(5.0, rel=0.1)
    assert abs(rows[1]["z"]) < 0.5


def test_feature_zscores_zero_std_ref():
    X = np.zeros((10, 2))
    X[5:, 0] = 1.0
    m_ref = np.zeros(10, bool); m_ref[:5] = True
    rows = feature_zscores(X, ["flat", "jump"], m_ref, ~m_ref)
    d = {r["name"]: r for r in rows}
    assert d["flat"]["z"] == 0.0


def test_feature_zscores_validation():
    with pytest.raises(ValueError):
        feature_zscores(np.zeros((4, 2)), ["a"], np.ones(4, bool), np.zeros(4, bool))
    with pytest.raises(ValueError):
        feature_zscores(np.zeros((4, 2)), ["a", "b"], np.ones(3, bool), np.ones(3, bool))


def test_prediction_breakdown():
    y = np.array([0, 0, 1, 3])
    b = prediction_breakdown(y, ["Normal", "IR", "OR", "B"])
    assert b["n"] == 4
    assert b["counts"] == {"Normal": 2, "IR": 1, "OR": 0, "B": 1}
    assert b["fraction"]["Normal"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        prediction_breakdown(np.array([], dtype=int), ["a"])
