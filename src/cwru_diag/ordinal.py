"""类型×损伤程度的序数（ordinal）标签与分层指标。

10 分类里"错一档尺寸"与"错一个部位"严重性不同，普通准确率分不开。
这里提供：
  - CLASSES10 / TYPE_OF / to_labels10   —— 10 类标签编码
  - size_level / hierarchical_scores    —— 类型正确率 + 条件尺寸指标 + 序数误差
"""
from __future__ import annotations

import numpy as np

TYPES = ["Normal", "IR", "OR", "B"]
SEVS = [0.007, 0.014, 0.021]
CLASSES10 = ["Normal"] + [f"{t}{int(s * 1000):03d}" for t in ("IR", "OR", "B") for s in SEVS]
TYPE_OF = {c: ("Normal" if c == "Normal" else c[:2]) for c in CLASSES10}
# 损伤程度序数：Normal=0，0.007″=1，0.014″=2，0.021″=3
SIZE_LEVEL = {"Normal": 0, "007": 1, "014": 2, "021": 3}


def to_labels10(ws) -> np.ndarray:
    """WindowSet（4 类 label + severity 列）→ 10 类整数标签。"""
    y = np.empty(len(ws), dtype=int)
    for i in range(len(ws)):
        t = TYPES[int(ws.label[i])]
        sev = float(ws.severity[i])
        name = "Normal" if t == "Normal" else f"{t}{int(round(sev * 1000)):03d}"
        y[i] = CLASSES10.index(name)
    return y


def size_level(cls: str) -> int:
    """类名 → 损伤程度序数（0–3）。"""
    return SIZE_LEVEL["Normal" if cls == "Normal" else cls[-3:]]


def hierarchical_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """分层指标：
      type_accuracy              —— 部位（Normal/IR/OR/B）正确率
      size_acc_given_type        —— 部位正确条件下尺寸也正确
      size_within1_given_type    —— 部位正确条件下尺寸误差 ≤ 1 档
      mean_abs_level_err_given_type —— 部位正确条件下 |Δ尺寸档| 的均值
      size_level_accuracy        —— （全样本）尺寸档位正确率，部位错了也照算
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        raise ValueError("空样本")
    t_true = np.array([TYPE_OF[CLASSES10[i]] for i in y_true])
    t_pred = np.array([TYPE_OF[CLASSES10[i]] for i in y_pred])
    type_ok = t_true == t_pred
    lv_true = np.array([size_level(CLASSES10[i]) for i in y_true])
    lv_pred = np.array([size_level(CLASSES10[i]) for i in y_pred])
    dl = np.abs(lv_true - lv_pred)
    n_type = int(type_ok.sum())
    out = {
        "n": n,
        "type_accuracy": float(type_ok.mean()),
        "size_level_accuracy": float((dl == 0).mean()),
    }
    if n_type > 0:
        out["size_acc_given_type"] = float((dl[type_ok] == 0).mean())
        out["size_within1_given_type"] = float((dl[type_ok] <= 1).mean())
        out["mean_abs_level_err_given_type"] = float(dl[type_ok].mean())
    else:
        out["size_acc_given_type"] = float("nan")
        out["size_within1_given_type"] = float("nan")
        out["mean_abs_level_err_given_type"] = float("nan")
    return out
