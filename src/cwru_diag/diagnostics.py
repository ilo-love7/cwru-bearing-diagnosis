"""诊断工具：异常件特征 z 分数、预测分布分解。纯函数，便于单测。"""
from __future__ import annotations

import numpy as np


def feature_zscores(X: np.ndarray, names: list[str],
                    ref_mask: np.ndarray, target_mask: np.ndarray) -> list[dict]:
    """target 子集的中位数相对 ref 子集的稳健 z 分数。

    z = (median_target − median_ref) / std_ref（std 为 0 时 z 记 0）。
    返回按 |z| 降序的 [{name, z, median_target, median_ref, std_ref}, ...]。
    """
    X = np.asarray(X, dtype=np.float64)
    ref_mask = np.asarray(ref_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X 列数与 names 不一致")
    if not ref_mask.any() or not target_mask.any():
        raise ValueError("ref/target 掩码不能为空")
    if len(ref_mask) != len(X) or len(target_mask) != len(X):
        raise ValueError("掩码长度与 X 行数不一致")
    ref, tgt = X[ref_mask], X[target_mask]
    med_ref = np.median(ref, axis=0)
    med_tgt = np.median(tgt, axis=0)
    std_ref = ref.std(axis=0)
    z = np.where(std_ref > 0, (med_tgt - med_ref) / np.where(std_ref > 0, std_ref, 1.0), 0.0)
    rows = sorted(
        ({"name": names[j], "z": float(z[j]),
          "median_target": float(med_tgt[j]), "median_ref": float(med_ref[j]),
          "std_ref": float(std_ref[j])} for j in range(len(names))),
        key=lambda r: -abs(r["z"]))
    return rows


def prediction_breakdown(y_pred: np.ndarray, labels: list[str],
                         pred_labels: list[str] | None = None) -> dict:
    """预测分布：每个类别的窗口数与占比。"""
    y_pred = np.asarray(y_pred, dtype=int)
    pred_labels = pred_labels or labels
    if len(pred_labels) != len(labels):
        raise ValueError("pred_labels 与 labels 长度不一致")
    n = len(y_pred)
    if n == 0:
        raise ValueError("空样本")
    counts = {labels[i]: int((y_pred == i).sum()) for i in range(len(labels))}
    return {"n": n, "counts": counts,
            "fraction": {k: v / n for k, v in counts.items()}}
