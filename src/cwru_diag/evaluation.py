"""评测指标与结果落盘。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             labels: list[str] | None = None) -> dict:
    """返回可 JSON 序列化的指标字典（含混淆矩阵与逐类 P/R/F1）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels_idx = list(range(len(labels))) if labels else present
    names = labels if labels else [str(i) for i in labels_idx]

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_idx, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": names,
        "per_class": {
            names[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(support[i]),
            } for i in labels_idx
        },
    }


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"不可序列化: {type(o)}")

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    return path


def summarize(metrics: dict) -> str:
    """一行摘要，便于日志里横向对比。"""
    return (f"acc={metrics['accuracy']:.4f}  bacc={metrics['balanced_accuracy']:.4f}  "
            f"macroF1={metrics['macro_f1']:.4f}  n={metrics['n']}")


def aggregate_runs(runs: list[dict], n_labels: int) -> dict:
    """多随机种子聚合：均值/标准差 + 混淆矩阵求和。"""
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    mean = {k: float(np.mean([r["metrics"][k] for r in runs])) for k in keys}
    std = {k: float(np.std([r["metrics"][k] for r in runs], ddof=0)) for k in keys}
    cm = np.sum([np.asarray(r["metrics"]["confusion_matrix"]) for r in runs], axis=0)
    per_class: dict[str, dict[str, float]] = {}
    for name in runs[0]["metrics"]["per_class"]:
        per_class[name] = {
            "precision": float(np.mean([r["metrics"]["per_class"][name]["precision"] for r in runs])),
            "recall": float(np.mean([r["metrics"]["per_class"][name]["recall"] for r in runs])),
            "f1": float(np.mean([r["metrics"]["per_class"][name]["f1"] for r in runs])),
        }
    return {
        "mean": mean,
        "std": std,
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "seeds": runs,
    }
