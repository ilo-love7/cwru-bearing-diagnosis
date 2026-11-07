"""09 · 1D-CNN 补齐分布外口径：留一载荷 / 未见程度 / 未见位置 / 10 分类。

03 只跑了同工况 4 分类（随机切窗 + 时间块），报告 §5.5 里 "CNN 未在位置/程度口径上跑"
的缺口在这里补齐，口径与 04/05/07 的特征模型完全一致，可直接横向对比：

  A 4 分类 · 留一载荷（4 折，seed=0）
  B 4 分类 · 未见损伤程度 0.028"（训练 3 档 → 测试第 4 档）
  C 4 分类 · 未见外圈位置 @3/@12 点（训练只看 @6 点）
  D 10 分类 · 时间块（seed 0/1）
  E 10 分类 · 留一载荷（4 折，seed=0），含类型正确率与剔除 217.mat 敏感性

产出
  results/metrics/cnn_ood.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import predict_cnn, train_cnn             # noqa: E402
from cwru_diag.pipeline import (carve_val, features_for, load_all,  # noqa: E402
                                time_block_split)

# ---- 10 分类标签（与 07_severity_10class.py 同口径，脚本自包含故复制）----
TYPES = ["Normal", "IR", "OR", "B"]
SEVS = [0.007, 0.014, 0.021]
CLASSES = ["Normal"] + [f"{t}{int(s * 1000):03d}" for t in ("IR", "OR", "B") for s in SEVS]
TYPE_OF = {c: ("Normal" if c == "Normal" else c[:2]) for c in CLASSES}


def to_labels10(ws) -> np.ndarray:
    y = np.empty(len(ws), dtype=int)
    for i in range(len(ws)):
        t = TYPES[ws.label[i]]
        sev = float(ws.severity[i])
        y[i] = CLASSES.index("Normal" if t == "Normal" else f"{t}{int(round(sev * 1000)):03d}")
    return y


def type_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t_true = np.array([TYPE_OF[CLASSES[i]] for i in y_true])
    t_pred = np.array([TYPE_OF[CLASSES[i]] for i in y_pred])
    return float((t_true == t_pred).mean())


def fit_predict(X, y, tr, va, te, n_classes, cfg, seed):
    """训练一次 CNN 并返回（测试预测、训练信息）。"""
    t0 = time.time()
    res = train_cnn(
        X[tr], y[tr], X[va], y[va], n_classes=n_classes,
        epochs=int(cfg["cnn"]["epochs"]), batch_size=int(cfg["cnn"]["batch_size"]),
        lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
        dropout=float(cfg["cnn"]["dropout"]), seed=seed, verbose=False,
    )
    pred = predict_cnn(res.model, X[te])
    info = {"best_epoch": res.best_epoch, "best_val_acc": float(res.best_val_acc),
            "elapsed_s": round(time.time() - t0, 1)}
    return pred, info


def main() -> int:
    cfg = load_config()
    ws_all = load_all(cfg)
    ws_main = ws_all.subset(ws_all.role == "main")
    X = ws_main.X
    y4 = np.asarray(ws_main.label)
    print(f"主线窗口 {X.shape}，4 类分布 {np.bincount(y4).tolist()}")

    out: dict = {"models": {"CNN": {}}}

    # ---------- A 4 分类 · 留一载荷 ----------
    print("\n=== A 4 分类 / 留一载荷（CNN）===")
    loo4 = {"folds": {}}
    for load in (0, 1, 2, 3):
        tr, te = ws_main.load_hp != load, ws_main.load_hp == load
        tr2, va = carve_val(ws_main, tr, val_frac=0.1)
        pred, info = fit_predict(X, y4, tr2, va, te, len(LABELS), cfg, seed=0)
        m = evaluate(y4[te], pred, LABELS)
        keep217 = ws_main.file_id[te] != 217
        fold = {"metrics": m, **info,
                "accuracy_excl_217": float((pred[keep217] == y4[te][keep217]).mean())}
        loo4["folds"][str(load)] = fold
        print(f"  {load} HP  {summarize(m)}  剔217 acc={fold['accuracy_excl_217']:.4f}  "
              f"({info['elapsed_s']}s)")
    accs = [loo4["folds"][str(k)]["metrics"]["accuracy"] for k in (0, 1, 2, 3)]
    accs217 = [loo4["folds"][str(k)]["accuracy_excl_217"] for k in (0, 1, 2, 3)]
    loo4["mean_accuracy"] = float(np.mean(accs))
    loo4["std_accuracy"] = float(np.std(accs))
    loo4["mean_accuracy_excl_217"] = float(np.mean(accs217))
    out["models"]["CNN"]["loo_4class"] = loo4
    print(f"  → LOO acc={loo4['mean_accuracy']:.4f}±{loo4['std_accuracy']:.4f}  "
          f"剔217={loo4['mean_accuracy_excl_217']:.4f}")
    save_json(out, ROOT / "results" / "metrics" / "cnn_ood.json")

    # ---------- B 未见损伤程度 ----------
    print("\n=== B 未见程度 0.028\"（训练 3 档 → 测试第 4 档，CNN）===")
    m_train = ws_all.role == "main"
    m_test = ws_all.role == "severity"
    y_all = np.asarray(ws_all.label)
    tr2, va = carve_val(ws_all, m_train, val_frac=0.1)
    pred, info = fit_predict(ws_all.X, y_all, tr2, va, m_test, len(LABELS), cfg, seed=0)
    truth = y_all[m_test]
    keep = np.isin(truth, [LABELS.index("IR"), LABELS.index("B")])
    two_class_acc = float((pred[keep] == truth[keep]).mean())
    dist = {LABELS[i]: int((pred == i).sum()) for i in range(len(LABELS))}
    sev_out = {"n_train": int(m_train.sum()), "n_test": int(m_test.sum()), **info,
               "two_class_accuracy": two_class_acc, "n_eval": int(keep.sum()),
               "pred_distribution": dist}
    out["models"]["CNN"]["unseen_severity"] = sev_out
    print(f"  IR/B 二分类 acc={two_class_acc:.4f}  预测分布={dist}  ({info['elapsed_s']}s)")
    save_json(out, ROOT / "results" / "metrics" / "cnn_ood.json")

    # ---------- C 未见外圈位置 ----------
    print("\n=== C 未见位置 @3/@12（训练 @6，CNN）===")
    m_pos = ws_all.role == "position"
    pred, info = fit_predict(ws_all.X, y_all, tr2, va, m_pos, len(LABELS), cfg, seed=0)
    truth = y_all[m_pos]
    or_idx = LABELS.index("OR")
    pos_out = {"n_test": int(m_pos.sum()), **info,
               "accuracy": float((pred == truth).mean()),
               "or_recall": float((pred == or_idx).mean()),
               "pred_distribution": {LABELS[i]: int((pred == i).sum())
                                     for i in range(len(LABELS))}}
    out["models"]["CNN"]["unseen_position"] = pos_out
    print(f"  acc={pos_out['accuracy']:.4f}  OR 召回={pos_out['or_recall']:.4f}  "
          f"预测分布={pos_out['pred_distribution']}  ({info['elapsed_s']}s)")
    save_json(out, ROOT / "results" / "metrics" / "cnn_ood.json")

    # ---------- D 10 分类 · 时间块 ----------
    print("\n=== D 10 分类 / 时间块（CNN，seed 0/1）===")
    y10 = to_labels10(ws_main)
    print(f"10 类分布 " + ", ".join(f"{c}:{int((y10 == i).sum())}" for i, c in enumerate(CLASSES)))
    runs = []
    for seed in (0, 1):
        tr, te = time_block_split(ws_main, 0.3, int(cfg["guard_windows"]))
        tr2, va = carve_val(ws_main, tr, val_frac=0.1)
        pred, info = fit_predict(X, y10, tr2, va, te, len(CLASSES), cfg, seed=seed)
        m = evaluate(y10[te], pred, CLASSES)
        m["type_accuracy"] = type_accuracy(y10[te], pred)
        runs.append({"seed": seed, "metrics": m, **info})
        print(f"  seed={seed}  {summarize(m)}  类型={m['type_accuracy']:.4f}  "
              f"({info['elapsed_s']}s)")
    agg = {
        "mean_accuracy": float(np.mean([r["metrics"]["accuracy"] for r in runs])),
        "std_accuracy": float(np.std([r["metrics"]["accuracy"] for r in runs])),
        "mean_type_accuracy": float(np.mean([r["metrics"]["type_accuracy"] for r in runs])),
        "runs": runs,
    }
    out["models"]["CNN"]["sev10_time_block"] = agg
    print(f"  → acc={agg['mean_accuracy']:.4f}  类型={agg['mean_type_accuracy']:.4f}")
    save_json(out, ROOT / "results" / "metrics" / "cnn_ood.json")

    # ---------- E 10 分类 · 留一载荷 ----------
    print("\n=== E 10 分类 / 留一载荷（CNN，seed=0）===")
    loo10 = {"folds": {}}
    for load in (0, 1, 2, 3):
        tr, te = ws_main.load_hp != load, ws_main.load_hp == load
        tr2, va = carve_val(ws_main, tr, val_frac=0.1)
        pred, info = fit_predict(X, y10, tr2, va, te, len(CLASSES), cfg, seed=0)
        m = evaluate(y10[te], pred, CLASSES)
        m["type_accuracy"] = type_accuracy(y10[te], pred)
        keep217 = ws_main.file_id[te] != 217
        loo10["folds"][str(load)] = {
            "metrics": m, **info,
            "accuracy_excl_217": float((pred[keep217] == y10[te][keep217]).mean()),
            "type_accuracy_excl_217": type_accuracy(y10[te][keep217], pred[keep217]),
        }
        print(f"  {load} HP  acc={m['accuracy']:.4f}  类型={m['type_accuracy']:.4f}  "
              f"剔217={loo10['folds'][str(load)]['accuracy_excl_217']:.4f}  ({info['elapsed_s']}s)")
    for key, src in (("mean_accuracy", "accuracy"), ("mean_type_accuracy", "type_accuracy")):
        loo10[key] = float(np.mean(
            [loo10["folds"][str(k)]["metrics"][src] for k in (0, 1, 2, 3)]))
    loo10["mean_accuracy_excl_217"] = float(np.mean(
        [loo10["folds"][str(k)]["accuracy_excl_217"] for k in (0, 1, 2, 3)]))
    out["models"]["CNN"]["sev10_loo"] = loo10
    print(f"  → LOO acc={loo10['mean_accuracy']:.4f}  类型={loo10['mean_type_accuracy']:.4f}  "
          f"剔217={loo10['mean_accuracy_excl_217']:.4f}")
    save_json(out, ROOT / "results" / "metrics" / "cnn_ood.json")
    print("\n完成 → results/metrics/cnn_ood.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
