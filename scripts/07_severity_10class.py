"""07 · 类型 × 损伤程度 10 分类：把"诊断到什么程度"单独作为任务。

4 分类（正常/内圈/外圈/滚动体）在同工况内已经饱和（见 classical_time_block.json），
无法区分方法与划分口径的优劣。这里把损伤尺寸并入标签：
    Normal, IR007/IR014/IR021, OR007/OR014/OR021, B007/B014/B021

除了 10 分类准确率，还报告"类型正确率"（忽略尺寸，只看故障部位），
用来回答：错的时候是错在部位还是错在尺寸。

产出
  results/metrics/severity_10class.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                       # noqa: E402
from cwru_diag.evaluation import aggregate_runs, evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import make_rf, make_svm                 # noqa: E402
from cwru_diag.pipeline import (features_for, load_all, random_window_split,  # noqa: E402
                                time_block_split)

TYPES = ["Normal", "IR", "OR", "B"]
SEVS = [0.007, 0.014, 0.021]
CLASSES = ["Normal"] + [f"{t}{int(s * 1000):03d}" for t in ("IR", "OR", "B") for s in SEVS]
TYPE_OF = {c: ("Normal" if c == "Normal" else c[:2]) for c in CLASSES}


def to_labels(ws) -> np.ndarray:
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


def run(fit_fn, X, y, tr, te, seeds) -> tuple[dict, list[dict]]:
    runs = []
    for s in seeds:
        t0 = time.time()
        clf = fit_fn(s).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        m = evaluate(y[te], pred, CLASSES)
        m["type_accuracy"] = type_accuracy(y[te], pred)
        runs.append({"seed": s, "metrics": m})
        print(f"    seed={s} {summarize(m)}  类型正确率={m['type_accuracy']:.4f}  "
              f"({time.time() - t0:.1f}s)")
    return aggregate_runs(runs, len(CLASSES)), runs


def main() -> int:
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws = load_all(cfg, ("main",))
    X, names = features_for(ws, ("time", "freq", "env"), cfg)
    y = to_labels(ws)
    print(f"{len(ws)} 窗口 / {len(CLASSES)} 类 / {X.shape[1]} 维特征")
    print("类别分布: " + ", ".join(f"{c}:{int((y == i).sum())}" for i, c in enumerate(CLASSES)))

    models = {
        "SVM": lambda s: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=s),
        "RF": lambda s: make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=s),
    }
    out = {"classes": CLASSES, "n_windows": int(len(ws)), "n_features": len(names), "protocols": {}}

    for protocol, split_fn in (("random_window", random_window_split),
                               ("time_block", time_block_split)):
        print(f"\n=== 10 分类 / {protocol} ===")
        res = {"models": {}}
        for name, fn in models.items():
            if protocol == "random_window":
                tr, te = split_fn(ws, 0.3, seeds[0])
            else:
                tr, te = split_fn(ws, 0.3, int(cfg["guard_windows"]))
            agg, runs = run(fn, X, y, tr, te, seeds)
            agg["mean_type_accuracy"] = float(np.mean([r["metrics"]["type_accuracy"] for r in runs]))
            res["models"][name] = agg
            res["n_test"] = int(te.sum())
            print(f"  → {name} acc={agg['mean']['accuracy']:.4f}  "
                  f"类型={agg['mean_type_accuracy']:.4f}")
        out["protocols"][protocol] = res
        save_json(out, ROOT / "results" / "metrics" / "severity_10class.json")

    # 留一载荷
    print("\n=== 10 分类 / 留一载荷 ===")
    loo = {"models": {}, "folds": {}}
    for load in [0, 1, 2, 3]:
        tr, te = ws.load_hp != load, ws.load_hp == load
        fold = {}
        for name, fn in models.items():
            agg, runs = run(fn, X, y, tr, te, seeds)
            agg["mean_type_accuracy"] = float(np.mean(
                [r["metrics"]["type_accuracy"] for r in runs]))
            fold[name] = agg
        loo["folds"][str(load)] = fold
        print(f"  {load} HP: " + "  ".join(
            f"{m} acc={fold[m]['mean']['accuracy']:.3f}/类型={fold[m]['mean_type_accuracy']:.3f}"
            for m in models))
    for name in models:
        accs = [loo["folds"][str(k)][name]["mean"]["accuracy"] for k in [0, 1, 2, 3]]
        taccs = [loo["folds"][str(k)][name]["mean_type_accuracy"] for k in [0, 1, 2, 3]]
        loo["models"][name] = {
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "mean_type_accuracy": float(np.mean(taccs)),
            "per_fold_accuracy": {str(k): float(v) for k, v in zip([0, 1, 2, 3], accs)},
        }
        print(f"  → {name} LOO acc={loo['models'][name]['mean_accuracy']:.4f}  "
              f"类型={loo['models'][name]['mean_type_accuracy']:.4f}")
    # 剔除异常文件 217.mat 的敏感性（该文件官方列在 3 HP，但实测 1752 rpm、41 s，
    # 与同工况其余文件不符，4 分类下会被整体误判为滚动体故障）
    sens = {}
    for load in [0, 1, 2, 3]:
        tr, te = ws.load_hp != load, ws.load_hp == load
        keep = ws.file_id[te] != 217
        row = {}
        for name, fn in models.items():
            pred = fn(0).fit(X[tr], y[tr]).predict(X[te])
            row[name] = {
                "accuracy_excl_217": float((pred[keep] == y[te][keep]).mean()),
                "type_accuracy_excl_217": type_accuracy(y[te][keep], pred[keep]),
                "n_excl": int(keep.sum()),
            }
        sens[str(load)] = row
        print(f"  敏感性 {load} HP 剔除 217: " + "  ".join(
            f"{m} acc={row[m]['accuracy_excl_217']:.4f}" for m in models))
    loo["sensitivity_excl_217"] = sens
    out["protocols"]["leave_one_load_out"] = loo
    save_json(out, ROOT / "results" / "metrics" / "severity_10class.json")
    print("\n完成 → results/metrics/severity_10class.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
