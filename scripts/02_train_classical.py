"""02 · 经典模型基线：SVM / 随机森林，两种划分协议对照。

协议
  random_window  同一记录的重叠窗口随机分到训练/测试 —— 有泄漏，仅作上界参考
  time_block     记录内按时间块划分 + 隔离带 —— 无泄漏的同工况评测

产出
  results/metrics/classical_random_window.json
  results/metrics/classical_time_block.json
  results/models/{svm,rf}_time_block.joblib
  results/metrics/rf_importance.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                       # noqa: E402
from cwru_diag.dataset import LABELS                           # noqa: E402
from cwru_diag.evaluation import aggregate_runs, evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import make_rf, make_svm                 # noqa: E402
from cwru_diag.pipeline import (features_for, load_all, random_window_split,  # noqa: E402
                                time_block_split)

PROTOCOLS = {
    "random_window": random_window_split,
    "time_block": time_block_split,
}
MODELS = {
    "SVM": lambda seed, cfg: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=seed),
    "RF": lambda seed, cfg: make_rf(
        n_estimators=int(cfg["classical"]["rf"]["n_estimators"]),
        max_depth=cfg["classical"]["rf"]["max_depth"], seed=seed),
}


def run_protocol(name: str, split_fn, X, y, ws, feature_names, cfg, seeds) -> dict:
    out = {"protocol": name, "models": {}}
    print(f"\n=== 协议 {name} ===")
    for model_name, factory in MODELS.items():
        runs = []
        if name == "time_block":
            train_idx, test_idx = split_fn(ws, 0.3, int(cfg["guard_windows"]))
        for seed in seeds:
            if name == "random_window":
                train_idx, test_idx = split_fn(ws, 0.3, seed)
            t0 = time.time()
            clf = factory(seed, cfg)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            m = evaluate(y[test_idx], pred, LABELS)
            runs.append({"seed": seed, "metrics": m})
            print(f"  {model_name} seed={seed}  {summarize(m)}  ({time.time() - t0:.1f}s)")

            if name == "time_block" and seed == seeds[0]:
                path = ROOT / "results" / "models" / f"{model_name.lower()}_time_block.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, path)
                if model_name == "RF":
                    imp = clf.named_steps["clf"].feature_importances_
                    order = np.argsort(imp)[::-1]
                    save_json(
                        {"features": [feature_names[i] for i in order],
                         "importance": imp[order].tolist()},
                        ROOT / "results" / "metrics" / "rf_importance.json")
        agg = aggregate_runs(runs, len(LABELS))
        out["models"][model_name] = agg
        out["n_train"] = int(train_idx.sum())
        out["n_test"] = int(test_idx.sum())
        print(f"  → {model_name}  acc={agg['mean']['accuracy']:.4f}±{agg['std']['accuracy']:.4f}")
    return out


def main() -> int:
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    print("装配数据 ...")
    ws = load_all(cfg, ("main",))
    X, feature_names = features_for(ws, ("time", "freq", "env"), cfg)
    y = np.asarray(ws.label)
    print(f"特征矩阵 {X.shape}，类别 {LABELS}")

    # 各类别特征均值/标准差：给 tools/predict.py 做“证据对照”
    means = {}
    for i, name in enumerate(LABELS):
        m = y == i
        means[name] = {"mean": X[m].mean(axis=0).tolist(), "std": X[m].std(axis=0).tolist(),
                       "n": int(m.sum())}
    save_json({"features": feature_names, "classes": means},
              ROOT / "results" / "metrics" / "class_feature_stats.json")

    results = {}
    for name, fn in PROTOCOLS.items():
        res = run_protocol(name, fn, X, y, ws, feature_names, cfg, seeds)
        results[name] = res
        save_json(res, ROOT / "results" / "metrics" / f"classical_{name}.json")

    # 泄漏量化：随机窗口 vs 时间块
    leak = {}
    for m in MODELS:
        a = results["random_window"]["models"][m]["mean"]["accuracy"]
        b = results["time_block"]["models"][m]["mean"]["accuracy"]
        leak[m] = {"random_window": a, "time_block": b, "gap": a - b}
        print(f"泄漏差 {m}: {a:.4f} → {b:.4f}（{a - b:+.4f}）")
    save_json(leak, ROOT / "results" / "metrics" / "leakage_gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
