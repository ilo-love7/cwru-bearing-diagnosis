"""05 · 三组补充实验：特征消融 / 未见损伤程度 / 外圈故障位置泛化。

A 特征消融   在留一载荷口径下比较 时域 / 时域+频域 / 仅包络阶次 / 全特征
B 未见损伤   训练只看 0.007/0.014/0.021，测试 0.028（官方额外提供的 8 个文件）
C 位置泛化   训练只看外圈 6 点钟方向，测试 3 点/12 点方向

产出
  results/metrics/ablation_features.json
  results/metrics/severity_extrapolation.json
  results/metrics/position_generalization.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import make_rf, make_svm                  # noqa: E402
from cwru_diag.pipeline import features_for, load_all           # noqa: E402

LOADS = [0, 1, 2, 3]
GROUP_SETS = {
    "time": ("time",),
    "time+freq": ("time", "freq"),
    "env_only": ("env",),
    "all": ("time", "freq", "env"),
}


def loo_svm(ws, X, cfg, seeds) -> dict:
    """留一载荷 + SVM，返回每折与汇总指标。"""
    y = np.asarray(ws.label)
    folds = {}
    for load in LOADS:
        tr, te = ws.load_hp != load, ws.load_hp == load
        accs, f1s = [], []
        for s in seeds:
            clf = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=s).fit(X[tr], y[tr])
            m = evaluate(y[te], clf.predict(X[te]), LABELS)
            accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
        folds[str(load)] = {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
    return {
        "folds": folds,
        "mean_accuracy": float(np.mean([v["accuracy"] for v in folds.values()])),
        "std_accuracy": float(np.std([v["accuracy"] for v in folds.values()])),
        "mean_macro_f1": float(np.mean([v["macro_f1"] for v in folds.values()])),
    }


def main() -> int:
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws_all = load_all(cfg)
    ws = ws_all.subset(ws_all.role == "main")
    y = np.asarray(ws.label)

    # ---------- A 特征消融 ----------
    print("=== A 特征消融（留一载荷 / SVM）===")
    ablation = {}
    for name, groups in GROUP_SETS.items():
        X, cols = features_for(ws, groups, cfg)
        res = loo_svm(ws, X, cfg, seeds)
        res["n_features"] = len(cols)
        ablation[name] = res
        print(f"  {name:10s} ({len(cols):2d} 维)  LOO acc={res['mean_accuracy']:.4f}"
              f"±{res['std_accuracy']:.4f}")
    save_json(ablation, ROOT / "results" / "metrics" / "ablation_features.json")

    # ---------- B 未见损伤程度 ----------
    print("\n=== B 未见损伤程度：训练 3 档 → 测试 0.028\" ===")
    X_all, _ = features_for(ws_all, ("time", "freq", "env"), cfg)
    m_train = (ws_all.role == "main").astype(bool)
    m_test = (ws_all.role == "severity").astype(bool)
    y_all = np.asarray(ws_all.label)
    sev_out = {"n_train": int(m_train.sum()), "n_test": int(m_test.sum()), "models": {}}
    for name, clf in (
        ("SVM", make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0)),
        ("RF", make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=0)),
    ):
        clf.fit(X_all[m_train], y_all[m_train])
        pred = clf.predict(X_all[m_test])
        truth = y_all[m_test]
        keep = np.isin(truth, [LABELS.index("IR"), LABELS.index("B")])
        err = (pred[keep] != truth[keep]).astype(float)
        two_class_acc = float(1 - err.mean())
        dist = {LABELS[i]: int((pred == i).sum()) for i in range(len(LABELS))}
        sev_out["models"][name] = {
            "two_class_accuracy": two_class_acc,
            "n_eval": int(keep.sum()),
            "pred_distribution": dist,
            "confusion_matrix_2class": [
                [int(((truth == LABELS.index("IR")) & (pred == LABELS.index("IR"))).sum()),
                 int(((truth == LABELS.index("IR")) & (pred == LABELS.index("B"))).sum())],
                [int(((truth == LABELS.index("B")) & (pred == LABELS.index("IR"))).sum()),
                 int(((truth == LABELS.index("B")) & (pred == LABELS.index("B"))).sum())],
            ],
        }
        print(f"  {name}: IR/B 二分类 acc={two_class_acc:.4f}  预测分布={dist}")
    save_json(sev_out, ROOT / "results" / "metrics" / "severity_extrapolation.json")

    # ---------- C 外圈故障位置泛化 ----------
    print("\n=== C 外圈位置：训练 @6 点 → 测试 @3/@12 点 ===")
    m_pos = (ws_all.role == "position").astype(bool)
    pos_out = {"n_test": int(m_pos.sum()), "models": {}}
    for name, clf in (
        ("SVM", make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0)),
        ("RF", make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=0)),
    ):
        clf.fit(X_all[m_train], y_all[m_train])
        pred = clf.predict(X_all[m_pos])
        truth = y_all[m_pos]
        or_idx = LABELS.index("OR")
        recall_or = float((pred == or_idx).mean())
        acc = float((pred == truth).mean())
        dist = {LABELS[i]: int((pred == i).sum()) for i in range(len(LABELS))}
        pos_out["models"][name] = {
            "accuracy": acc, "or_recall": recall_or, "pred_distribution": dist,
            "n_windows": int(m_pos.sum()),
        }
        print(f"  {name}: acc={acc:.4f}  OR 召回={recall_or:.4f}  预测分布={dist}")
    save_json(pos_out, ROOT / "results" / "metrics" / "position_generalization.json")

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
