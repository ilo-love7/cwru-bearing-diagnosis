"""04 · 跨载荷泛化：留一载荷（leave-one-load-out）+ 单源 0→3 HP。

训练集与测试集来自完全不同的载荷工况（电机转速、转矩、轴承径向力都不同），
是比“同工况随机切窗”更接近现场部署的评测口径。

产出
  results/metrics/cross_load.json
  results/metrics/cross_load_0_to_3.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import make_rf, make_svm, predict_cnn, train_cnn  # noqa: E402
from cwru_diag.pipeline import carve_val, features_for, load_all  # noqa: E402

LOADS = [0, 1, 2, 3]


def _agg(fold_metrics: list[dict]) -> dict:
    """把同一折内多个随机种子的指标聚合成 mean/std/混淆矩阵。"""
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    cm = np.sum([np.asarray(m["confusion_matrix"]) for m in fold_metrics], axis=0)
    per_class = {}
    for name in fold_metrics[0]["per_class"]:
        per_class[name] = {
            k: float(np.mean([m["per_class"][name][k] for m in fold_metrics]))
            for k in ("precision", "recall", "f1")
        }
    return {
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys},
        "std": {k: float(np.std([m[k] for m in fold_metrics])) for k in keys},
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _agg_folds(fold_models: list[dict]) -> dict:
    """跨折汇总：fold_models 是每折的 {"mean": {...}, "std": {...}, ...}。"""
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    means = [fm["mean"] for fm in fold_models]
    cm = np.sum([np.asarray(fm["confusion_matrix"]) for fm in fold_models], axis=0)
    per_class = {}
    for name in fold_models[0]["per_class"]:
        per_class[name] = {
            k: float(np.mean([fm["per_class"][name][k] for fm in fold_models]))
            for k in ("precision", "recall", "f1")
        }
    return {
        "mean": {k: float(np.mean([m[k] for m in means])) for k in keys},
        "std": {k: float(np.std([m[k] for m in means])) for k in keys},
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def run_fold(ws, X, feats_ok, train_mask, test_mask, cfg, seeds, cnn_seeds, cnn_epochs):
    y = np.asarray(ws.label)
    out = {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
    for name, factory in (
        ("SVM", lambda s: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=s)),
        ("RF", lambda s: make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=s)),
    ):
        ms = []
        for s in seeds:
            clf = factory(s).fit(X[train_mask], y[train_mask])
            m = evaluate(y[test_mask], clf.predict(X[test_mask]), LABELS)
            ms.append(m)
            print(f"    {name} seed={s} {summarize(m)}")
        out[name] = _agg(ms)

    if cnn_epochs > 0 and feats_ok:
        tr2, va = carve_val(ws, train_mask, val_frac=0.1)
        ms = []
        for s in cnn_seeds:
            res = train_cnn(
                ws.X[tr2], y[tr2], ws.X[va], y[va], n_classes=len(LABELS),
                epochs=cnn_epochs, batch_size=int(cfg["cnn"]["batch_size"]),
                lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
                dropout=float(cfg["cnn"]["dropout"]), seed=s, verbose=False,
            )
            m = evaluate(y[test_mask], predict_cnn(res.model, ws.X[test_mask]), LABELS)
            ms.append(m)
            print(f"    CNN seed={s} {summarize(m)}")
        out["CNN"] = _agg(ms)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-seeds", type=int, default=1)
    ap.add_argument("--cnn-epochs", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    X, names = features_for(ws, ("time", "freq", "env"), cfg)
    y = np.asarray(ws.label)
    print(f"主实验 {len(ws)} 窗口 / {X.shape[1]} 维特征 / 类别 {LABELS}")

    # ---- 留一载荷 ----
    folds = {}
    for load in LOADS:
        print(f"\n=== 留出 {load} HP ===")
        tr = ws.load_hp != load
        te = ws.load_hp == load
        t0 = time.time()
        folds[str(load)] = run_fold(ws, X, True, tr, te, cfg, seeds,
                                    list(range(args.cnn_seeds)), args.cnn_epochs)
        print(f"  用时 {time.time() - t0:.0f}s")

    models = [m for m in ("SVM", "RF", "CNN") if m in folds["0"]]
    agg = {m: _agg_folds([folds[k][m] for k in folds]) for m in models}

    # 逐文件诊断 + 剔除异常文件 217.mat 的敏感性对照
    diag, sens = {}, {}
    for load in LOADS:
        tr, te = ws.load_hp != load, ws.load_hp == load
        clf = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0).fit(X[tr], y[tr])
        pred_all = clf.predict(X[te])
        y_te = y[te]
        rows = []
        for fid in np.unique(ws.file_id[te]):
            m = ws.file_id[te] == fid
            rows.append({
                "file_id": int(fid), "n": int(m.sum()),
                "accuracy": float((pred_all[m] == y_te[m]).mean()),
                "true": LABELS[int(y_te[m][0])],
            })
        diag[str(load)] = rows
        keep = ws.file_id[te] != 217
        sens[str(load)] = {
            "accuracy_excl_217": float((pred_all[keep] == y_te[keep]).mean()),
            "n_excl": int(keep.sum()),
        }

    result = {
        "protocol": "leave_one_load_out",
        "folds": folds,
        "aggregate": agg,
        "per_file_diagnostic": diag,
        "sensitivity_excl_217": sens,
        "n_features": len(names),
    }
    save_json(result, ROOT / "results" / "metrics" / "cross_load.json")
    print("\n=== 留一载荷汇总 ===")
    for m, a in agg.items():
        print(f"  {m:4s} acc={a['mean']['accuracy']:.4f}±{a['std']['accuracy']:.4f}  "
              f"macroF1={a['mean']['macro_f1']:.4f}")

    print("\n=== 逐文件误差（SVM, seed=0）===")
    for load in LOADS:
        bad = [r for r in diag[str(load)] if r["accuracy"] < 0.999]
        print(f"  {load} HP: " + ("全部文件正确" if not bad else
              "，".join(f"{r['file_id']}.mat({r['true']}) acc={r['accuracy']:.3f}" for r in bad)))
    print("\n=== 剔除 217.mat 的敏感性 ===")
    for load in LOADS:
        v = sens[str(load)]
        print(f"  {load} HP: acc={v['accuracy_excl_217']:.4f}（n={v['n_excl']}）")

    # ---- 单源 0 → 3 HP（提案里的口径）----
    print("\n=== 单源 0 HP → 3 HP ===")
    tr = ws.load_hp == 0
    te = ws.load_hp == 3
    single = run_fold(ws, X, True, tr, te, cfg, seeds,
                      list(range(args.cnn_seeds)), args.cnn_epochs)
    single_out = {
        "protocol": "single_source_0_to_3",
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "models": {m: single[m] for m in models},
    }
    save_json(single_out, ROOT / "results" / "metrics" / "cross_load_0_to_3.json")
    for m in models:
        print(f"  {m:4s} acc={single[m]['mean']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
