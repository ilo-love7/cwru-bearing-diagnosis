"""14 · 10 分类扩展口径 + 特征消融 × 划分协议。

既有 10 分类只报了整体准确率与类型正确率两个数。本脚本补齐：

  A 序数/分层指标   LOO 与时间块两个协议下：类型正确率、条件尺寸正确率、
                    差 ≤1 档比例、平均 |Δ档|；逐类召回表 + 尺寸档位召回
  B 消融 × 协议     特征组（time / freq / time+freq / env_only / all）
                    × 协议（时间块 / 留一载荷）× 模型（SVM/RF）——
                    检验"仅包络阶次最强"是否只属于跨载荷口径

产出
  results/metrics/10class_ordinal_ablation.json
  results/figures/fig23_10class_ordinal.png
  results/figures/fig24_ablation_protocol.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.evaluation import save_json                      # noqa: E402
from cwru_diag.models import make_rf, make_svm                  # noqa: E402
from cwru_diag.ordinal import CLASSES10, hierarchical_scores, to_labels10  # noqa: E402
from cwru_diag.pipeline import (features_for, load_all,          # noqa: E402
                                time_block_split)

LOADS = [0, 1, 2, 3]
GROUP_SETS = {
    "time": ("time",),
    "freq": ("freq",),
    "time+freq": ("time", "freq"),
    "env_only": ("env",),
    "all": ("time", "freq", "env"),
}


def run_seeded(factory, X, y, tr, te, seeds):
    """多种子分层指标聚合 + 混淆矩阵求和。"""
    scores, cm_sum = [], np.zeros((len(CLASSES10), len(CLASSES10)), dtype=int)
    for s in seeds:
        pred = factory(s).fit(X[tr], y[tr]).predict(X[te])
        scores.append(hierarchical_scores(y[te], pred))
        for t, p in zip(y[te], pred):
            cm_sum[t, p] += 1
    keys = ["type_accuracy", "size_level_accuracy", "size_acc_given_type",
            "size_within1_given_type", "mean_abs_level_err_given_type"]
    agg = {k: float(np.mean([s[k] for s in scores])) for k in keys}
    agg["std_type_accuracy"] = float(np.std([s["type_accuracy"] for s in scores]))
    agg["std_size_level_accuracy"] = float(np.std([s["size_level_accuracy"] for s in scores]))
    agg["confusion_matrix"] = cm_sum.tolist()
    return agg, scores


def per_class_recall(cm: np.ndarray) -> dict:
    tot = cm.sum(axis=1)
    return {CLASSES10[i]: float(cm[i, i] / tot[i]) if tot[i] else float("nan")
            for i in range(len(CLASSES10))}


def main() -> int:
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    y = to_labels10(ws)
    print(f"{len(ws)} 窗口 / {len(CLASSES10)} 类")
    print("类别分布: " + ", ".join(f"{c}:{int((y == i).sum())}" for i, c in enumerate(CLASSES10)))

    factories = {
        "SVM": lambda s: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=s),
        "RF": lambda s: make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=s),
    }

    out: dict = {"classes": CLASSES10, "A_ordinal": {}, "B_ablation_protocol": {}}

    # ---------- A 序数/分层指标 ----------
    print("\n=== A 10 分类序数/分层指标 ===")
    X_all, names_all = features_for(ws, ("time", "freq", "env"), cfg)

    # A1 时间块协议
    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    tb = {"n_test": int(te.sum()), "models": {}}
    for mname, factory in factories.items():
        agg, _ = run_seeded(factory, X_all, y, tr, te, seeds)
        agg["per_class_recall"] = per_class_recall(np.array(agg["confusion_matrix"]))
        tb["models"][mname] = agg
        print(f"  时间块 {mname}: type={agg['type_accuracy']:.4f} "
              f"size|type={agg['size_acc_given_type']:.4f} "
              f"≤1档={agg['size_within1_given_type']:.4f} "
              f"mean|Δ|={agg['mean_abs_level_err_given_type']:.4f}")
    out["A_ordinal"]["time_block"] = tb

    # A2 留一载荷（折汇总 + 汇总混淆）
    loo = {"folds": {}, "models": {}}
    cm_total = {m: np.zeros((len(CLASSES10), len(CLASSES10)), dtype=int) for m in factories}
    for load in LOADS:
        tr, te = ws.load_hp != load, ws.load_hp == load
        fold = {"n_test": int(te.sum())}
        for mname, factory in factories.items():
            agg, _ = run_seeded(factory, X_all, y, tr, te, seeds)
            cm_total[mname] += np.array(agg["confusion_matrix"])
            fold[mname] = agg
        loo["folds"][str(load)] = fold
        print(f"  LOO {load}HP: " + "  ".join(
            f"{m} type={fold[m]['type_accuracy']:.3f} size|type={fold[m]['size_acc_given_type']:.3f}"
            for m in factories))
    for mname in factories:
        folds = [loo["folds"][str(l)][mname] for l in LOADS]
        loo["models"][mname] = {
            "mean_type_accuracy": float(np.mean([f["type_accuracy"] for f in folds])),
            "mean_size_level_accuracy": float(np.mean([f["size_level_accuracy"] for f in folds])),
            "mean_size_acc_given_type": float(np.mean([f["size_acc_given_type"] for f in folds])),
            "mean_size_within1_given_type": float(np.mean([f["size_within1_given_type"] for f in folds])),
            "mean_abs_level_err_given_type": float(np.mean([f["mean_abs_level_err_given_type"] for f in folds])),
            "per_class_recall": per_class_recall(cm_total[mname]),
            "confusion_matrix_sum_over_folds": cm_total[mname].tolist(),
        }
        r = loo["models"][mname]
        print(f"  → {mname} LOO: type={r['mean_type_accuracy']:.4f} "
              f"size|type={r['mean_size_acc_given_type']:.4f} "
              f"≤1档={r['mean_size_within1_given_type']:.4f}")
    out["A_ordinal"]["leave_one_load_out"] = loo

    # ---------- B 消融 × 协议 ----------
    print("\n=== B 特征消融 × 划分协议（4 分类，与 §3.3 同任务）===")
    y4 = np.asarray(ws.label)
    from cwru_diag.dataset import LABELS
    ablation = {}
    for gname, groups in GROUP_SETS.items():
        X, _ = features_for(ws, groups, cfg)
        entry = {}
        for protocol in ("time_block", "leave_one_load_out"):
            prot = {}
            for mname, factory in factories.items():
                if protocol == "time_block":
                    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
                    accs = []
                    for s in seeds:
                        pred = factory(s).fit(X[tr], y4[tr]).predict(X[te])
                        accs.append(float((pred == y4[te]).mean()))
                    prot[mname] = {"mean_accuracy": float(np.mean(accs)),
                                   "std_accuracy": float(np.std(accs))}
                else:
                    accs = []
                    for load in LOADS:
                        tr, te = ws.load_hp != load, ws.load_hp == load
                        fold_accs = []
                        for s in seeds:
                            pred = factory(s).fit(X[tr], y4[tr]).predict(X[te])
                            fold_accs.append(float((pred == y4[te]).mean()))
                        accs.append(float(np.mean(fold_accs)))
                    prot[mname] = {"mean_accuracy": float(np.mean(accs)),
                                   "std_accuracy": float(np.std(accs)),
                                   "per_fold": {str(l): v for l, v in zip(LOADS, accs)}}
            entry[protocol] = prot
            print(f"  {gname:10s} {protocol:18s} " +
                  "  ".join(f"{m}={entry[protocol][m]['mean_accuracy']:.4f}"
                            for m in factories))
        ablation[gname] = entry
    out["B_ablation_protocol"] = {"labels4": LABELS, "groups": ablation}

    save_json(out, ROOT / "results" / "metrics" / "10class_ordinal_ablation.json")
    print("\n完成 → results/metrics/10class_ordinal_ablation.json")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt
    from cwru_diag.plots import CLASS_COLORS, save, setup_style
    setup_style()

    # 图 23：LOO 逐类召回 + 分层指标对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    width = 0.38
    x = np.arange(len(CLASSES10))
    for k, (mname, col) in enumerate((("SVM", "#4C78A8"), ("RF", "#F2A93B"))):
        rec = out["A_ordinal"]["leave_one_load_out"]["models"][mname]["per_class_recall"]
        vals = [rec[c] for c in CLASSES10]
        ax.bar(x + (k - 0.5) * width, vals, width, label=mname, color=col)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES10, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("召回率"); ax.set_ylim(0, 1.02)
    ax.set_title("A LOO 10 分类逐类召回（4 折求和口径）")
    ax.legend(fontsize=7)

    ax = axes[1]
    metrics = ["mean_type_accuracy", "mean_size_acc_given_type",
               "mean_size_within1_given_type", "mean_size_level_accuracy"]
    pretty = ["类型正确", "尺寸对|类型对", "≤1档|类型对", "尺寸档对(全体)"]
    x = np.arange(len(metrics))
    for k, (mname, col) in enumerate((("SVM", "#4C78A8"), ("RF", "#F2A93B"))):
        vals = [out["A_ordinal"]["leave_one_load_out"]["models"][mname][m] for m in metrics]
        ax.bar(x + (k - 0.5) * width, vals, width, label=mname, color=col)
    ax.set_xticks(x); ax.set_xticklabels(pretty, fontsize=7.5)
    ax.set_ylim(0.8, 1.02); ax.set_ylabel("准确率")
    ax.set_title("A 分层指标：错尺寸远多于错部位")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p23 = save(fig, "fig23_10class_ordinal.png")

    # 图 24：消融 × 协议
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    groups = list(GROUP_SETS)
    for ax, mname in zip(axes, ("SVM", "RF")):
        for protocol, col, marker in (("time_block", "#4C78A8", "o"),
                                      ("leave_one_load_out", "#E45756", "s")):
            accs = [out["B_ablation_protocol"]["groups"][g][protocol][mname]["mean_accuracy"]
                    for g in groups]
            ax.plot(range(len(groups)), accs, marker=marker, color=col, label=protocol)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel("4 分类准确率")
        ax.set_title(f"B 消融 × 协议 · {mname}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p24 = save(fig, "fig24_ablation_protocol.png")
    print(f"图 → {p23}\n图 → {p24}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
