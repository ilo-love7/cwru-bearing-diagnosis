"""13 · 跨载荷泛化深化：4×4 有向迁移矩阵 + 源工况数量曲线 + 载荷信息残留探针。

既有口径只有"留一载荷（3→1）"与"单源 0→3"两个点。本脚本补齐：

  A 迁移矩阵     训练载荷 i → 测试载荷 j 的 12 个有向对（对角为载荷内时间块参考），
                 全特征与仅包络阶次两套特征，SVM/RF 各 5 种子；0→3 与 3→0 补 CNN
  B 源工况数量   固定目标载荷（3HP 与 0HP），源从其余 3 个载荷的 1/2/3 个组合训练，
                 看"多采集几个工况"对泛化的边际收益
  C 载荷探针     用特征预测载荷标签（时间块划分，机会水平 25%）——量化"阶次归一化
                 之后特征里还剩多少载荷信息"，检验报告 §4.1 的机理解释

产出
  results/metrics/transfer_matrix.json
  results/figures/fig22_transfer.png
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
from cwru_diag.evaluation import evaluate, save_json            # noqa: E402
from cwru_diag.models import make_rf, make_svm, predict_cnn, train_cnn  # noqa: E402
from cwru_diag.pipeline import (carve_val, features_for, load_all,  # noqa: E402
                                time_block_split)

LOADS = [0, 1, 2, 3]
GROUP_SETS = {
    "all": ("time", "freq", "env"),
    "env_only": ("env",),
    "time": ("time",),
    "freq": ("freq",),
}


def fit_predict(X, y, tr, te, factory, seeds):
    accs = []
    for s in seeds:
        pred = factory(s).fit(X[tr], y[tr]).predict(X[te])
        accs.append(float((pred == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs)), accs


def main() -> int:
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    y = np.asarray(ws.label)
    Xs, names = {}, {}
    for tag, groups in GROUP_SETS.items():
        Xs[tag], names[tag] = features_for(ws, groups, cfg)
    print(f"{len(ws)} 窗口 / 特征组 {[(k, Xs[k].shape[1]) for k in GROUP_SETS]}")

    factories = {
        "SVM": lambda s: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=s),
        "RF": lambda s: make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=s),
    }

    out: dict = {"A_transfer_matrix": {}, "B_source_diversity": {},
                 "C_load_probe": {}}

    # ---------- A 4×4 有向迁移矩阵 ----------
    print("\n=== A 迁移矩阵（行=训练载荷，列=测试载荷）===")
    for feat in ("all", "env_only"):
        X = Xs[feat]
        mat = {}
        for mname, factory in factories.items():
            grid = np.zeros((4, 4)); grid_std = np.zeros((4, 4))
            for i, src in enumerate(LOADS):
                for j, dst in enumerate(LOADS):
                    if src == dst:
                        tr, te = time_block_split(ws.subset(ws.load_hp == src),
                                                  0.3, int(cfg["guard_windows"]))
                        # subset 后掩码回到全集坐标
                        idx = np.where(ws.load_hp == src)[0]
                        tr_full = np.zeros(len(ws), bool); tr_full[idx[tr]] = True
                        te_full = np.zeros(len(ws), bool); te_full[idx[te]] = True
                    else:
                        tr_full = ws.load_hp == src
                        te_full = ws.load_hp == dst
                    mu, sd, _ = fit_predict(X, y, tr_full, te_full, factory, seeds)
                    grid[i, j], grid_std[i, j] = mu, sd
            mat[mname] = {"mean": grid.tolist(), "std": grid_std.tolist()}
            print(f"  [{feat}/{mname}]")
            print("     " + "  ".join(f"{d}HP" for d in LOADS))
            for i, src in enumerate(LOADS):
                print(f"  {src}HP " + "  ".join(f"{grid[i, j]:.3f}" for j in range(4)))
        out["A_transfer_matrix"][feat] = mat

    # CNN 不对称性抽查：0→3 与 3→0
    cnn_asym = {}
    for src, dst in ((0, 3), (3, 0)):
        tr, te = ws.load_hp == src, ws.load_hp == dst
        tr2, va = carve_val(ws, tr, val_frac=0.1)
        t0 = time.time()
        res = train_cnn(ws.X[tr2], y[tr2], ws.X[va], y[va], n_classes=len(LABELS),
                        epochs=20, batch_size=int(cfg["cnn"]["batch_size"]),
                        lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
                        dropout=float(cfg["cnn"]["dropout"]), seed=0, verbose=False)
        pred = predict_cnn(res.model, ws.X[te])
        acc = float((pred == y[te]).mean())
        cnn_asym[f"{src}_to_{dst}"] = {"accuracy": acc, "n_train": int(tr.sum()),
                                       "n_test": int(te.sum())}
        print(f"  CNN {src}→{dst}: acc={acc:.4f} ({time.time() - t0:.0f}s)")
    out["A_transfer_matrix"]["cnn_asymmetry"] = cnn_asym

    # ---------- B 源工况数量曲线 ----------
    print("\n=== B 源工况数量（目标 3HP / 0HP）===")
    from itertools import combinations
    diversity = {}
    for dst in (3, 0):
        pool = [l for l in LOADS if l != dst]
        te = ws.load_hp == dst
        rows = []
        for k in (1, 2, 3):
            for combo in combinations(pool, k):
                tr = np.isin(ws.load_hp, combo)
                entry = {"sources": list(combo), "n_train": int(tr.sum())}
                for mname, factory in (("SVM", factories["SVM"]), ("RF", factories["RF"])):
                    mu, sd, _ = fit_predict(Xs["all"], y, tr, te, factory, seeds)
                    entry[mname] = {"accuracy": mu, "std": sd}
                rows.append(entry)
                print(f"  目标{dst}HP 源{combo}: " +
                      " ".join(f"{m}={entry[m]['accuracy']:.3f}" for m in ("SVM", "RF")))
        by_k = {}
        for k in (1, 2, 3):
            sel = [r for r in rows if len(r["sources"]) == k]
            by_k[str(k)] = {
                m: {"mean_accuracy": float(np.mean([r[m]["accuracy"] for r in sel])),
                    "min_accuracy": float(np.min([r[m]["accuracy"] for r in sel])),
                    "max_accuracy": float(np.max([r[m]["accuracy"] for r in sel]))}
                for m in ("SVM", "RF")}
        diversity[str(dst)] = {"per_subset": rows, "by_source_count": by_k}
    out["B_source_diversity"] = diversity

    # ---------- C 载荷信息残留探针 ----------
    print("\n=== C 载荷探针（预测载荷标签，机会水平 25%）===")
    y_load = np.asarray(ws.load_hp)
    probe_factory = lambda s: make_rf(n_estimators=200, seed=s)
    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    probe = {}
    for tag in ("time", "freq", "env_only", "all"):
        accs = []
        for s in seeds[:3]:
            pred = probe_factory(s).fit(Xs[tag][tr], y_load[tr]).predict(Xs[tag][te])
            accs.append(float((pred == y_load[te]).mean()))
        probe[tag] = {"accuracy": float(np.mean(accs)), "std": float(np.std(accs)),
                      "per_class": {}}
        print(f"  {tag:8s} 载荷探针 acc={probe[tag]['accuracy']:.4f}")
    # 分类别探针（排除"类别构成差异"这一捷径）
    for li, lname in enumerate(LABELS):
        m_cls = y == li
        idx = np.where(m_cls)[0]
        sub = ws.subset(m_cls)
        tr_s, te_s = time_block_split(sub, 0.3, int(cfg["guard_windows"]))
        accs = []
        for s in seeds[:3]:
            pred = probe_factory(s).fit(Xs["all"][idx][tr_s], y_load[idx][tr_s]) \
                                   .predict(Xs["all"][idx][te_s])
            accs.append(float((pred == y_load[idx][te_s]).mean()))
        probe["all"]["per_class"][lname] = float(np.mean(accs))
        print(f"    类内探针 {lname:6s} acc={probe['all']['per_class'][lname]:.4f}")
    out["C_load_probe"] = {"chance": 0.25, "probe_model": "RF(n=200)", "groups": probe}

    save_json(out, ROOT / "results" / "metrics" / "transfer_matrix.json")
    print("\n完成 → results/metrics/transfer_matrix.json")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt
    from cwru_diag.plots import save, setup_style
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for ax, feat, title in ((axes[0], "env_only", "迁移矩阵（仅包络阶次）"),
                            (axes[1], "all", "迁移矩阵（全特征）")):
        grid = np.array(out["A_transfer_matrix"][feat]["SVM"]["mean"])
        im = ax.imshow(grid, vmin=0.6, vmax=1.0, cmap="RdYlGn")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{d}HP" for d in LOADS])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{s}HP" for s in LOADS])
        ax.set_xlabel("测试载荷"); ax.set_ylabel("训练载荷")
        ax.set_title(title + " · SVM"); ax.grid(False)

    ax = axes[2]
    ks = [1, 2, 3]
    for dst, col in ((3, "#E45756"), (0, "#4C78A8")):
        by_k = out["B_source_diversity"][str(dst)]["by_source_count"]
        means = [by_k[str(k)]["SVM"]["mean_accuracy"] for k in ks]
        mins = [by_k[str(k)]["SVM"]["min_accuracy"] for k in ks]
        maxs = [by_k[str(k)]["SVM"]["max_accuracy"] for k in ks]
        ax.plot(ks, means, "o-", color=col, label=f"目标 {dst}HP")
        ax.fill_between(ks, mins, maxs, color=col, alpha=0.15)
    ax.axhline(0.25, color="#9AA3AD", lw=0.7, ls="--")
    ax.set_xticks(ks); ax.set_xlabel("源工况数量")
    ax.set_ylabel("SVM 准确率"); ax.set_ylim(0.5, 1.02)
    ax.set_title("B 源工况数量 vs 泛化（阴影=组合范围）")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = save(fig, "fig22_transfer.png")
    print(f"图 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
