"""06 · 汇总图表：把 metrics/*.json 渲染成 README 与报告用图。

产出（results/figures）
  fig10_leakage.png             随机切窗 vs 时间块划分
  fig11_crossload.png           留一载荷准确率
  fig12_confusion_crossload.png 留一载荷累计混淆矩阵
  fig13_ablation.png            特征组消融
  fig14_cnn_curves.png          CNN 训练曲线
  fig15_severity_position.png   未见损伤程度 / 外圈位置
  fig16_feature_importance.png  随机森林特征重要性
  fig18_10class.png             类型×程度 10 分类
  hero.png                      结论图：难度阶梯 + 加噪退化
docs/figures 同步一份精选图。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.dataset import LABELS                       # noqa: E402
from cwru_diag.plots import (CLASS_COLORS, grouped_bars, save,  # noqa: E402
                             setup_style)

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
MODEL_COLORS = {"SVM": "#4C78A8", "RF": "#54A24B", "CNN": "#E45756"}


def load(name: str) -> dict | None:
    p = MET / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_leakage() -> None:
    rw, tb = load("classical_random_window.json"), load("classical_time_block.json")
    cw, cb = load("cnn_random_window.json"), load("cnn_time_block.json")
    if not (rw and tb):
        return
    series = {}
    for model in ("SVM", "RF"):
        series[model] = [rw["models"][model]["mean"]["accuracy"],
                         tb["models"][model]["mean"]["accuracy"]]
    if cw and cb:
        series["CNN"] = [cw["models"]["CNN"]["mean"]["accuracy"],
                         cb["models"]["CNN"]["mean"]["accuracy"]]
    grouped_bars(["随机切窗（有泄漏）", "时间块划分（无泄漏）"], series,
                 "准确率", "同一工况内的划分口径：任务已饱和，泄漏差被天花板掩盖",
                 "fig10_leakage.png", ylim=(0.90, 1.005), hline=1.0)


def fig_crossload() -> None:
    d = load("cross_load.json")
    if not d:
        return
    models = [m for m in ("SVM", "RF", "CNN") if m in d["aggregate"]]
    folds = sorted(d["folds"].keys(), key=int)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(folds))
    for m in models:
        vals = [d["folds"][f][m]["mean"]["accuracy"] for f in folds]
        ax.plot(x, vals, "o-", lw=1.6, ms=5, color=MODEL_COLORS[m], label=m)
    ax.annotate("217.mat（官方 3 HP，实测 1752 rpm）\n478 窗口全部误判为滚动体",
                xy=(3, d["folds"]["3"]["SVM"]["mean"]["accuracy"]),
                xytext=(1.95, 0.79), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
    ax.set_xticks(x, [f"{f} HP" for f in folds])
    ax.set_xlabel("留出的载荷工况")
    ax.set_ylabel("准确率")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("跨载荷泛化：留一载荷（4 分类）")
    ax.legend(ncol=len(models), loc="lower left", fontsize=8)
    save(fig, "fig11_crossload.png")


def fig_confusion() -> None:
    d = load("cross_load.json")
    if not d:
        return
    models = [m for m in ("SVM", "RF", "CNN") if m in d["aggregate"]]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = np.asarray(d["aggregate"][m]["confusion_matrix"], dtype=float)
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(f"{m} · acc={d['aggregate'][m]['mean']['accuracy']:.3f}")
        ax.grid(False)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 0.55 else "#333333")
    fig.suptitle("留一载荷累计混淆矩阵（行归一化，含 217.mat）", y=1.02)
    fig.tight_layout()
    save(fig, "fig12_confusion_crossload.png")


def fig_ablation() -> None:
    d = load("ablation_features.json")
    if not d:
        return
    order = [k for k in ["time", "time+freq", "all", "env_only"] if k in d]
    names = {"time": "时域", "time+freq": "时域+频域",
             "all": "全特征", "env_only": "仅包络阶次"}
    colors = {"time": "#4C78A8", "time+freq": "#72B7B2",
              "all": "#B279A2", "env_only": "#F2A93B"}
    vals = [d[k]["mean_accuracy"] for k in order]
    errs = [d[k]["std_accuracy"] for k in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(range(len(vals)), vals, 0.62, yerr=errs, capsize=3,
                  color=[colors[k] for k in order], edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(order)), [f"{names[k]}\n({d[k]['n_features']} 维)" for k in order])
    ax.set_ylim(0.6, 1.06)
    ax.set_ylabel("留一载荷平均准确率")
    ax.set_title("特征组消融：物理阶次特征单独用最强，堆特征反而更差")
    save(fig, "fig13_ablation.png")


def fig_cnn_curves() -> None:
    d = load("cnn_time_block.json")
    if not d:
        return
    runs = d["models"]["CNN"]["seeds"]
    hist = runs[0]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axes[0].plot(hist["train_loss"], label="训练", color="#4C78A8")
    axes[0].plot(hist["val_loss"], label="验证", color="#E45756")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交叉熵")
    axes[0].set_title("损失")
    axes[0].legend(fontsize=8)
    axes[1].plot(hist["val_acc"], color="#54A24B")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("验证准确率")
    axes[1].set_title(f"验证准确率（最优 epoch {runs[0]['best_epoch']}）")
    fig.suptitle("1D-CNN 训练过程（时间块划分，seed=0）", y=1.02)
    fig.tight_layout()
    save(fig, "fig14_cnn_curves.png")


def fig_severity_position() -> None:
    sev, pos = load("severity_extrapolation.json"), load("position_generalization.json")
    if not (sev and pos):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    ax = axes[0]
    models = list(sev["models"])
    bottom = np.zeros(len(models))
    for lab in LABELS:
        vals = np.array([sev["models"][m]["pred_distribution"][lab] for m in models], float)
        vals = vals / np.array([sev["n_test"] for m in models])
        ax.bar(models, vals, 0.55, bottom=bottom, label=lab, color=CLASS_COLORS[lab])
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("预测占比")
    ax.set_title("训练 3 档损伤 → 测试 0.028″（未见程度）")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    for i, m in enumerate(models):
        ax.text(i, 1.03, f"IR/B acc={sev['models'][m]['two_class_accuracy']:.3f}",
                ha="center", fontsize=7.5)

    ax = axes[1]
    ms = list(pos["models"])
    x = np.arange(len(ms))
    accs = [pos["models"][m]["accuracy"] for m in ms]
    ors = [pos["models"][m]["or_recall"] for m in ms]
    ax.bar(x - 0.18, accs, 0.34, label="整体准确率", color="#4C78A8")
    ax.bar(x + 0.18, ors, 0.34, label="判为 OR 的比例", color="#F2A93B")
    for i, (a, o) in enumerate(zip(accs, ors)):
        ax.text(i - 0.18, a + 0.012, f"{a:.3f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, o + 0.012, f"{o:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, ms)
    ax.set_ylim(0, 1.0)
    ax.set_title("训练 @6 点 → 测试 @3/@12 点（未见位置）")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("两个分布外泛化维度：损伤程度可外推，故障位置不行", y=1.03)
    fig.tight_layout()
    save(fig, "fig15_severity_position.png")


def fig_importance() -> None:
    d = load("rf_importance.json")
    if not d:
        return
    top = 15
    names, vals = d["features"][:top][::-1], d["importance"][:top][::-1]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    colors = ["#F2A93B" if n.startswith("env_") else
              "#54A24B" if n.startswith("fd_") else "#4C78A8" for n in names]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("不纯度下降")
    ax.set_title("随机森林 Top-15 特征（橙=包络阶次，绿=频域，蓝=时域）")
    save(fig, "fig16_feature_importance.png")


def fig_10class() -> None:
    d = load("severity_10class.json")
    if not d:
        return
    loo = d["protocols"]["leave_one_load_out"]
    folds = ["0", "1", "2", "3"]
    models = list(loo["models"])
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(folds) + 1)
    w = 0.36
    for i, m in enumerate(models):
        vals = [loo["folds"][f][m]["mean"]["accuracy"] for f in folds] + \
               [loo["models"][m]["mean_accuracy"]]
        off = (i - 0.5) * w
        bars = ax.bar(x + off, vals, w, color=MODEL_COLORS.get(m, "#888888"),
                      label=f"{m}（均值 {vals[-1]:.3f}）", edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=6.5)
    ax.axvline(len(folds) - 0.5, color="#CCCCCC", lw=0.8, ls="--")
    ax.set_xticks(x, [f"{f} HP" for f in folds] + ["留一均值"])
    ax.set_ylim(0.6, 1.06)
    ax.set_ylabel("10 分类准确率")
    ax.set_title("类型 × 损伤程度（10 类）：跨载荷下比 4 分类明显更难")
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "fig18_10class.png")


def fig_hero() -> None:
    """结论图：左=四种泛化维度的难度阶梯，右=加噪退化曲线。"""
    tb = load("classical_time_block.json")
    cl = load("cross_load.json")
    sev = load("severity_extrapolation.json")
    pos = load("position_generalization.json")
    noise = load("noise_robustness.json")
    if not (tb and cl and sev and pos):
        return

    ladder = {
        "同工况（时间块）": tb["models"]["SVM"]["mean"]["accuracy"],
        "跨载荷（剔 217）": float(np.mean(
            [v["accuracy_excl_217"] for v in cl["sensitivity_excl_217"].values()])),
        "未见损伤程度": sev["models"]["SVM"]["two_class_accuracy"],
        "未见故障位置": pos["models"]["SVM"]["accuracy"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    keys = list(ladder)
    vals = [ladder[k] for k in keys]
    colors = ["#54A24B", "#72B7B2", "#F2A93B", "#E45756"]
    bars = ax.barh(range(len(keys))[::-1], vals, 0.6, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5)
    ax.set_yticks(range(len(keys))[::-1], keys)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("SVM（全特征）准确率")
    ax.set_title("同一模型的四种泛化难度")

    ax = axes[1]
    if noise:
        labels = list(noise["results"])
        style = {"SVM_all": ("#4C78A8", "SVM（全特征）"),
                 "SVM_env": ("#F2A93B", "SVM（仅包络阶次）"),
                 "CNN": ("#E45756", "1D-CNN（波形）")}
        for key, (c, lab) in style.items():
            ys = [noise["results"][k][key]["accuracy"] for k in labels]
            ax.plot(range(len(ys)), ys, "o-", lw=1.6, ms=4.5, color=c, label=lab)
        ax.set_xticks(range(len(labels)),
                      ["干净"] + [l.replace("dB", " dB") for l in labels[1:]])
        ax.set_xlabel("测试信噪比（训练始终干净）")
        ax.set_ylabel("准确率")
        ax.set_ylim(0.1, 1.05)
        ax.set_title("加噪退化：物理特征先崩，CNN 更稳")
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("CWRU 轴承诊断：难在哪、什么方法在什么条件下失灵", y=1.02)
    fig.tight_layout()
    save(fig, "hero.png")
    docs = ROOT / "docs" / "figures"
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG / "hero.png", docs / "hero.png")


def main() -> int:
    setup_style()
    for fn in (fig_leakage, fig_crossload, fig_confusion, fig_ablation,
               fig_cnn_curves, fig_severity_position, fig_importance,
               fig_10class, fig_hero):
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {fn.__name__}: {exc}")

    docs = ROOT / "docs" / "figures"
    docs.mkdir(parents=True, exist_ok=True)
    for name in ["fig10_leakage.png", "fig11_crossload.png", "fig12_confusion_crossload.png",
                 "fig13_ablation.png", "fig15_severity_position.png",
                 "fig16_feature_importance.png", "fig17_noise_robustness.png",
                 "fig18_10class.png"]:
        src = FIG / name
        if src.exists():
            shutil.copy(src, docs / name)
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
