"""绘图：统一浅色风格，中文字体回退到微软雅黑。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
CLASS_COLORS = {"Normal": "#4C78A8", "IR": "#E45756", "OR": "#F2A93B", "B": "#54A24B"}


def setup_style() -> None:
    from matplotlib import font_manager

    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in installed), None)
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#9AA3AD",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E3E7EC",
        "grid.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
        "font.sans-serif": ([chosen] if chosen else []) + ["DejaVu Sans"],
    })


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def confusion_heatmap(cm: list[list[int]], labels: list[str], title: str,
                      path: str, normalize: bool = True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * (1 if normalize else cm.max()) else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return save(fig, path)


def grouped_bars(groups: list[str], series: dict[str, list[float]], ylabel: str,
                 title: str, path: str, ylim: tuple[float, float] | None = None,
                 hline: float | None = None):
    n_s = len(series)
    x = np.arange(len(groups))
    w = 0.8 / max(n_s, 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = ["#4C78A8", "#E45756", "#F2A93B", "#54A24B", "#8C6BB1"]
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n_s - 1) / 2) * w, vals, w, label=name,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#444444")
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="#999999", ls="--", lw=0.8)
    ax.legend(ncol=n_s, loc="lower right", fontsize=8)
    return save(fig, path)


def scatter_embedding(Z: np.ndarray, color_by: np.ndarray, title: str, path: str,
                      legend_title: str = "", cmap_labels: list[str] | None = None,
                      palette: dict | None = None, xlabel: str = "PC1", ylabel: str = "PC2"):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    uniq = list(dict.fromkeys(color_by.tolist()))
    palette = palette or {}
    default = ["#4C78A8", "#E45756", "#F2A93B", "#54A24B", "#8C6BB1", "#72B7B2"]
    for i, u in enumerate(uniq):
        m = color_by == u
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.55, linewidths=0,
                   color=palette.get(u, default[i % len(default)]), label=str(u))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=7.5, title=legend_title, loc="best")
    return save(fig, path)
