"""01 · 数据体检：清单统计、波形/包络谱、特征空间结构。

产出
  results/metrics/eda_summary.json
  results/records.csv
  results/figures/fig01_waveforms.png
  results/figures/fig02_envelope_spectra.png
  results/figures/fig03_feature_pca.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                      # noqa: E402
from cwru_diag.dataset import LABELS, record_table            # noqa: E402
from cwru_diag.evaluation import save_json                    # noqa: E402
from cwru_diag.kinematics import BearingGeometry, fault_frequencies  # noqa: E402
from cwru_diag.io import load_record                          # noqa: E402
from cwru_diag.pipeline import features_for, load_all         # noqa: E402
from cwru_diag.plots import CLASS_COLORS, save, setup_style, scatter_embedding  # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
from scipy.signal import hilbert                              # noqa: E402
from sklearn.decomposition import PCA                         # noqa: E402


def plot_waveforms(cfg) -> None:
    """同载荷（0 HP）下四类轴承的 0.2 s 原始波形。"""
    examples = {"Normal": "97.mat", "IR": "105.mat", "OR": "130.mat", "B": "118.mat"}
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
    fs = cfg["fs"]
    n = int(0.2 * fs)
    for ax, (name, fname) in zip(axes, examples.items()):
        rec = load_record(Path(cfg["raw_path"]) / fname, fs=fs)
        t = np.arange(n) / fs * 1e3
        ax.plot(t, rec.de[:n], lw=0.55, color=CLASS_COLORS[name])
        ax.set_ylabel("加速度 (g)")
        ax.set_title(f"{name}  ·  {fname}  ·  {rec.rpm:.0f} rpm", loc="left", fontsize=9)
        ax.set_xlim(0, t[-1])
    axes[-1].set_xlabel("时间 (ms)")
    fig.suptitle("CWRU 驱动端振动：四类状态的时域形态（0 HP）", y=0.995)
    fig.tight_layout()
    save(fig, "fig01_waveforms.png")


def plot_envelope(cfg) -> None:
    """包络谱：理论故障特征频率处是否出现峰。"""
    examples = [("Normal", "97.mat"), ("IR", "105.mat"), ("OR", "130.mat"), ("B", "118.mat")]
    geom = BearingGeometry()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    for ax, (name, fname) in zip(axes.ravel(), examples):
        rec = load_record(Path(cfg["raw_path"]) / fname, fs=cfg["fs"])
        x = rec.de - rec.de.mean()
        env = np.abs(hilbert(x))
        env = env - env.mean()
        n = len(env)
        E = 2.0 * np.abs(np.fft.rfft(env)) / n
        f = np.fft.rfftfreq(n, 1 / cfg["fs"])
        m = f <= 500
        ax.plot(f[m], E[m], lw=0.7, color=CLASS_COLORS[name])
        ff = fault_frequencies(rec.fr, geom)
        for key, val in ff.items():
            if key == "BSF":
                continue
            if val <= 500:
                ax.axvline(val, color="#888888", ls=":", lw=0.8)
                ax.text(val, ax.get_ylim()[1] * 0.92, key, rotation=90,
                        fontsize=6.5, color="#666666", ha="right", va="top")
        ax.set_title(f"{name}（{rec.rpm:.0f} rpm）", loc="left")
        ax.set_ylabel("包络幅值")
        ax.set_xlabel("频率 (Hz)")
    fig.suptitle("包络谱与理论故障特征频率（点线）", y=1.0)
    fig.tight_layout()
    save(fig, "fig02_envelope_spectra.png")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="main,position,severity")
    args = ap.parse_args()

    setup_style()
    cfg = load_config()
    roles = tuple(r.strip() for r in args.roles.split(","))
    print(f"装配数据（roles={roles}）...")
    plot_waveforms(cfg)
    plot_envelope(cfg)
    ws = load_all(cfg, roles)
    tab = record_table(ws)
    tab.to_csv(ROOT / "results" / "records.csv", index=False, encoding="utf-8-sig")

    rec = (tab.groupby(["file_id", "label", "load_hp", "severity", "position", "role"])
              .agg(n_windows=("start", "size"), rpm=("rpm", "first"))
              .reset_index())
    rec.to_csv(ROOT / "results" / "records_by_file.csv", index=False, encoding="utf-8-sig")

    print(f"窗口总数 {len(ws)}，记录数 {rec.shape[0]}")
    print(tab.groupby("label").size().to_string())

    print("抽特征（PCA 用）...")
    X, names = features_for(ws, ("time", "freq", "env"), cfg)

    Z = PCA(n_components=2, random_state=0).fit_transform(
        (X - X.mean(0)) / (X.std(0) + 1e-9))
    labels = np.array([LABELS[i] for i in ws.label])
    scatter_embedding(Z, labels, "特征空间 PCA：按故障类型着色", "fig03_feature_pca.png",
                      legend_title="类别", palette=CLASS_COLORS)
    scatter_embedding(Z, ws.load_hp.astype(str), "特征空间 PCA：按载荷着色",
                      "fig04_feature_pca_by_load.png", legend_title="载荷 (HP)",
                      palette={"0": "#4C78A8", "1": "#72B7B2", "2": "#F2A93B", "3": "#B279A2"})

    summary = {
        "n_windows": int(len(ws)),
        "n_records": int(rec.shape[0]),
        "n_features": len(names),
        "feature_names": names,
        "window": cfg["window"],
        "overlap": cfg["overlap"],
        "fs": cfg["fs"],
        "classes": LABELS,
        "windows_per_class": tab.groupby("label").size().to_dict(),
        "windows_per_load": tab.groupby("load_hp").size().to_dict(),
        "windows_per_role": tab.groupby("role").size().to_dict(),
        "records": rec.to_dict(orient="records"),
        "pca_explained_var": PCA(n_components=2, random_state=0)
                              .fit((X - X.mean(0)) / (X.std(0) + 1e-9))
                              .explained_variance_ratio_.tolist(),
    }
    save_json(summary, ROOT / "results" / "metrics" / "eda_summary.json")
    print("完成 → results/metrics/eda_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
