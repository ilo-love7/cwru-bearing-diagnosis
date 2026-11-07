"""08 · 加噪鲁棒性：干净训练、带噪测试（不重新训练）。

现场振动信号的信噪比远低于实验室台架。这里在测试窗口上叠加高斯白噪声，
把 SNR 从干净逐级降到 −4 dB，看三类方案谁先崩：
  SVM(全特征) / SVM(仅包络阶次) / 1D-CNN(原始波形)

CNN 直接复用 03_train_cnn.py 存下的 cnn_time_block.pt（同一切分，seed=0）。

产出
  results/metrics/noise_robustness.json
  results/figures/fig17_noise_robustness.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                       # noqa: E402
from cwru_diag.dataset import LABELS                           # noqa: E402
from cwru_diag.evaluation import evaluate, save_json, summarize  # noqa: E402
from cwru_diag.features import extract                         # noqa: E402
from cwru_diag.models import BearingCNN, make_svm, predict_cnn  # noqa: E402
from cwru_diag.pipeline import (_geom, features_for, load_all,  # noqa: E402
                                time_block_split)
from cwru_diag.plots import save, setup_style                  # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

SNRS = [None, 10, 6, 2, 0, -4]


def add_noise(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """按每个窗口自身的功率加高斯白噪声。"""
    rng = np.random.default_rng(seed)
    p_sig = (X ** 2).mean(axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return X + rng.normal(size=X.shape) * np.sqrt(p_noise)


def main() -> int:
    cfg = load_config()
    ws = load_all(cfg, ("main",))
    y = np.asarray(ws.label)
    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    Xall, names_all = features_for(ws, ("time", "freq", "env"), cfg)
    Xenv, _ = features_for(ws, ("env",), cfg)
    print(f"训练 {tr.sum()} / 测试 {te.sum()} 窗口，特征 {Xall.shape[1]} 维")

    svm_all = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0).fit(Xall[tr], y[tr])
    svm_env = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0).fit(Xenv[tr], y[tr])

    ckpt = ROOT / "results" / "models" / "cnn_time_block.pt"
    if not ckpt.exists():
        print("缺少 cnn_time_block.pt，先跑 scripts/03_train_cnn.py")
        return 2
    cnn = BearingCNN(n_classes=len(LABELS), dropout=float(cfg["cnn"]["dropout"]))
    cnn.load_state_dict(torch.load(ckpt, map_location="cpu"))

    results = {}
    for snr in SNRS:
        Xte_raw = ws.X[te] if snr is None else add_noise(ws.X[te], snr, seed=42)
        # 特征必须从加噪后的波形重算，否则测的不是同一件事
        if snr is None:
            f_all, f_env = Xall[te], Xenv[te]
        else:
            f_all, _ = extract(Xte_raw, cfg["fs"], ws.rpm[te], geom=_geom(cfg),
                               groups=("time", "freq", "env"))
            f_env, _ = extract(Xte_raw, cfg["fs"], ws.rpm[te], geom=_geom(cfg), groups=("env",))

        row = {}
        for name, clf, feats in (("SVM_all", svm_all, f_all), ("SVM_env", svm_env, f_env)):
            m = evaluate(y[te], clf.predict(feats), LABELS)
            row[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        m = evaluate(y[te], predict_cnn(cnn, Xte_raw), LABELS)
        row["CNN"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

        key = "clean" if snr is None else f"{snr}dB"
        results[key] = row
        print(f"  SNR={key:>6s}  " + "  ".join(
            f"{k}={v['accuracy']:.4f}" for k, v in row.items()))

    save_json({"n_test": int(te.sum()), "snr_levels": [str(s) for s in SNRS],
               "results": results},
              ROOT / "results" / "metrics" / "noise_robustness.json")

    setup_style()
    labels = ["clean"] + [f"{s} dB" for s in SNRS[1:]]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    style = {"SVM_all": ("#4C78A8", "SVM（全特征）"), "SVM_env": ("#F2A93B", "SVM（仅包络阶次）"),
             "CNN": ("#E45756", "1D-CNN（原始波形）")}
    for key in ("SVM_all", "SVM_env", "CNN"):
        vals = [results[k][key]["accuracy"] for k in results]
        ax.plot(range(len(vals)), vals, "o-", lw=1.7, ms=5, color=style[key][0],
                label=style[key][1])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("测试信噪比（训练始终用干净信号）")
    ax.set_ylabel("准确率")
    ax.set_ylim(0.2, 1.02)
    ax.set_title("加噪鲁棒性：物理特征 vs 端到端深度特征")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig17_noise_robustness.png")
    print("完成 → results/metrics/noise_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
