"""10 · 噪声增广训练：训练时见过噪声，能不能救回加噪崩溃的特征模型。

08 只测了"零适配"（干净训练、带噪测试）：10 dB 下 SVM 掉到 0.545、CNN 还有 0.985。
报告 §5.3 指出这只说明"不适配会怎样"，不能据此否定特征方法。这里补上工程上
正确的做法——训练时加噪：

  CNN      aug_snrs=[None,10,6,2,0,-4]，每个 epoch 逐窗重抽 SNR（models.augment_noise）
  SVM_all  干净 + 10/2/-4 dB 三个固定噪声副本的特征拼接后训练
  SVM_env  同上（仅包络阶次特征）

测试口径与 08 完全一致（同一切分、同一 SNR 网格、同一 add_noise 实现），
零适配曲线直接读 results/metrics/noise_robustness.json，不重跑。

产出
  results/metrics/noise_augmented.json
  results/figures/fig19_noise_aug.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import evaluate, save_json, summarize  # noqa: E402
from cwru_diag.features import extract                          # noqa: E402
from cwru_diag.models import (augment_noise, make_svm,          # noqa: E402
                              predict_cnn, train_cnn)
from cwru_diag.pipeline import (_geom, features_for, load_all,  # noqa: E402
                                time_block_split, carve_val)
from cwru_diag.plots import save, setup_style                   # noqa: E402

SNRS = [None, 10, 6, 2, 0, -4]
AUG_SNRS = [None, 10, 6, 2, 0, -4]      # CNN 训练时逐窗重抽
AUG_COPIES = [10, 2, -4]                # SVM 的固定噪声副本
SEEDS = [0, 1]


def main() -> int:
    cfg = load_config()
    ws = load_all(cfg, ("main",))
    y = np.asarray(ws.label)
    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    tr2, va = carve_val(ws, tr, val_frac=0.1)
    print(f"训练 {tr2.sum()} / 验证 {va.sum()} / 测试 {te.sum()} 窗口")

    zero_path = ROOT / "results" / "metrics" / "noise_robustness.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))["results"] if zero_path.exists() else None
    if zero is None:
        print("提示：缺 noise_robustness.json（08 的产物），图里只画增广后曲线。")

    out: dict = {"n_train": int(tr2.sum()), "n_test": int(te.sum()),
                 "aug": {"cnn": {"snr_choices": [str(s) for s in AUG_SNRS], "seeds": SEEDS},
                         "svm": {"copies_db": AUG_COPIES}},
                 "results": {}}

    # ---------- 增广 CNN ----------
    print("\n=== 增广 CNN（每个 epoch 逐窗重抽 SNR）===")
    cnn_preds: dict[int, np.ndarray] = {}
    cnn_runs = []
    for seed in SEEDS:
        t0 = time.time()
        res = train_cnn(
            ws.X[tr2], y[tr2], ws.X[va], y[va], n_classes=len(LABELS),
            epochs=int(cfg["cnn"]["epochs"]), batch_size=int(cfg["cnn"]["batch_size"]),
            lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
            dropout=float(cfg["cnn"]["dropout"]), seed=seed, verbose=False,
            aug_snrs=AUG_SNRS,
        )
        cnn_preds[seed] = predict_cnn(res.model, ws.X[te])
        cnn_runs.append({"seed": seed, "best_epoch": res.best_epoch,
                         "best_val_acc": float(res.best_val_acc),
                         "elapsed_s": round(time.time() - t0, 1)})
        print(f"  seed={seed}  best_epoch={res.best_epoch}  ({cnn_runs[-1]['elapsed_s']}s)")

    # ---------- 增广 SVM（特征拼接固定噪声副本）----------
    print("\n=== 增广 SVM（干净 + 10/2/-4 dB 副本的特征拼接）===")
    Xtr_parts_all, Xtr_parts_env = [], []
    for tag, snr in [("clean", None)] + [(f"{s}dB", s) for s in AUG_COPIES]:
        t0 = time.time()
        raw = ws.X[tr2] if snr is None else augment_noise(ws.X[tr2], [snr], seed=100 + snr if snr else 7)
        f_all, _ = extract(raw, cfg["fs"], ws.rpm[tr2], geom=_geom(cfg),
                           groups=("time", "freq", "env"))
        f_env, _ = extract(raw, cfg["fs"], ws.rpm[tr2], geom=_geom(cfg), groups=("env",))
        Xtr_parts_all.append(f_all)
        Xtr_parts_env.append(f_env)
        print(f"  副本 {tag:6s} 特征提取 {time.time() - t0:.0f}s")
    Xtr_all = np.vstack(Xtr_parts_all)
    Xtr_env = np.vstack(Xtr_parts_env)
    ytr_aug = np.tile(y[tr2], len(Xtr_parts_all))
    print(f"  增广后训练矩阵 all={Xtr_all.shape} env={Xtr_env.shape}")

    t0 = time.time()
    svm_all = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0).fit(Xtr_all, ytr_aug)
    svm_env = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0).fit(Xtr_env, ytr_aug)
    print(f"  两个 SVM 拟合 {time.time() - t0:.0f}s")

    # ---------- 各 SNR 测试 ----------
    print("\n=== 测试（SNR 网格与 08 相同）===")
    for snr in SNRS:
        Xte_raw = ws.X[te] if snr is None else augment_noise(ws.X[te], [snr], seed=42)
        if snr is None:
            f_all, f_env = features_for(ws, ("time", "freq", "env"), cfg)[0][te], \
                           features_for(ws, ("env",), cfg)[0][te]
        else:
            f_all, _ = extract(Xte_raw, cfg["fs"], ws.rpm[te], geom=_geom(cfg),
                               groups=("time", "freq", "env"))
            f_env, _ = extract(Xte_raw, cfg["fs"], ws.rpm[te], geom=_geom(cfg), groups=("env",))

        row = {}
        for name, clf, feats in (("SVM_all", svm_all, f_all), ("SVM_env", svm_env, f_env)):
            m = evaluate(y[te], clf.predict(feats), LABELS)
            row[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        accs = [evaluate(y[te], cnn_preds[s], LABELS)["accuracy"] for s in SEEDS]
        f1s = [evaluate(y[te], cnn_preds[s], LABELS)["macro_f1"] for s in SEEDS]
        row["CNN"] = {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
                      "std_accuracy": float(np.std(accs))}

        key = "clean" if snr is None else f"{snr}dB"
        out["results"][key] = row
        print(f"  SNR={key:>6s}  " + "  ".join(
            f"{k}={v['accuracy']:.4f}" for k, v in row.items()))

    out["cnn_runs"] = cnn_runs
    save_json(out, ROOT / "results" / "metrics" / "noise_augmented.json")

    # ---------- 图：零适配 vs 增广 ----------
    setup_style()
    labels = ["clean"] + [f"{s} dB" for s in SNRS[1:]]
    keys = list(out["results"].keys())
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    style = {"SVM_all": "#4C78A8", "SVM_env": "#F2A93B", "CNN": "#E45756"}
    pretty = {"SVM_all": "SVM（全特征）", "SVM_env": "SVM（仅包络阶次）", "CNN": "1D-CNN"}
    for key in ("SVM_all", "SVM_env", "CNN"):
        aug_vals = [out["results"][k][key]["accuracy"] for k in keys]
        ax.plot(range(len(aug_vals)), aug_vals, "o-", lw=1.8, ms=5,
                color=style[key], label=f"{pretty[key]}（增广训练）")
        if zero:
            z_vals = [zero[k][key]["accuracy"] for k in keys]
            ax.plot(range(len(z_vals)), z_vals, "s--", lw=1.1, ms=3.5,
                    color=style[key], alpha=0.55, label=f"{pretty[key]}（零适配）")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("测试信噪比")
    ax.set_ylabel("准确率")
    ax.set_ylim(0.2, 1.02)
    ax.set_title("噪声增广训练 vs 零适配：加噪崩溃能不能救回来")
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    save(fig, "fig19_noise_aug.png")
    print("\n完成 → results/metrics/noise_augmented.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
