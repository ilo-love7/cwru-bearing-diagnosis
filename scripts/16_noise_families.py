"""16 · v1.3 E33：噪声类型与增广公平性——白噪增广能不能跨噪声族泛化？

报告 §5.3 明确把"只做了高斯白噪声"列为局限，并指出 §3.9 里 SVM 用固定噪声
副本、CNN 逐窗重抽，公平性不对等。本脚本补齐：

  噪声族   white（与 08 同口径）/ colored（转频谐波 + 结构共振）/ impulse（稀疏脉冲）
  SNR 网格 clean, 10, 4, 0, −4, −8 dB（逐窗精确归一，三族功率严格一致）
  训练配置 clean        干净训练（零适配，与 08 对照）
            white_fixed  v1.1 口径：3 个固定白噪副本 [10, 2, −4] dB
            white_fair   逐窗随机白噪 × 3 个重抽副本（与 CNN 每 epoch 重抽对等）
            mix_fair     逐窗随机三族噪声 × 3 个重抽副本
            cnn_clean / cnn_white / cnn_mix   1D-CNN 三配置（aug_fn 注入多噪声族）

核心问题：v1.1 的"噪声增广救回加噪崩溃"是否只在**同一噪声族内**成立。
零适配参考线直接读 08 的 noise_robustness.json，不重跑。

产出
  results/metrics/noise_families.json
  results/figures/fig26_noise_families.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import save_json                      # noqa: E402
from cwru_diag.features import extract                          # noqa: E402
from cwru_diag.models import make_svm, predict_cnn, train_cnn  # noqa: E402
from cwru_diag.noise import (NOISE_KINDS, add_noise_at_snr,     # noqa: E402
                             mixed_augment, noise_shape_stats)
from cwru_diag.pipeline import (_geom, carve_val, features_for,  # noqa: E402
                                load_all, time_block_split)
from cwru_diag.plots import save, setup_style                   # noqa: E402

SNRS: tuple[float | None, ...] = (None, 10.0, 4.0, 0.0, -4.0, -8.0)
AUG_COPIES = (10.0, 2.0, -4.0)         # v1.1 固定副本口径
AUG_SNRS = (None, 10.0, 4.0, 0.0, -4.0)
N_REPEATS = 3
CNN_EPOCHS = 30
TEST_SEED = 42


def key(snr: float | None) -> str:
    return "clean" if snr is None else f"{snr:g}dB"


def extract_pair(raw, rpm, cfg, geom):
    fa, _ = extract(raw, cfg["fs"], rpm, geom=geom, groups=("time", "freq", "env"))
    fe, _ = extract(raw, cfg["fs"], rpm, geom=geom, groups=("env",))
    return fa, fe


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    fs = float(cfg["fs"])
    ws = load_all(cfg, ("main",))
    y = np.asarray(ws.label)
    tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    tr2, va = carve_val(ws, tr, val_frac=0.1)
    geom = _geom(cfg)
    Xall, _ = features_for(ws, ("time", "freq", "env"), cfg)
    Xenv, _ = features_for(ws, ("env",), cfg)
    rpm_te = ws.rpm[te]
    rpm_tr = float(np.mean(ws.rpm[tr2]))
    print(f"训练 {tr2.sum()} / 验证 {va.sum()} / 测试 {te.sum()} 窗口")

    # ---------- 预计算所有测试组合（特征必须从加噪波形重算）----------
    print("\n=== 预计算测试特征（3 族 × 6 SNR）===")
    test_raw: dict = {}
    test_feat: dict = {}
    for kind in NOISE_KINDS:
        test_raw[kind], test_feat[kind] = {}, {}
        for snr in SNRS:
            raw = (ws.X[te] if snr is None else
                   add_noise_at_snr(ws.X[te], kind, snr, seed=TEST_SEED,
                                    fs=fs, rpm=rpm_te))
            if snr is None:
                fa, fe = Xall[te], Xenv[te]
            else:
                fa, fe = extract_pair(raw, rpm_te, cfg, geom)
            test_raw[kind][key(snr)] = raw
            test_feat[kind][key(snr)] = (fa, fe)
        print(f"  {kind:8s} 完成 ({time.time()-t0:.0f}s)")

    # 噪声形状诊断（0 dB）
    shape_stats = {}
    for kind in NOISE_KINDS:
        shape_stats[kind] = noise_shape_stats(ws.X[te], test_raw[kind]["0dB"], fs=fs)

    # ---------- 特征模型：四种训练配置 ----------
    print("\n=== 特征模型训练 ===")
    Xtr_parts: dict[str, tuple[list, list]] = {
        "clean": ([Xall[tr2]], [Xenv[tr2]]),
        "white_fixed": ([Xall[tr2]], [Xenv[tr2]]),
        "white_fair": ([Xall[tr2]], [Xenv[tr2]]),
        "mix_fair": ([Xall[tr2]], [Xenv[tr2]]),
    }
    # v1.1 口径：3 个固定白噪副本
    for snr in AUG_COPIES:
        raw = add_noise_at_snr(ws.X[tr2], "white", snr, seed=100 + int(snr),
                               fs=fs, rpm=rpm_tr)
        fa, fe = extract_pair(raw, ws.rpm[tr2], cfg, geom)
        Xtr_parts["white_fixed"][0].append(fa)
        Xtr_parts["white_fixed"][1].append(fe)
    for cfg_name, kinds in (("white_fair", ("white",)), ("mix_fair", NOISE_KINDS)):
        for rep in range(N_REPEATS):
            raw = mixed_augment(ws.X[tr2], kinds, AUG_SNRS,
                                seed=7000 + rep * 31, fs=fs, rpm=rpm_tr)
            fa, fe = extract_pair(raw, ws.rpm[tr2], cfg, geom)
            Xtr_parts[cfg_name][0].append(fa)
            Xtr_parts[cfg_name][1].append(fe)

    clfs: dict[str, dict[str, object]] = {}
    for cfg_name, (pa, pe) in Xtr_parts.items():
        xa, xe = np.vstack(pa), np.vstack(pe)
        ytr = np.tile(y[tr2], len(pa))
        c = float(cfg["classical"]["svm"]["C"])
        clfs[cfg_name] = {
            "SVM_all": make_svm(C=c, seed=0).fit(xa, ytr),
            "SVM_env": make_svm(C=c, seed=0).fit(xe, ytr),
            "n_train": int(len(ytr)),
        }
        print(f"  {cfg_name:12s} n={len(ytr):5d} ({time.time()-t0:.0f}s)")

    # ---------- CNN 三配置 ----------
    print("\n=== CNN 训练（clean / white / mix）===")
    aug_white = AUG_SNRS
    cnn_specs = {
        "cnn_clean": {"aug_snrs": None, "aug_fn": None},
        "cnn_white": {"aug_snrs": list(aug_white), "aug_fn": None},
        "cnn_mix": {"aug_snrs": None,
                    "aug_fn": lambda x, ep: mixed_augment(
                        x, NOISE_KINDS, AUG_SNRS, seed=9000 + ep, fs=fs, rpm=rpm_tr)},
    }
    cnn_models = {}
    for name, spec in cnn_specs.items():
        res = train_cnn(ws.X[tr2], y[tr2], ws.X[va], y[va], n_classes=len(LABELS),
                        epochs=CNN_EPOCHS, batch_size=int(cfg["cnn"]["batch_size"]),
                        lr=float(cfg["cnn"]["lr"]),
                        weight_decay=float(cfg["cnn"]["weight_decay"]),
                        dropout=float(cfg["cnn"]["dropout"]), seed=0, verbose=False,
                        aug_snrs=spec["aug_snrs"], aug_fn=spec["aug_fn"])
        cnn_models[name] = res.model
        print(f"  {name:10s} best_epoch={res.best_epoch} "
              f"val={res.best_val_acc:.4f} ({time.time()-t0:.0f}s)")

    # ---------- 全矩阵评估 ----------
    print("\n=== 评估矩阵 ===")
    results: dict = {}
    for cfg_name in clfs:
        results[cfg_name] = {"SVM_all": {}, "SVM_env": {},
                             "n_train": clfs[cfg_name]["n_train"]}
        for kind in NOISE_KINDS:
            for m in ("SVM_all", "SVM_env"):
                results[cfg_name][m][kind] = {}
                for snr in SNRS:
                    fa, fe = test_feat[kind][key(snr)]
                    f = fa if m == "SVM_all" else fe
                    results[cfg_name][m][kind][key(snr)] = float(
                        (clfs[cfg_name][m].predict(f) == y[te]).mean())
    for name, model in cnn_models.items():
        results[name] = {}
        for kind in NOISE_KINDS:
            results[name][kind] = {}
            for snr in SNRS:
                pred = predict_cnn(model, test_raw[kind][key(snr)])
                results[name][kind][key(snr)] = float((pred == y[te]).mean())

    for cfg_name in results:
        is_svm = "SVM_all" in results[cfg_name]
        vals = [results[cfg_name]["SVM_all"][k]["0dB"] if is_svm
                else results[cfg_name][k]["0dB"] for k in NOISE_KINDS]
        print(f"  {cfg_name:12s} @0dB  " +
              "  ".join(f"{k}={v:.3f}" for k, v in zip(NOISE_KINDS, vals)))

    # ---------- 派生指标 ----------
    derive = {}
    for cfg_name in ("white_fixed", "white_fair", "mix_fair",
                     "cnn_white", "cnn_mix"):
        base = "clean" if not cfg_name.startswith("cnn") else "cnn_clean"
        m = "SVM_all" if not cfg_name.startswith("cnn") else None

        def get(c, k, s):
            return results[c][m][k][s] if m else results[c][k][s]

        derive[cfg_name] = {
            kind: {s: get(cfg_name, kind, s) - get(base, kind, s)
                   for s in [key(x) for x in SNRS]}
            for kind in NOISE_KINDS
        }
    derive["fairness_white_fair_minus_fixed"] = {
        s: results["white_fair"]["SVM_all"]["white"][s]
           - results["white_fixed"]["SVM_all"]["white"][s]
        for s in [key(x) for x in SNRS]
    }

    zero_path = ROOT / "results" / "metrics" / "noise_robustness.json"
    zero_ref = None
    if zero_path.exists():
        zero_ref = json.loads(zero_path.read_text(encoding="utf-8"))["results"]

    out = {
        "protocol": {"snr_levels": [key(s) for s in SNRS], "kinds": list(NOISE_KINDS),
                     "test_seed": TEST_SEED, "cnn_epochs": CNN_EPOCHS,
                     "n_repeats": N_REPEATS, "aug_fixed_copies": list(AUG_COPIES),
                     "aug_fair_snrs": [str(s) for s in AUG_SNRS],
                     "note": "三族噪声按逐窗自身功率精确归一到同一 SNR；"
                             "特征均从加噪波形重算；white_fixed 为 v1.1 口径，"
                             "*_fair 每副本逐窗重抽，与 CNN 每 epoch 重抽对等。"},
        "n_train_clean": int(tr2.sum()), "n_test": int(te.sum()),
        "noise_shape_at_0db": shape_stats, "results": results,
        "delta_vs_clean_train": derive, "zero_adaptation_reference_v08": zero_ref,
    }
    save_json(out, ROOT / "results" / "metrics" / "noise_families.json")
    print(f"\n完成 → results/metrics/noise_families.json ({time.time()-t0:.0f}s)")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt
    setup_style()
    labels = [key(s) for s in SNRS]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    ax = axes[0]
    styles = {"white": "#4C78A8", "colored": "#E45756", "impulse": "#54A24B"}
    for kind in NOISE_KINDS:
        vals = [results["clean"]["SVM_all"][kind][s] for s in labels]
        ax.plot(range(len(labels)), vals, "o-", color=styles[kind],
                label=f"{kind}（零适配）")
    if zero_ref:
        # 08 的 SNR 网格是 [clean,10,6,2,0,-4]，只画两者共有的档次
        common = [s for s in labels if s in zero_ref]
        ax.plot([labels.index(s) for s in common],
                [zero_ref[s]["SVM_all"]["accuracy"] for s in common],
                "s--", color="#9AA3AD", ms=3.5, label="08 白噪参考")
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0.15, 1.02)
    ax.set_ylabel("SVM 全特征准确率"); ax.set_xlabel("测试 SNR")
    ax.set_title("A 三族噪声的零适配崩溃曲线"); ax.legend(fontsize=7)

    ax = axes[1]
    cfgs = ("clean", "white_fixed", "white_fair", "mix_fair")
    x = np.arange(len(NOISE_KINDS)); w = 0.2
    for i, c in enumerate(cfgs):
        vals = [results[c]["SVM_all"][k]["0dB"] for k in NOISE_KINDS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=c)
    ax.set_xticks(x, list(NOISE_KINDS)); ax.set_ylim(0.2, 1.03)
    ax.set_ylabel("0 dB 准确率"); ax.set_title("B 跨噪声族增广收益")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    for name, col in (("cnn_clean", "#9AA3AD"), ("cnn_white", "#4C78A8"),
                      ("cnn_mix", "#E45756")):
        vals = [np.mean([results[name][k][s] for k in NOISE_KINDS]) for s in labels]
        ax.plot(range(len(labels)), vals, "o-", color=col, label=name)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0.2, 1.03)
    ax.set_xlabel("测试 SNR"); ax.set_ylabel("三族平均准确率")
    ax.set_title("C CNN 增广配置"); ax.legend(fontsize=7)
    fig.tight_layout()
    p = save(fig, "fig26_noise_families.png")
    print(f"图 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
