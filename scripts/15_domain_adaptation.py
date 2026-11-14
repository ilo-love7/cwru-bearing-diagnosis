"""15 · v1.3 E32：特征级域适应能不能补上跨载荷迁移的不对称缺口？

v1.2 §3.12 测出 0→3 = 0.739 / 3→0 = 0.948，且载荷探针证明特征里残留强载荷
信息（全特征 0.895）。按常规思路，把源域边际分布对齐到目标域应当有用。本脚本
做四组有向对 × 含/剔 217 双口径 × 六种方法：

  baseline             源域训练，直接预测
  coral                CORAL 白化-重着色（无标签目标域，直推）
  coral_half           shrink=0.5 的部分对齐
  kmm                  核均值匹配给源样本加权（同样只用目标域无标签特征）
  fewshot5             目标域适配池每类 5 个有标签窗口拼入训练（上界对照）
  ceiling              目标域全部有标签训练（理论天花板，仅供参照）

公平口径：目标域按时间块划出前 10% 作为适配池，其余 90% 作为**所有方法共用**的
测试集；对齐方法拟合只用测试集的无标签特征，few-shot 只用适配池标签。217 双口径
在测试集内再剔一次。另报 baseline 在完整目标域上的准确率，便于对照 v1.2 数字。

诊断（用目标域**事后**标签，绝不参与变换/权重拟合）：边际 CORAL/MMD 距离、
目标域 Fisher 类间/类内比、源→目标类条件中心漂移、217 窗口预测分布。

产出
  results/metrics/domain_adaptation.json
  results/figures/fig25_domain_adaptation.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.adaptation import (class_conditional_drift, coral_distance,  # noqa: E402
                                  feature_scale, fisher_ratio, fit_coral,
                                  kernel_mean_weights, mmd_rbf)
from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import save_json                      # noqa: E402
from cwru_diag.models import make_svm                           # noqa: E402
from cwru_diag.pipeline import features_for, load_all           # noqa: E402

PAIRS = [(0, 3), (3, 0), (0, 1), (1, 3)]
METHODS = ("baseline", "coral", "coral_half", "kmm", "fewshot5")
SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
ANOMALY_FID = 217
ADAPT_FRAC = 0.10
K_SHOT = 5


def adapt_pool_mask(ws) -> np.ndarray:
    """每个文件按时间块取前 10% 的块作为适配池（与 carve_val 反向）。"""
    out = np.zeros(len(ws), bool)
    for fid in np.unique(ws.file_id):
        m = ws.file_id == fid
        blocks = np.unique(ws.block[m])
        n = max(1, int(round(len(blocks) * ADAPT_FRAC)))
        out[m & np.isin(ws.block, blocks[:n])] = True
    return out


def stratified_shot(idx: np.ndarray, y: np.ndarray, k: int,
                    rng: np.random.Generator) -> np.ndarray:
    """适配池里每类抽 k 个（不足则全取），返回索引。"""
    picked = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        picked.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.concatenate(picked) if picked else np.empty(0, int)


def acc(clf, x, y) -> float:
    return float((clf.predict(x) == y).mean())


def transformed(method: str, xs: np.ndarray, ys: np.ndarray,
                xt: np.ndarray, yt: np.ndarray, seed: int,
                x_shot: np.ndarray, y_shot: np.ndarray, salt: int = 0,
                kmm_w: np.ndarray | None = None):
    """返回 (源训练特征, 源标签, 目标测试特征, 拟合好的分类器)。

    对齐变换只依赖无标签特征，与 seed 无关；few-shot 抽样由 seed 决定。
    """
    if method == "baseline":
        clf = make_svm(C=10.0, seed=seed).fit(xs, ys)
        return xs, ys, xt, clf
    if method in ("coral", "coral_half"):
        # CORAL 只把**源域**变换到目标域分布；目标域保持原始特征，否则
        # 等于给目标又乘了一次 W，边际距离会被放大几个数量级。
        tr = fit_coral(xs, xt, shrink=0.0 if method == "coral" else 0.5)
        zs = tr(xs)
        return zs, ys, xt, make_svm(C=10.0, seed=seed).fit(zs, ys)
    if method == "kmm":
        w = kernel_mean_weights(xs, xt) if kmm_w is None else kmm_w
        return xs, ys, xt, make_svm(C=10.0, seed=seed).fit(xs, ys, clf__sample_weight=w)
    if method == "fewshot5":
        rng = np.random.default_rng(seed * 1009 + salt)
        sel = stratified_shot(np.arange(len(y_shot)), y_shot, K_SHOT, rng)
        zs = np.vstack([xs, x_shot[sel]])
        zys = np.concatenate([ys, y_shot[sel]])
        return zs, zys, xt, make_svm(C=10.0, seed=seed).fit(zs, zys)
    raise ValueError(method)


def diagnostics(zs: np.ndarray, ys: np.ndarray,
                zt: np.ndarray, yt: np.ndarray,
                scale: np.ndarray) -> dict:
    """结构诊断（yt 仅事后使用）。

    CORAL 距离在各维标准化后计算（原始量纲跨数量级会让 Frobenius 差被单个
    大尺度特征主导，不可解释）；MMD 用至多 400 行子样本控制 O(n²) 开销。
    """
    rng = np.random.default_rng(0)
    a = zs[rng.choice(len(zs), size=min(400, len(zs)), replace=False)]
    b = zt[rng.choice(len(zt), size=min(400, len(zt)), replace=False)]
    return {"coral_dist": coral_distance(zs / scale, zt / scale),
            "mmd": mmd_rbf(a / scale, b / scale),
            "fisher_source": fisher_ratio(zs, ys),
            "fisher_target": fisher_ratio(zt, yt),
            "class_drift": class_conditional_drift(zs, ys, zt, yt)["mean"]}


def main() -> int:
    t_start = time.time()
    cfg = load_config()
    seeds = [int(s) for s in cfg["seeds"]]
    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    y = np.asarray(ws.label)
    X, names = features_for(ws, ("time", "freq", "env"), cfg)
    SCALE = feature_scale(X, X)          # 全局统一尺度，所有诊断都用同一把尺
    is_217 = ws.file_id == ANOMALY_FID
    adapt_mask = adapt_pool_mask(ws)
    print(f"{len(ws)} 窗口 / {X.shape[1]} 维特征 / 217 占 {is_217.sum()} / "
          f"适配池 {adapt_mask.sum()}")

    out: dict = {
        "protocol": {
            "pairs": [list(p) for p in PAIRS], "methods": list(METHODS),
            "adapt_pool_frac": ADAPT_FRAC, "fewshot_k": K_SHOT,
            "seeds": seeds, "feature_group": "all",
            "note": "目标域按时间块划 10% 适配池，其余 90% 为所有方法共用测试集；"
                    "对齐拟合只用测试集无标签特征；诊断标签仅用于事后计算。",
        },
        "pairs": {}, "D_shrink_sweep": {},
    }

    for src, dst in PAIRS:
        src_mask = ws.load_hp == src
        tgt_mask = ws.load_hp == dst
        # 适配池 / 测试集（目标域内部）
        pool_mask = tgt_mask & adapt_mask
        te_all = tgt_mask & ~adapt_mask
        idx_pool = np.where(pool_mask)[0]

        # full-target 参考：与 v1.2 迁移矩阵同口径
        ref_clf = make_svm(C=10.0, seed=0).fit(X[src_mask], y[src_mask])
        full_ref = float((ref_clf.predict(X[tgt_mask]) == y[tgt_mask]).mean())

        pair_out: dict = {"n_train": int(src_mask.sum()),
                          "n_adapt_pool": int(pool_mask.sum()),
                          "n_test_all": int(te_all.sum()),
                          "baseline_full_target_reference": full_ref,
                          "protocols": {}}
        for proto in ("with_217", "without_217"):
            te = te_all & (~is_217 if proto == "without_217" else np.ones(len(ws), bool))
            idx_te = np.where(te)[0]
            xs, ys = X[src_mask], y[src_mask]
            xt, yt = X[idx_te], y[idx_te]

            # 适配池索引要映射到 xt 的行号：shot 样本必须落在测试集之外，
            # 因此 few-shot 用「适配池特征」单独拼进训练，而不是从 xt 里取。
            x_shot, y_shot = X[idx_pool], y[idx_pool]
            # KMM 权重只依赖无标签特征，与 seed 无关：整轮只解一次
            w_kmm = kernel_mean_weights(xs, xt)
            row: dict = {"n_test": int(len(yt)), "n_217_in_test": int(is_217[te].sum()),
                         "kmm_weight_stats": {"mean": float(w_kmm.mean()),
                                              "std": float(w_kmm.std()),
                                              "min": float(w_kmm.min()),
                                              "max": float(w_kmm.max()),
                                              "frac_near_one": float(np.mean(np.abs(w_kmm - 1) < 0.05))},
                         "methods": {}}
            salt = src * 17 + dst
            for method in METHODS:
                vals = []
                zs0 = zys0 = zt0 = clf0 = None
                for s in seeds:
                    zs, zys, zt, clf = transformed(method, xs, ys, xt, yt, s,
                                                   x_shot, y_shot, salt, w_kmm)
                    vals.append(acc(clf, zt, yt))
                    if s == seeds[0]:
                        zs0, zys0, zt0, clf0 = zs, zys, zt, clf
                diag = diagnostics(zs0, zys0, zt0, yt, SCALE)
                dist_217 = None
                if is_217[te].any():
                    p217 = clf0.predict(zt0[is_217[idx_te]])
                    uniq, cnt = np.unique(p217, return_counts=True)
                    dist_217 = {LABELS[int(u)]: float(v / len(p217))
                                for u, v in zip(uniq, cnt)}
                row["methods"][method] = {
                    "accuracy": float(np.mean(vals)), "std": float(np.std(vals)),
                    "per_seed": [float(v) for v in vals],
                    "diag": diag, "pred_217_distribution": dist_217,
                }
                print(f"  {src}→{dst} {proto:12s} {method:10s} "
                      f"acc={np.mean(vals):.4f}±{np.std(vals):.4f}")
            pair_out["protocols"][proto] = row
        out["pairs"][f"{src}_to_{dst}"] = pair_out

    # ---------- D 对齐强度扫描（最难的 0→3，含 217）----------
    print("\n=== D 对齐强度扫描（0→3, with_217）===")
    src, dst = 0, 3
    src_mask = ws.load_hp == src
    te = (ws.load_hp == dst) & ~adapt_mask
    xs, ys = X[src_mask], y[src_mask]
    xt, yt = X[te], y[te]
    sweep = {}
    for sh in SHRINKS:
        tr = fit_coral(xs, xt, shrink=sh)
        zs = tr(xs)                       # 目标 xt 不变换
        vals = [acc(make_svm(C=10.0, seed=s).fit(zs, ys), xt, yt) for s in seeds]
        cd = coral_distance(zs / SCALE, xt / SCALE)
        sweep[str(sh)] = {
            "accuracy": float(np.mean(vals)), "std": float(np.std(vals)),
            "coral_dist_after": cd,
            "fisher_source": fisher_ratio(zs, ys),
            "class_drift": class_conditional_drift(zs, ys, xt, yt)["mean"],
        }
        print(f"  shrink={sh:.2f} acc={np.mean(vals):.4f} "
              f"coral_dist={cd:.6f} "
              f"fisher_src={sweep[str(sh)]['fisher_source']:.4f}")
    out["D_shrink_sweep"] = {"pair": [0, 3], "protocol": "with_217",
                             "shrinks": list(SHRINKS), "rows": sweep,
                             "coral_dist_before": coral_distance(xs / SCALE, xt / SCALE),
                             "fisher_source_before": fisher_ratio(xs, ys),
                             "class_drift_before": class_conditional_drift(xs, ys, xt, yt)["mean"]}

    save_json(out, ROOT / "results" / "metrics" / "domain_adaptation.json")
    print(f"\n完成 → results/metrics/domain_adaptation.json ({time.time()-t_start:.0f}s)")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt
    from cwru_diag.plots import save, setup_style
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    ax = axes[0]
    x = np.arange(len(PAIRS)); w = 0.16
    colors = {"baseline": "#9AA3AD", "coral": "#E45756", "coral_half": "#F2A93B",
              "kmm": "#54A24B", "fewshot5": "#4C78A8"}
    for i, m in enumerate(METHODS):
        vals = [out["pairs"][f"{s}_to_{d}"]["protocols"]["with_217"]["methods"][m]["accuracy"]
                for s, d in PAIRS]
        ax.bar(x + (i - 2) * w, vals, w, label=m, color=colors[m])
    ax.set_xticks(x, [f"{s}→{d}HP" for s, d in PAIRS])
    ax.set_ylim(0.4, 1.03); ax.set_ylabel("准确率（含 217）")
    ax.set_title("A 域适应方法对比"); ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for proto, col, hatch in (("with_217", "#E45756", ""), ("without_217", "#4C78A8", "//")):
        for j, m in enumerate(("baseline", "coral", "fewshot5")):
            vals = [out["pairs"][f"{s}_to_{d}"]["protocols"][proto]["methods"][m]["accuracy"]
                    for s, d in PAIRS]
            pos = x + (j - 1) * 0.22 + (-0.11 if proto == "with_217" else 0.11)
            ax.bar(pos, vals, 0.2, color=col, alpha=0.55 if proto == "without_217" else 1.0,
                   hatch=hatch, label=f"{proto}/{m}" if j == 0 else None)
    ax.set_xticks(x, [f"{s}→{d}HP" for s, d in PAIRS])
    ax.set_ylim(0.4, 1.03); ax.set_title("B 含 / 剔 217 双口径")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[2]
    sh = list(SHRINKS)
    a2 = [sweep[str(s)]["accuracy"] for s in sh]
    ax.plot(sh, a2, "o-", color="#4C78A8", label="准确率")
    ax.set_xlabel("CORAL 收缩 s（1=不对齐）"); ax.set_ylabel("准确率", color="#4C78A8")
    ax.set_ylim(0.55, 1.02)
    ax2 = ax.twinx()
    ax2.plot(sh, [sweep[str(s)]["coral_dist_after"] for s in sh], "s--",
             color="#E45756", label="CORAL 距离")
    ax2.set_ylabel("边际距离", color="#E45756"); ax2.grid(False)
    ax.set_title("D 对齐强度：边际越近≠越准")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, fontsize=7, loc="lower left")
    fig.tight_layout()
    p = save(fig, "fig25_domain_adaptation.png")
    print(f"图 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
