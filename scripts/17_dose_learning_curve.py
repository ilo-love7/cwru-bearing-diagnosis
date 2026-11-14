"""17 · v1.3 E34：217.mat 剂量曲线 + 样本效率学习曲线 + 数据多样性去混杂。

v1.2 §3.11-B3 只扫了「117 窗 vs 478 窗」一个剂量点，且 116 窗是**固定前缀**；
报告 §5.6 自己把"完整剂量曲线未做""upsample 只排除了数据量、没排除数据多样性"
列为局限。本脚本补齐：

  A 完整剂量曲线   3HP→0/1/2，k ∈ {0,10,25,50,75,116,200,300,478}，每剂量
                   5 个独立随机子集；三条对照：
                     217_random  异常件的随机 k 窗
                     upsample    从其余 3HP 文件有放回抽 k 窗（只补数据量）
                     borrow_real 从**非目标载荷**的同尺寸真实 IR021 文件（209/210/211）
                                 无放回（不足时有放回）借 k 窗（补数据多样性）
                     prefix      v1.2 口径：前 k 窗（单条参考）
  B 学习曲线        time_block / 留一载荷(3HP) 两协议，按文件分层抽 2%–100%
                   训练窗口，5 seed，SVM + RF
  C 去混杂         在 k=116 处同时给 SVM/RF 的 4 条件数字，直接回答
                   "收益来自异常多样性、数据量、还是真实新样本"

产出
  results/metrics/dose_learning_curve.json
  results/figures/fig27_dose_learning.png
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
from cwru_diag.evaluation import save_json                      # noqa: E402
from cwru_diag.models import make_rf, make_svm                  # noqa: E402
from cwru_diag.pipeline import features_for, load_all, time_block_split  # noqa: E402
from cwru_diag.sample_efficiency import (dose_subsets, marginal_gain,  # noqa: E402
                                         stratified_file_subsets)
from cwru_diag.plots import save, setup_style                   # noqa: E402

ANOMALY_FID = 217
REAL_IR021_FILES = (209, 210, 211)      # 同尺寸真实内圈文件，分属 0/1/2 HP
DOSES = (0, 10, 25, 50, 75, 116, 200, 300, 478)
N_REPEATS = 5
FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
SEEDS = (0, 1, 2, 3, 4)
FOCUS_DOSE = 116


def fit_acc(factory, X, y, tr_idx, te_idx) -> float:
    clf = factory().fit(X[tr_idx], y[tr_idx])
    return float((clf.predict(X[te_idx]) == y[te_idx]).mean())


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    c_svm = float(cfg["classical"]["svm"]["C"])
    n_trees = int(cfg["classical"]["rf"]["n_estimators"])
    svm_f = lambda: make_svm(C=c_svm, seed=0)
    rf_f = lambda: make_rf(n_estimators=n_trees, seed=0)

    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    y = np.asarray(ws.label)
    X, _ = features_for(ws, ("time", "freq", "env"), cfg)
    is_217 = ws.file_id == ANOMALY_FID
    idx_217 = np.where(is_217)[0]
    base_idx = np.where((ws.load_hp == 3) & ~is_217)[0]
    print(f"{len(ws)} 窗口 / 217={len(idx_217)} / 3HP 其余={len(base_idx)} / "
          f"特征 {X.shape[1]} 维")

    subsets = dose_subsets(idx_217, DOSES, seed=0, n_repeats=N_REPEATS)
    prefix = dose_subsets(idx_217, DOSES, seed=0, prefix=True)

    # ---------- A 剂量曲线 ----------
    print("\n=== A 完整剂量曲线（3HP→0/1/2）===")
    out_a: dict = {}
    for dst in (0, 1, 2):
        te_idx = np.where(ws.load_hp == dst)[0]
        borrow_pool = np.where(np.isin(ws.file_id, REAL_IR021_FILES)
                               & (ws.load_hp != dst))[0]
        print(f"  目标 {dst}HP：测试 {len(te_idx)} 窗，可借真实样本池 {len(borrow_pool)} 窗")
        row: dict = {"n_test": int(len(te_idx)),
                     "borrow_pool_size": int(len(borrow_pool)),
                     "borrow_pool_files": [int(f) for f in np.unique(ws.file_id[borrow_pool])],
                     "conditions": {}}
        for cond in ("217_random", "upsample", "borrow_real", "prefix"):
            row["conditions"][cond] = {}
            for k in DOSES:
                vals, ntr = [], []
                reps = 1 if cond == "prefix" else N_REPEATS
                for rep in range(reps):
                    rng = np.random.default_rng(1000 + dst * 97 + k * 7 + rep)
                    if cond == "prefix":
                        extra = prefix[str(k)][0]
                    elif cond == "217_random":
                        extra = subsets[str(k)][rep]
                    elif cond == "upsample":
                        extra = rng.choice(base_idx, size=k, replace=True)
                    else:
                        extra = rng.choice(borrow_pool, size=k,
                                           replace=k > len(borrow_pool))
                    tr_idx = np.concatenate([base_idx, extra])
                    vals.append(fit_acc(svm_f, X, y, tr_idx, te_idx))
                    ntr.append(int(len(tr_idx)))
                row["conditions"][cond][str(k)] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "per_repeat": [float(v) for v in vals],
                    "n_train": int(np.mean(ntr)),
                }
        out_a[str(dst)] = row
        cr = row["conditions"]["217_random"]
        print(f"    217 随机剂量: " + " ".join(
            f"{k}={cr[str(k)]['mean']:.3f}" for k in DOSES))
        print(f"    借真实样本:   " + " ".join(
            f"{k}={row['conditions']['borrow_real'][str(k)]['mean']:.3f}" for k in DOSES))
        print(f"    (耗时 {time.time()-t0:.0f}s)")

    # ---------- B 学习曲线 ----------
    print("\n=== B 学习曲线（文件分层抽样）===")
    tr_tb, te_tb = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
    protocols = {
        "time_block": (np.where(tr_tb)[0], np.where(te_tb)[0]),
        "loo_load3": (np.where(ws.load_hp != 3)[0], np.where(ws.load_hp == 3)[0]),
    }
    out_b: dict = {}
    for pname, (tr_idx, te_idx) in protocols.items():
        out_b[pname] = {"n_test": int(len(te_idx)), "models": {}}
        for mname, factory in (("SVM", svm_f), ("RF", rf_f)):
            curve, ns = {}, {}
            for frac in FRACTIONS:
                vals, n_train = [], []
                for seed in SEEDS:
                    sub = stratified_file_subsets(
                        ws.file_id[tr_idx], (frac,), seed=seed,
                        min_per_file=2)[str(frac)]
                    idx = tr_idx[sub]
                    vals.append(fit_acc(factory, X, y, idx, te_idx))
                    n_train.append(int(len(idx)))
                curve[str(frac)] = float(np.mean(vals))
                ns[str(frac)] = {"mean": float(np.mean(n_train)),
                                 "std": float(np.std(vals))}
            out_b[pname]["models"][mname] = {
                "curve": curve, "std": ns,
                "marginal_gain": marginal_gain(curve),
            }
            print(f"  {pname:10s} {mname:3s} " + " ".join(
                f"{f}={curve[str(f)]:.3f}" for f in FRACTIONS)
                + f"  ({time.time()-t0:.0f}s)")

    # ---------- C 去混杂（k=116，加 RF）----------
    print("\n=== C k=116 去混杂（SVM + RF）===")
    out_c: dict = {}
    for dst in (0, 1, 2):
        te_idx = np.where(ws.load_hp == dst)[0]
        borrow_pool = np.where(np.isin(ws.file_id, REAL_IR021_FILES)
                               & (ws.load_hp != dst))[0]
        row = {}
        for cond in ("without", "217_prefix", "217_random", "upsample", "borrow_real"):
            vals_by_model = {"SVM": [], "RF": []}
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(3000 + dst * 97 + rep)
                if cond == "without":
                    extra = np.empty(0, int)
                elif cond == "217_prefix":
                    extra = prefix[str(FOCUS_DOSE)][0]
                elif cond == "217_random":
                    extra = subsets[str(FOCUS_DOSE)][rep]
                elif cond == "upsample":
                    extra = rng.choice(base_idx, size=FOCUS_DOSE, replace=True)
                else:
                    extra = rng.choice(borrow_pool, size=FOCUS_DOSE,
                                       replace=FOCUS_DOSE > len(borrow_pool))
                tr_idx = np.concatenate([base_idx, extra])
                for mname, factory in (("SVM", svm_f), ("RF", rf_f)):
                    vals_by_model[mname].append(fit_acc(factory, X, y, tr_idx, te_idx))
            row[cond] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                         for m, v in vals_by_model.items()}
        out_c[str(dst)] = row
        print(f"  {dst}HP " + " ".join(
            f"{c}={row[c]['SVM']['mean']:.3f}" for c in row))

    # ---------- 派生结论 ----------
    peaks = {}
    for dst, row in out_a.items():
        cr = row["conditions"]["217_random"]
        k_best = max(DOSES, key=lambda k: cr[str(k)]["mean"])
        peaks[dst] = {"peak_dose": int(k_best),
                      "peak_accuracy": cr[str(k_best)]["mean"],
                      "zero_accuracy": cr["0"]["mean"],
                      "full_accuracy": cr["478"]["mean"],
                      "prefix116": row["conditions"]["prefix"]["116"]["mean"],
                      "random116_mean": cr["116"]["mean"],
                      "random116_std": cr["116"]["std"],
                      "borrow116": row["conditions"]["borrow_real"]["116"]["mean"]}

    out = {
        "protocol": {"doses": list(DOSES), "n_repeats": N_REPEATS,
                     "fractions": list(FRACTIONS), "seeds": list(SEEDS),
                     "focus_dose": FOCUS_DOSE,
                     "real_ir021_files": list(REAL_IR021_FILES),
                     "note": "3HP→目标载荷；borrow_real 只从非目标载荷的同尺寸真实文件"
                             "借用，避免同文件既训又测；upsample 与 borrow 在 k 超过池"
                             "大小时有放回。prefix 复现 v1.2 的固定前 116 窗口径。"},
        "A_dose_curve": out_a, "B_learning_curve": out_b,
        "C_deconfound": out_c, "D_peaks": peaks,
    }
    save_json(out, ROOT / "results" / "metrics" / "dose_learning_curve.json")
    print(f"\n完成 → results/metrics/dose_learning_curve.json ({time.time()-t0:.0f}s)")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    ax = axes[0]
    cond_style = {"217_random": ("#E45756", "-"), "upsample": ("#9AA3AD", "--"),
                  "borrow_real": ("#54A24B", "-"), "prefix": ("#4C78A8", ":")}
    for cond, (col, ls) in cond_style.items():
        means, stds = [], []
        for k in DOSES:
            v = [out_a[d]["conditions"][cond][str(k)]["mean"] for d in ("0", "1", "2")]
            means.append(float(np.mean(v)))
            stds.append(float(np.std(v)))
        ax.plot(range(len(DOSES)), means, ls, marker="o", ms=3.5,
                color=col, label=cond)
        if cond == "217_random":
            ax.fill_between(range(len(DOSES)), np.array(means) - 0.5 * np.array(stds),
                            np.array(means) + 0.5 * np.array(stds), color=col, alpha=0.12)
    ax.set_xticks(range(len(DOSES)), [str(k) for k in DOSES], fontsize=7)
    ax.set_xlabel("217.mat 剂量 k（窗口数）"); ax.set_ylabel("3 个目标载荷平均准确率")
    ax.set_title("A 完整剂量曲线（5 随机子集）"); ax.legend(fontsize=7)

    ax = axes[1]
    for pname, col in (("time_block", "#4C78A8"), ("loo_load3", "#E45756")):
        for m, ls, mk in (("SVM", "-", "o"), ("RF", "--", "s")):
            cv = out_b[pname]["models"][m]["curve"]
            ax.plot(range(len(FRACTIONS)), [cv[str(f)] for f in FRACTIONS],
                    ls, marker=mk, ms=3, color=col, label=f"{pname}/{m}")
    ax.set_xticks(range(len(FRACTIONS)), [f"{int(f*100)}%" for f in FRACTIONS], fontsize=7)
    ax.set_xlabel("训练窗口比例（文件分层）"); ax.set_ylabel("准确率")
    ax.set_ylim(0.5, 1.03); ax.set_title("B 样本效率学习曲线")
    ax.legend(fontsize=7)

    ax = axes[2]
    conds = ("without", "217_prefix", "217_random", "upsample", "borrow_real")
    x = np.arange(len(conds)); w = 0.35
    for j, (dst, col) in enumerate((("0", "#4C78A8"), ("1", "#E45756"))):
        vals = [out_c[dst][c]["SVM"]["mean"] for c in conds]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, alpha=0.85, label=f"目标 {dst}HP")
    ax.set_xticks(x, conds, fontsize=7, rotation=20)
    ax.set_ylim(0.85, 1.005); ax.set_ylabel("SVM 准确率")
    ax.set_title(f"C k={FOCUS_DOSE} 去混杂"); ax.legend(fontsize=7)
    fig.tight_layout()
    p = save(fig, "fig27_dose_learning.png")
    print(f"图 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
