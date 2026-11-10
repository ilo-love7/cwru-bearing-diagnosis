"""12 · 217.mat 异常件敏感性双口径：含/剔两套数字 + 训练污染 + 特征层证据。

217.mat 官方列为 IR0021@3HP，但实测 1752 rpm（3HP 其余 1721–1728）、
时长 41 s（其余 ~10 s）、RMS 0.760 超出全部内圈文件上界、包络以 BPFO 为主。
既有报告只给了 SVM 的"剔 217 后均值"，本脚本把它升级为系统化口径：

  A 双口径主表   凡测试集含 217 的场景（LOO 第 3 折、单源 0→3、10 分类 LOO），
                 所有模型/特征组都给 含217 / 剔217 两套数字
  B 训练污染     217 在训练侧时（LOO 0/1/2 折、以 3HP 为源），剔掉它再训
                 是否改变目标载荷上的准确率
  C 分层诊断     217 相对内圈同侪的特征 z 分数排序；各模型对 217 窗口的预测分布

产出
  results/metrics/sensitivity_217.json
  results/figures/fig21_217_sensitivity.png
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
from cwru_diag.diagnostics import feature_zscores, prediction_breakdown  # noqa: E402
from cwru_diag.evaluation import evaluate, save_json            # noqa: E402
from cwru_diag.models import make_rf, make_svm, predict_cnn, train_cnn  # noqa: E402
from cwru_diag.ordinal import hierarchical_scores, to_labels10  # noqa: E402
from cwru_diag.pipeline import carve_val, features_for, load_all  # noqa: E402

ANOMALY_FID = 217
LOADS = [0, 1, 2, 3]


def dual_eval(y_true: np.ndarray, y_pred: np.ndarray,
              keep_mask: np.ndarray) -> dict:
    """同一预测向量上的 含 / 剔 两套指标（keep_mask=False 处为剔除样本）。"""
    with_m = evaluate(y_true, y_pred, LABELS)
    if (~keep_mask).sum() == 0:
        return {"with": with_m, "without": None}
    wo_m = evaluate(y_true[keep_mask], y_pred[keep_mask], LABELS)
    return {"with": with_m, "without": wo_m,
            "delta_accuracy": with_m["accuracy"] - wo_m["accuracy"]}


def train_pred(X, y, tr, te, factory, seed=0):
    clf = factory(seed).fit(X[tr], y[tr])
    return clf.predict(X[te])


def cnn_pred(ws, y, tr, te, cfg, seed=0, epochs=20):
    tr2, va = carve_val(ws, tr, val_frac=0.1)
    res = train_cnn(ws.X[tr2], y[tr2], ws.X[va], y[va], n_classes=len(LABELS),
                    epochs=epochs, batch_size=int(cfg["cnn"]["batch_size"]),
                    lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
                    dropout=float(cfg["cnn"]["dropout"]), seed=seed, verbose=False)
    return predict_cnn(res.model, ws.X[te])


def main() -> int:
    cfg = load_config()
    # 工厂闭包只收 seed（train_pred 按 factory(seed) 调用）
    _svm = lambda seed: make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=seed)
    _rf = lambda seed: make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=seed)
    ws_all = load_all(cfg, ("main",))
    ws = ws_all.subset(ws_all.role == "main")
    y = np.asarray(ws.label)
    X_all, names_all = features_for(ws, ("time", "freq", "env"), cfg)
    X_env, names_env = features_for(ws, ("env",), cfg)
    is_217 = ws.file_id == ANOMALY_FID
    n_217 = int(is_217.sum())
    print(f"{len(ws)} 窗口（217.mat 占 {n_217}） / 全特征 {X_all.shape[1]} 维 / env {X_env.shape[1]} 维")

    out: dict = {"anomaly_file": ANOMALY_FID, "n_217_windows": n_217,
                 "n_total": int(len(ws)), "A_dual_protocol": {},
                 "B_train_pollution": {}, "C_stratum": {}}

    # ---------- A 双口径主表 ----------
    print("\n=== A 双口径主表 ===")
    # A 组训练侧为 0/1/2（不含 217，217 只出现在 3HP），敏感性只作用在测试侧
    scenarios = {
        "loo_fold_3": (ws.load_hp != 3, ws.load_hp == 3),
        "single_source_0_to_3": (ws.load_hp == 0, ws.load_hp == 3),
    }
    for sc_name, (tr, te) in scenarios.items():
        keep = ~is_217[te] if is_217[te].any() else np.ones(te.sum(), bool)
        row = {"n_test_with": int(te.sum()),
               "n_test_without": int(keep.sum()),
               "models": {}}
        for mname, X, factory in (("SVM_all", X_all, _svm), ("SVM_env", X_env, _svm),
                                  ("RF_all", X_all, _rf), ("RF_env", X_env, _rf)):
            pred = train_pred(X, y, tr, te, factory)
            row["models"][mname] = dual_eval(y[te], pred, keep)
            w = row["models"][mname]["with"]["accuracy"]
            wo = row["models"][mname]["without"]["accuracy"]
            print(f"  {sc_name:22s} {mname:8s} 含={w:.4f} 剔={wo:.4f} Δ={w - wo:+.4f}")
        t0 = time.time()
        pred = cnn_pred(ws, y, tr, te, cfg, seed=0, epochs=20)
        row["models"]["CNN"] = dual_eval(y[te], pred, keep)
        w = row["models"]["CNN"]["with"]["accuracy"]
        wo = row["models"]["CNN"]["without"]["accuracy"]
        print(f"  {sc_name:22s} {'CNN':8s} 含={w:.4f} 剔={wo:.4f} Δ={w - wo:+.4f}"
              f"  ({time.time() - t0:.0f}s)")
        out["A_dual_protocol"][sc_name] = row

    # 10 分类 LOO 第 3 折（类型正确率同步给两套）
    print("  -- 10 分类 LOO 第 3 折 --")
    y10 = to_labels10(ws)
    tr, te = ws.load_hp != 3, ws.load_hp == 3
    keep = ~is_217[te]
    row10 = {"n_test_with": int(te.sum()), "n_test_without": int(keep.sum()), "models": {}}
    for mname, X, factory in (("SVM_all", X_all, _svm), ("RF_all", X_all, _rf)):
        pred = train_pred(X, y10, tr, te, factory)
        h_with = hierarchical_scores(y10[te], pred)
        h_wo = hierarchical_scores(y10[te][keep], pred[keep])
        row10["models"][mname] = {"with": h_with, "without": h_wo}
        print(f"  {mname:8s} 含 acc={h_with['size_level_accuracy']:.4f}/type={h_with['type_accuracy']:.4f}"
              f"  剔 acc={h_wo['size_level_accuracy']:.4f}/type={h_wo['type_accuracy']:.4f}")
    out["A_dual_protocol"]["loo_fold_3_10class"] = row10

    # ---------- B 训练污染 ----------
    print("\n=== B 训练污染（217 在训练侧）===")
    # B1: LOO 折 0/1/2 —— 训练含 3HP（含 217）vs 训练剔 217，测试不含 217
    b1 = {}
    for load in [0, 1, 2]:
        te = ws.load_hp == load
        tr_full = ws.load_hp != load
        tr_wo = tr_full & ~is_217
        row = {}
        for mname, X, factory in (("SVM_all", X_all, _svm), ("RF_all", X_all, _rf),
                                  ("SVM_env", X_env, _svm)):
            p_full = train_pred(X, y, tr_full, te, factory)
            p_wo = train_pred(X, y, tr_wo, te, factory)
            a_full = evaluate(y[te], p_full, LABELS)["accuracy"]
            a_wo = evaluate(y[te], p_wo, LABELS)["accuracy"]
            row[mname] = {"train_with_217": a_full, "train_without_217": a_wo,
                          "delta": a_full - a_wo,
                          "n_train_full": int(tr_full.sum()), "n_train_wo": int(tr_wo.sum())}
            print(f"  LOO {load}HP {mname:8s} 训练含={a_full:.4f} 训练剔={a_wo:.4f} Δ={a_full - a_wo:+.4f}")
        b1[str(load)] = row
    out["B_train_pollution"]["loo_folds_0_1_2"] = b1

    # B2: 源=3HP（含 217）→ 目标 0/1/2
    b2 = {}
    for dst in [0, 1, 2]:
        te = ws.load_hp == dst
        tr_full = ws.load_hp == 3
        tr_wo = tr_full & ~is_217
        row = {}
        for mname, X, factory in (("SVM_all", X_all, _svm), ("RF_all", X_all, _rf)):
            p_full = train_pred(X, y, tr_full, te, factory)
            p_wo = train_pred(X, y, tr_wo, te, factory)
            a_full = evaluate(y[te], p_full, LABELS)["accuracy"]
            a_wo = evaluate(y[te], p_wo, LABELS)["accuracy"]
            row[mname] = {"train_with_217": a_full, "train_without_217": a_wo,
                          "delta": a_full - a_wo}
            print(f"  3→{dst}HP {mname:8s} 训练含={a_full:.4f} 训练剔={a_wo:.4f} Δ={a_full - a_wo:+.4f}")
        b2[str(dst)] = row
    out["B_train_pollution"]["source_3hp"] = b2

    # B3 去混杂：217 有 478 窗（普通文件 ~116 窗），"训练含 217" 同时也是"多 32% 数据"。
    # 四个条件分离"数据量"与"异常多样性"：
    #   full      —— 217 全部 478 窗
    #   capped    —— 217 只取 116 窗（与普通文件同量）
    #   without   —— 剔除 217
    #   upsampled —— 剔除 217，但从其余 3HP 文件有放回重采样 478 窗补足数据量
    rng = np.random.default_rng(0)
    is_3 = ws.load_hp == 3
    other3 = np.where(is_3 & ~is_217)[0]
    extra_idx = rng.choice(other3, size=478, replace=True)   # 有放回，可能重复
    base_idx = np.where(is_3 & ~is_217)[0]
    idx_217 = np.where(is_217)[0]
    # 用索引数组而非掩码：upsample 条件含重复样本，掩码会去重
    conds = {
        "full": np.concatenate([base_idx, idx_217]),
        "capped116": np.concatenate([base_idx, idx_217[:116]]),
        "without": base_idx,
        "upsampled478": np.concatenate([base_idx, extra_idx]),
    }
    b3 = {}
    for dst in [0, 1, 2]:
        te = ws.load_hp == dst
        row = {}
        for cname, tr_idx in conds.items():
            entry = {"n_train": int(len(tr_idx))}
            for mname, factory in (("SVM_all", _svm), ("RF_all", _rf)):
                pred = factory(0).fit(X_all[tr_idx], y[tr_idx]).predict(X_all[te])
                entry[mname] = float((pred == y[te]).mean())
            row[cname] = entry
            print(f"  3→{dst} {cname:12s} n={entry['n_train']:5d} "
                  f"SVM={entry['SVM_all']:.4f} RF={entry['RF_all']:.4f}")
        b3[str(dst)] = row
    out["B_train_pollution"]["volume_deconfound"] = b3

    # ---------- C 分层诊断 ----------
    print("\n=== C 217 分层诊断 ===")
    ref_mask = (y == LABELS.index("IR")) & ~is_217          # 其余全部内圈文件
    z_rows = feature_zscores(X_all, names_all, ref_mask, is_217)
    out["C_stratum"]["feature_zscores_vs_ir_peers"] = z_rows[:15]
    print("  特征 z 分数 top8（217 vs 其余内圈）:")
    for r in z_rows[:8]:
        print(f"    {r['name']:22s} z={r['z']:+.2f}  217中位={r['median_target']:.4f}"
              f"  同侪中位={r['median_ref']:.4f}")

    tr3, te3 = ws.load_hp != 3, ws.load_hp == 3
    pred_map = {}
    for mname, X, factory in (("SVM_all", X_all, _svm), ("SVM_env", X_env, _svm),
                              ("RF_all", X_all, _rf), ("RF_env", X_env, _rf)):
        pred_map[mname] = train_pred(X, y, tr3, te3, factory)
    breakdown = {}
    for mname, pred in pred_map.items():
        breakdown[mname] = prediction_breakdown(pred[is_217[te3]], LABELS)
        fr = breakdown[mname]["fraction"]
        print(f"  {mname:8s} 217 窗口预测分布: " +
              ", ".join(f"{k}={v:.3f}" for k, v in fr.items() if v > 0))
    out["C_stratum"]["prediction_breakdown_217"] = breakdown

    save_json(out, ROOT / "results" / "metrics" / "sensitivity_217.json")
    print("\n完成 → results/metrics/sensitivity_217.json")

    # ---------- 图 ----------
    from cwru_diag.plots import save, setup_style
    import matplotlib.pyplot as plt
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    settings, w_vals, wo_vals = [], [], []
    for sc in ("loo_fold_3", "single_source_0_to_3"):
        for mname in ("SVM_all", "SVM_env", "RF_all", "RF_env", "CNN"):
            d = out["A_dual_protocol"][sc]["models"][mname]
            settings.append(("0→3" if "single" in sc else "LOO3") + f"\n{mname}")
            w_vals.append(d["with"]["accuracy"])
            wo_vals.append(d["without"]["accuracy"])
    x = np.arange(len(settings))
    ax.bar(x - 0.18, w_vals, 0.36, label="含 217", color="#E45756")
    ax.bar(x + 0.18, wo_vals, 0.36, label="剔 217", color="#4C78A8")
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=6.5)
    ax.set_ylim(0.6, 1.02); ax.set_ylabel("准确率")
    ax.set_title("A 双口径：测试集含/剔 217.mat")
    ax.legend(fontsize=7)

    ax = axes[1]
    top = out["C_stratum"]["feature_zscores_vs_ir_peers"][:8][::-1]
    names = [r["name"] for r in top]
    zs = [r["z"] for r in top]
    cols = ["#E45756" if z > 0 else "#54A24B" for z in zs]
    ax.barh(names, zs, color=cols)
    ax.axvline(0, color="#9AA3AD", lw=0.8)
    ax.set_xlabel("z（217 中位数 vs 其余内圈）")
    ax.set_title("C 特征层证据：217 的异常特征 top8")
    fig.tight_layout()
    p = save(fig, "fig21_217_sensitivity.png")
    print(f"图 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
