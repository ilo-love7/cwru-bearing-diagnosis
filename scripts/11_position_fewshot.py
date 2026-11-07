"""11 · 外圈位置 few-shot 适配：给 0–10% 的目标位置样本，泛化缺口能不能补上。

05 的结论是"位置泛化是真难点"（@6 点训练 → @3/@12 测试 SVM 只有 0.266），
但那只是零样本外推。报告 §5.2 自我限定过：若训练时混入少量 3 点/12 点样本，
结果会不同——本实验补上这条消融曲线。

协议（测试集固定，适配集嵌套，便于横向比较）：
  - 目标 16 个文件（OR007/OR014/OR021 × @3/@12 × 4 载荷）按时间排序；
  - 每文件前 10% 窗口作为"适配池"，池后留 1 个窗口隔离带，其余为固定测试集；
  - 适配集取池内前 {0, 25, 50, 100}%（≈ 目标数据的 0 / 2.5% / 5% / 10%），
    并入主线训练集重新训练，测试集始终不动。

模型：SVM(全特征) / SVM(仅包络阶次) / RF / 1D-CNN 重训 / 1D-CNN 冻结骨干微调头（CNN_ft）。

实测发现把少量目标窗口混进训练集**救不了 CNN**（被 3880 个主线窗口淹没），
所以加了 CNN_ft：主线预训练一次，之后冻结卷积骨干、只用适配窗口重训分类头。

产出
  results/metrics/position_fewshot.json
  results/figures/fig20_position_fewshot.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import evaluate, save_json            # noqa: E402
from cwru_diag.models import (make_rf, make_svm, predict_cnn,   # noqa: E402
                              train_cnn)
from cwru_diag.pipeline import carve_val, features_for, load_all  # noqa: E402
from cwru_diag.plots import save, setup_style                   # noqa: E402

POOL_FRAC = 0.10                    # 每个目标文件切出多少做适配池
ADAPT_OF_POOL = [0.0, 0.25, 0.5, 1.0]
GUARD = 1


def finetune_head(base, X_ad: np.ndarray, y_ad: np.ndarray, seed: int = 0,
                  epochs: int = 100, lr: float = 1e-3, batch: int = 32):
    """冻结卷积骨干，只用适配窗口重训分类头（真正的 few-shot 适配姿势）。"""
    import copy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from cwru_diag.models import _standardize_per_window

    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(_standardize_per_window(X_ad)[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y_ad, dtype=torch.long)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def split_target(ws_all) -> tuple[dict, np.ndarray]:
    """返回（{frac: 适配掩码}, 固定测试掩码）。"""
    pos_idx = np.where(ws_all.role == "position")[0]
    test = np.zeros(len(ws_all), dtype=bool)
    per_file: dict[int, np.ndarray] = {}
    for fid in np.unique(ws_all.file_id[pos_idx]):
        idx = pos_idx[ws_all.file_id[pos_idx] == fid]
        idx = idx[np.argsort(ws_all.start[idx], kind="stable")]
        k = int(round(POOL_FRAC * len(idx)))
        per_file[int(fid)] = idx[:k]                 # 适配池（时间序）
        test[idx[k + GUARD:]] = True                 # 固定测试集
    adapt_masks = {}
    for frac in ADAPT_OF_POOL:
        m = np.zeros(len(ws_all), dtype=bool)
        for fid, pool in per_file.items():
            take = int(round(frac * len(pool)))
            m[pool[:take]] = True
        adapt_masks[frac] = m
    return adapt_masks, test


def main() -> int:
    cfg = load_config()
    ws_all = load_all(cfg)
    y = np.asarray(ws_all.label)
    m_train = ws_all.role == "main"
    adapt_masks, test = split_target(ws_all)
    n_test = int(test.sum())
    print(f"主线训练 {int(m_train.sum())} 窗，固定测试 {n_test} 窗（@3/@12）")

    X_all, _ = features_for(ws_all, ("time", "freq", "env"), cfg)
    X_env, _ = features_for(ws_all, ("env",), cfg)
    or_idx = LABELS.index("OR")

    models_meta = {
        "SVM_all": "feat_all", "SVM_env": "feat_env",
        "RF": "feat_all", "CNN": "raw", "CNN_ft": "ft",
    }
    out: dict = {"n_test": n_test, "pool_frac": POOL_FRAC,
                 "adapt_of_pool": ADAPT_OF_POOL, "models": {}}
    curve: dict[str, list] = {k: [] for k in models_meta}

    # 主线预训练一次，供 CNN_ft 微调头用（每个 frac 复用）
    tr2_base, va = carve_val(ws_all, m_train, val_frac=0.1)
    print("主线预训练 CNN（供 CNN_ft 微调头用）...")
    base_cnn = train_cnn(
        ws_all.X[tr2_base], y[tr2_base], ws_all.X[va], y[va], n_classes=len(LABELS),
        epochs=int(cfg["cnn"]["epochs"]), batch_size=int(cfg["cnn"]["batch_size"]),
        lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
        dropout=float(cfg["cnn"]["dropout"]), seed=0, verbose=False,
    ).model

    for frac in ADAPT_OF_POOL:
        adapt = adapt_masks[frac]
        n_adapt = int(adapt.sum())
        # 验证集只从主线文件里切；适配窗口必须全部留在训练集——
        # 否则 carve_val 会把目标文件仅有的几个块整块切进验证集（踩过：2.5%/5% 时
        # 训练集里适配窗口为 0，曲线与零样本完全相同）。
        tr2 = tr2_base | adapt
        print(f"\n=== 适配集 {frac * POOL_FRAC * 100:.1f}% 目标窗口（{n_adapt} 窗）===")

        row = {}
        for name, kind in models_meta.items():
            t0 = time.time()
            if kind == "raw":
                res = train_cnn(
                    ws_all.X[tr2], y[tr2], ws_all.X[va], y[va], n_classes=len(LABELS),
                    epochs=int(cfg["cnn"]["epochs"]), batch_size=int(cfg["cnn"]["batch_size"]),
                    lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
                    dropout=float(cfg["cnn"]["dropout"]), seed=0, verbose=False,
                )
                pred = predict_cnn(res.model, ws_all.X[test])
                extra = {"best_epoch": res.best_epoch}
            elif kind == "ft":
                if n_adapt == 0:
                    pred = predict_cnn(base_cnn, ws_all.X[test])
                else:
                    model_ft = finetune_head(base_cnn, ws_all.X[adapt], y[adapt], seed=0)
                    pred = predict_cnn(model_ft, ws_all.X[test])
                extra = {"mode": "frozen-backbone head finetune"}
            else:
                feats = X_env if kind == "feat_env" else X_all
                if name == "RF":
                    clf = make_rf(n_estimators=int(cfg["classical"]["rf"]["n_estimators"]), seed=0)
                else:
                    clf = make_svm(C=float(cfg["classical"]["svm"]["C"]), seed=0)
                clf.fit(feats[tr2], y[tr2])
                pred = clf.predict(feats[test])
                extra = {}
            truth = y[test]
            m = evaluate(truth, pred, LABELS)
            row[name] = {
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "or_recall": float((pred == or_idx).mean()),
                "n_adapt": n_adapt, "elapsed_s": round(time.time() - t0, 1), **extra,
            }
            curve[name].append(row[name]["accuracy"])
            print(f"  {name:8s} acc={m['accuracy']:.4f}  OR 召回={row[name]['or_recall']:.4f}"
                  f"  ({row[name]['elapsed_s']}s)")
        out["models"][str(frac)] = row
        save_json(out, ROOT / "results" / "metrics" / "position_fewshot.json")

    # ---------- 图 ----------
    setup_style()
    xs = [f * POOL_FRAC * 100 for f in ADAPT_OF_POOL]
    style = {"SVM_all": ("#4C78A8", "SVM（全特征）"), "SVM_env": ("#F2A93B", "SVM（仅包络阶次）"),
             "RF": ("#72B7B2", "随机森林"), "CNN": ("#E45756", "1D-CNN 重训"),
             "CNN_ft": ("#B279A2", "1D-CNN 微调头（冻结骨干）")}
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for name, (color, lab) in style.items():
        ax.plot(xs, curve[name], "o-", lw=1.7, ms=5, color=color, label=lab)
    ax.set_xlabel("适配集占目标位置窗口的比例（%）")
    ax.set_ylabel("固定测试集准确率")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("外圈位置 few-shot 适配：@3/@12 泛化缺口")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig20_position_fewshot.png")
    print("\n完成 → results/metrics/position_fewshot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
