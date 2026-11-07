"""03 · 1D-CNN 对照：原始波形端到端，与手工特征模型同协议比较。

协议
  random_window  有泄漏（同上界参考）
  time_block     无泄漏，记录内时间块划分

早停验证集从训练集中再切 10%（按时间块），测试集全程不参与任何选择。

产出
  results/metrics/cnn_random_window.json
  results/metrics/cnn_time_block.json
  results/models/cnn_time_block.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                        # noqa: E402
from cwru_diag.dataset import LABELS                            # noqa: E402
from cwru_diag.evaluation import aggregate_runs, evaluate, save_json, summarize  # noqa: E402
from cwru_diag.models import predict_cnn, train_cnn             # noqa: E402
from cwru_diag.pipeline import (carve_val, load_all, random_window_split,  # noqa: E402
                                time_block_split)

SEEDS_CNN = [0, 1]


def main() -> int:
    cfg = load_config()
    ws = load_all(cfg, ("main",))
    main_mask = ws.role == "main"
    ws = ws.subset(main_mask)
    X = ws.X
    y = np.asarray(ws.label)
    print(f"主实验窗口 {X.shape}，类别分布 {np.bincount(y).tolist()}")

    out = {}
    for protocol in ("random_window", "time_block"):
        print(f"\n=== CNN / 协议 {protocol} ===")
        runs = []
        for seed in SEEDS_CNN:
            if protocol == "random_window":
                tr, te = random_window_split(ws, 0.3, seed)
            else:
                tr, te = time_block_split(ws, 0.3, int(cfg["guard_windows"]))
            tr2, va = carve_val(ws, tr, val_frac=0.1)
            t0 = time.time()
            res = train_cnn(
                X[tr2], y[tr2], X[va], y[va], n_classes=len(LABELS),
                epochs=int(cfg["cnn"]["epochs"]), batch_size=int(cfg["cnn"]["batch_size"]),
                lr=float(cfg["cnn"]["lr"]), weight_decay=float(cfg["cnn"]["weight_decay"]),
                dropout=float(cfg["cnn"]["dropout"]), seed=seed, verbose=True,
            )
            pred = predict_cnn(res.model, X[te])
            m = evaluate(y[te], pred, LABELS)
            runs.append({"seed": seed, "metrics": m,
                         "best_epoch": res.best_epoch, "best_val_acc": res.best_val_acc,
                         "history": res.history})
            print(f"  seed={seed}  {summarize(m)}  best_epoch={res.best_epoch}  "
                  f"({time.time() - t0:.0f}s)")

            if protocol == "time_block" and seed == 0:
                path = ROOT / "results" / "models" / "cnn_time_block.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(res.model.state_dict(), path)

        agg = aggregate_runs(runs, len(LABELS))
        out[protocol] = {"protocol": protocol, "n_train": int(tr2.sum()),
                         "n_val": int(va.sum()), "n_test": int(te.sum()),
                         "models": {"CNN": agg}}
        save_json(out[protocol], ROOT / "results" / "metrics" / f"cnn_{protocol}.json")
        print(f"  → CNN acc={agg['mean']['accuracy']:.4f}±{agg['std']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
