"""单文件诊断演示：读一个 CWRU .mat，输出逐窗预测、类别投票与包络证据。

用法:
    python tools/predict.py D:/.../raw/105.mat
    python tools/predict.py 105.mat --model results/models/rf_time_block.joblib

依赖先跑过 scripts/02_train_classical.py（需要 models/ 与 class_feature_stats.json）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config                 # noqa: E402
from cwru_diag.dataset import LABELS                     # noqa: E402
from cwru_diag.features import FEATURE_GROUPS, extract   # noqa: E402
from cwru_diag.io import load_record                     # noqa: E402
from cwru_diag.kinematics import fault_frequencies       # noqa: E402
from cwru_diag.segmentation import segment_array         # noqa: E402

BAR = "─"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", help=".mat 路径（可只给文件名，会自动去数据目录找）")
    ap.add_argument("--model", default="results/models/rf_time_block.joblib")
    ap.add_argument("--top", type=int, default=6, help="展示几条包络证据")
    args = ap.parse_args()

    cfg = load_config()
    path = Path(args.mat)
    if not path.exists():
        path = Path(cfg["raw_path"]) / args.mat
    if not path.exists():
        print(f"找不到文件: {args.mat}")
        return 2

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"找不到模型 {model_path}，先跑 python scripts/02_train_classical.py")
        return 2

    rec = load_record(path, fs=cfg["fs"])
    X, starts = segment_array(rec.de, cfg["window"], cfg["overlap"])
    feats, names = extract(X, fs=cfg["fs"], rpm=np.full(len(X), rec.rpm),
                           groups=("time", "freq", "env"))
    clf = joblib.load(model_path)
    pred = clf.predict(feats)
    proba = clf.predict_proba(feats) if hasattr(clf, "predict_proba") else None

    counts = {LABELS[i]: int((pred == i).sum()) for i in range(len(LABELS))}
    share = {k: v / len(pred) for k, v in counts.items()}
    top = max(share, key=share.get)

    print(f"\n文件      {path.name}  ({len(rec.de)} 点 / {rec.duration_s:.1f}s @ {cfg['fs'] / 1000:.0f} kHz)")
    print(f"转速      {rec.rpm:.0f} rpm   → 转频 {rec.fr:.2f} Hz")
    ff = fault_frequencies(rec.fr)
    print(f"理论特征  BPFO {ff['BPFO']:.1f} Hz · BPFI {ff['BPFI']:.1f} Hz · "
          f"2×BSF {ff['BSF2']:.1f} Hz · FTF {ff['FTF']:.1f} Hz")
    print(f"窗口      {len(X)} 个 × {cfg['window']} 点")
    print(BAR * 58)
    for name in LABELS:
        bar = "█" * int(round(share[name] * 40))
        print(f"  {name:7s} {share[name]:6.1%}  {bar}")
    print(BAR * 58)
    print(f"结论      {top}（{share[top]:.1%} 的窗口一致）")

    if proba is not None:
        print(f"平均置信  {proba.mean(axis=0).max():.3f}")

    # 包络证据：与“该类训练均值”比较，给出最突出的几条
    stats_path = ROOT / "results" / "metrics" / "class_feature_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        mean = np.array(stats["classes"][top]["mean"])
        std = np.array(stats["classes"][top]["std"]) + 1e-9
        z = (feats.mean(axis=0) - mean) / std
        env_idx = [names.index(n) for n in FEATURE_GROUPS["env"] if n in names]
        ranked = sorted(env_idx, key=lambda i: -abs(z[i]))[:args.top]
        print(f"\n包络证据（相对「{top}」训练集均值的偏离，|z| 越大越典型）")
        for i in ranked:
            print(f"  {names[i]:24s} 实测 {feats.mean(axis=0)[i]:8.3f}   "
                  f"{top} 均值 {mean[i]:8.3f}   z={z[i]:+.2f}")

    print("\n提示：单文件诊断只做一致性检查，正式指标见 results/metrics/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
