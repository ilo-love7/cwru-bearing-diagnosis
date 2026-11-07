"""数据集装配：清单 + .mat → 带元数据的窗口集（带 npz 缓存）。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .io import load_record
from .segmentation import WindowSet, concat, segment_record

LABELS = ["Normal", "IR", "OR", "B"]
LABEL_TO_IDX = {k: i for i, k in enumerate(LABELS)}


def build_dataset(roles: tuple[str, ...] = ("main", "position", "severity"),
                  cfg: dict | None = None, cache: bool = True) -> WindowSet:
    """读取清单里所有记录，切窗并合并为一个 WindowSet。

    label 用整数编码（顺序见 LABELS）；rpm 优先取 .mat 内实测转速，缺失则回退清单标称值。
    """
    cfg = cfg or load_config()
    manifest = pd.read_csv(ROOT / cfg["manifest"])
    manifest = manifest[manifest["role"].isin(roles)].sort_values("file_id").reset_index(drop=True)

    cache_dir = ROOT / "results" / "cache"
    cache_file = cache_dir / f"windows_{'-'.join(sorted(roles))}_{cfg['window']}_{int(cfg['overlap'] * 100)}.npz"

    if cache and cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return WindowSet(
            X=d["X"], file_id=d["file_id"], block=d["block"], start=d["start"],
            label=d["label"], load_hp=d["load_hp"], severity=d["severity"],
            position=d["position"], role=d["role"], rpm=d["rpm"],
        )

    sets, meta_rows = [], []
    missing = []
    for _, row in manifest.iterrows():
        path = cfg["raw_path"] / row["mat_file"]
        if not path.exists():
            if os.environ.get("CWRU_ALLOW_MISSING") == "1":
                missing.append(row["mat_file"])
                continue
            raise FileNotFoundError(
                f"缺少 {path.name}。先运行 scripts/00_download_data.py --roles {row['role']}")
        rec = load_record(path, fs=cfg["fs"])
        ws = segment_record(rec, window=cfg["window"], overlap=cfg["overlap"])
        rpm = rec.rpm if np.isfinite(rec.rpm) else float(row["rpm"])
        sets.append(ws)
        meta_rows.append({
            "label": LABEL_TO_IDX[row["label"]],
            "load_hp": int(row["load_hp"]),
            "severity": float(row["severity_inch"]),
            "position": str(row["position"]),
            "role": str(row["role"]),
            "rpm": rpm,
            "n_windows": len(ws),
            "duration_s": rec.duration_s,
        })
        print(f"  {row['mat_file']:>11s}  {row['label']:<6s} sev={row['severity_inch']:.3f}\" "
              f"load={row['load_hp']}HP  rpm={rpm:.0f}  {len(ws):4d} windows  {rec.duration_s:6.1f}s")

    if missing:
        print(f"  ⚠ 跳过 {len(missing)} 个缺失文件（CWRU_ALLOW_MISSING=1）：{', '.join(missing)}")
    if not meta_rows:
        raise RuntimeError("没有任何可用文件，先下载数据")

    meta = {k: [m[k] for m in meta_rows] for k in meta_rows[0]}
    ds = concat(sets, meta)

    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file, X=ds.X.astype(np.float32), file_id=ds.file_id, block=ds.block,
            start=ds.start, label=ds.label, load_hp=ds.load_hp, severity=ds.severity,
            position=ds.position.astype(str), role=ds.role.astype(str), rpm=ds.rpm,
        )
        print(f"  缓存 → {cache_file.relative_to(ROOT)}")
    return ds


def record_table(ws: WindowSet) -> pd.DataFrame:
    """窗口级元数据表，便于统计与画图。"""
    return pd.DataFrame({
        "file_id": ws.file_id, "block": ws.block, "start": ws.start,
        "label": [LABELS[i] for i in ws.label], "load_hp": ws.load_hp,
        "severity": ws.severity, "position": ws.position, "role": ws.role, "rpm": ws.rpm,
    })
