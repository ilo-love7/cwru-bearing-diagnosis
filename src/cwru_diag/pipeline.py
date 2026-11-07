"""共享流水线：特征缓存 + 三种划分协议，供各脚本复用。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ROOT, load_config
from .dataset import LABELS, build_dataset
from .features import FEATURE_GROUPS, extract
from .segmentation import WindowSet, split_by_time_block


def load_all(cfg: dict | None = None,
             roles: tuple[str, ...] = ("main", "position", "severity")) -> WindowSet:
    cfg = cfg or load_config()
    return build_dataset(roles=roles, cfg=cfg)


def features_for(ws: WindowSet, groups: tuple[str, ...] = ("time", "freq", "env"),
                 cfg: dict | None = None, force: bool = False) -> tuple[np.ndarray, list[str]]:
    """按组抽特征，带磁盘缓存。"""
    cfg = cfg or load_config()
    tag = "-".join(groups)
    # 缓存键必须带上窗口数：同一特征组在不同数据子集（main / 含 position+severity）上
    # 窗口数不同，只按窗口长度做键会把子集结果串用。
    cache = ROOT / "results" / "cache" / f"features_{tag}_w{cfg['window']}_n{len(ws)}.npz"
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=True)
        return d["X"], list(d["names"])
    X, names = extract(ws.X, fs=cfg["fs"], rpm=ws.rpm,
                       geom=_geom(cfg), groups=groups)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X.astype(np.float32), names=np.array(names))
    return X, names


def _geom(cfg: dict):
    from .kinematics import BearingGeometry
    b = cfg["bearing"]
    return BearingGeometry(
        n_balls=int(b["n_balls"]), ball_dia_in=float(b["ball_dia_in"]),
        pitch_dia_in=float(b["pitch_dia_in"]), contact_angle_deg=float(b["contact_angle_deg"]),
    )


def random_window_split(ws: WindowSet, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """【有泄漏】窗口随机划分：同一段记录的重叠窗口会同时出现在训练与测试集。

    保留此协议只为量化“泄漏能把准确率抬高多少”，不作为结论口径。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ws))
    n_test = int(round(len(ws) * test_frac))
    test = np.zeros(len(ws), dtype=bool)
    test[idx[:n_test]] = True
    return ~test, test


def time_block_split(ws: WindowSet, test_frac: float = 0.3,
                     guard_blocks: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """【无泄漏】记录内时间块划分 + 隔离带。"""
    return split_by_time_block(ws, test_frac=test_frac, guard_blocks=guard_blocks)


def load_holdout(ws: WindowSet, holdout_load: int) -> tuple[np.ndarray, np.ndarray]:
    """留一载荷：整个载荷工况作为测试集。"""
    test = ws.load_hp == holdout_load
    return ~test, test


def carve_val(ws: WindowSet, train_mask: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """从训练集里再切一小块做早停验证集（按记录内时间块切，避免重叠泄漏）。

    返回 (新训练掩码, 验证掩码)，两者互斥且都落在原 train_mask 内。
    """
    val = np.zeros(len(ws), dtype=bool)
    for fid in np.unique(ws.file_id[train_mask]):
        m = train_mask & (ws.file_id == fid)
        blocks = np.unique(ws.block[m])
        n_val = max(1, int(round(len(blocks) * val_frac)))
        val_blocks = blocks[-n_val:]
        val[m & np.isin(ws.block, val_blocks)] = True
    train = train_mask & ~val
    return train, val


def labels_of(ws: WindowSet) -> np.ndarray:
    return np.asarray(ws.label)


def idx_to_names(idx: np.ndarray, labels: list[str] | None = None) -> np.ndarray:
    labels = labels or LABELS
    return np.array([labels[i] for i in idx])
