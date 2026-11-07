"""窗口分段：把一条长记录切成定长窗口，并保留“块号”以支持无泄漏的分组划分。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import Record


@dataclass
class WindowSet:
    X: np.ndarray                      # (n_windows, window)
    file_id: np.ndarray                # (n,)
    block: np.ndarray                  # (n,) 窗口在记录内的连续块编号（用于时间块划分）
    start: np.ndarray                  # (n,) 起始采样点
    label: np.ndarray = field(default=None)      # (n,) 类别
    load_hp: np.ndarray = field(default=None)
    severity: np.ndarray = field(default=None)
    position: np.ndarray = field(default=None)
    role: np.ndarray = field(default=None)
    rpm: np.ndarray = field(default=None)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(
            X=self.X[mask],
            file_id=self.file_id[mask],
            block=self.block[mask],
            start=self.start[mask],
            label=None if self.label is None else self.label[mask],
            load_hp=None if self.load_hp is None else self.load_hp[mask],
            severity=None if self.severity is None else self.severity[mask],
            position=None if self.position is None else self.position[mask],
            role=None if self.role is None else self.role[mask],
            rpm=None if self.rpm is None else self.rpm[mask],
        )


def segment_array(x: np.ndarray, window: int = 1024, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """按给定重叠率切窗，返回 (窗口数组, 起始索引)。"""
    if not 0 <= overlap < 1:
        raise ValueError("overlap 必须在 [0, 1) 内")
    step = int(round(window * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("步长必须为正")
    if len(x) < window:
        return np.empty((0, window)), np.empty(0, dtype=int)
    starts = np.arange(0, len(x) - window + 1, step, dtype=int)
    idx = starts[:, None] + np.arange(window)[None, :]
    return x[idx], starts


def segment_record(rec: Record, window: int = 1024, overlap: float = 0.5) -> WindowSet:
    """单条记录切窗，block 编号 = 窗口序号 // 8（用于时间块划分的粗粒度分组）。"""
    X, starts = segment_array(rec.de, window, overlap)
    n = len(X)
    return WindowSet(
        X=X,
        file_id=np.full(n, rec.file_id),
        block=np.arange(n) // 8,
        start=starts,
    )


def concat(sets: list[WindowSet], meta: dict[str, np.ndarray]) -> WindowSet:
    """合并多条记录的窗口，并写入按文件广播的元数据。"""
    if not sets:
        raise ValueError("没有可合并的窗口")
    X = np.concatenate([s.X for s in sets])
    file_id = np.concatenate([s.file_id for s in sets])
    block = np.concatenate([s.block for s in sets])
    start = np.concatenate([s.start for s in sets])
    # 让不同文件的 block 不重号
    offsets = {fid: i * 100000 for i, fid in enumerate(np.unique(file_id))}
    block = block + np.array([offsets[f] for f in file_id])

    out = WindowSet(X=X, file_id=file_id, block=block, start=start)
    for key, arr in meta.items():
        setattr(out, key, np.concatenate([np.full(len(s), v) for s, v in zip(sets, arr)]))
    return out


def split_by_time_block(ws: WindowSet, test_frac: float = 0.3, guard_blocks: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """按记录内时间块划分 train/test，并在交界处丢一个隔离带，避免重叠窗口跨界泄漏。

    返回 (train_mask, test_mask)。每个文件内部独立划分，因此各工况在两边都有代表。
    """
    train = np.zeros(len(ws), dtype=bool)
    test = np.zeros(len(ws), dtype=bool)
    for fid in np.unique(ws.file_id):
        m = ws.file_id == fid
        blocks = np.unique(ws.block[m])
        n_test = max(1, int(round(len(blocks) * test_frac)))
        test_blocks = blocks[-n_test:]
        guard = set(blocks[-(n_test + guard_blocks):-n_test].tolist()) if guard_blocks else set()
        test[m & np.isin(ws.block, test_blocks)] = True
        train[m & ~np.isin(ws.block, list(test_blocks) + list(guard))] = True
    return train, test
