"""样本效率：217.mat 剂量曲线的随机子集构造 + 文件分层学习曲线抽样。

v1.2 §3.11-B3 只扫了「117 窗 vs 478 窗」一个剂量点，且 117 窗是**固定前缀**
（报告 §5.6 自己标注为局限）。本模块把剂量曲线做成可复现的随机子集协议：

  dose_subsets            每个剂量 k 抽 n_repeats 个不同的无放回随机子集；
                          另支持 prefix 模式复现 v1.2 的固定前缀口径
  stratified_file_subsets 按文件等比例抽训练窗口（学习曲线），保证小样本下
                          每个文件/每类都还有代表，避免"抽到的全是某一类"
  dose_curve_summary      每个剂量的均值/标准差/极值，供报告表格使用

所有函数只返回索引数组，不碰数据与模型，便于单测钉死语义。
"""
from __future__ import annotations

import numpy as np


def dose_subsets(pool: np.ndarray, sizes: tuple[int, ...], seed: int = 0,
                 n_repeats: int = 5, prefix: bool = False) -> dict[str, list[np.ndarray]]:
    """从 pool（异常文件的全部窗口索引）里按剂量抽子集。

    prefix=True 时每个剂量只给一个「前 k 个」子集（v1.2 口径）；
    否则每个剂量给 n_repeats 个独立随机子集。k 不得超过 pool 大小。
    """
    pool = np.asarray(pool)
    if pool.ndim != 1:
        raise ValueError("pool 须为一维索引数组")
    n = len(pool)
    out: dict[str, list[np.ndarray]] = {}
    for k in sizes:
        k = int(k)
        if not 0 <= k <= n:
            raise ValueError(f"剂量 k={k} 超出 [0, {n}]")
        key = str(k)
        if prefix:
            out[key] = [pool[:k].copy()]
            continue
        rng = np.random.default_rng(int(seed) * 100003 + k)
        reps = []
        for _ in range(n_repeats):
            if k == 0:
                reps.append(np.empty(0, dtype=pool.dtype))
            else:
                reps.append(np.sort(rng.choice(pool, size=k, replace=False)))
        out[key] = reps
    return out


def stratified_file_subsets(file_ids: np.ndarray,
                            fractions: tuple[float, ...],
                            seed: int = 0,
                            min_per_file: int = 2) -> dict[str, np.ndarray]:
    """按文件分层抽样：每个文件抽取 round(f · n_file) 个窗口（下限 min_per_file）。

    返回 {str(fraction): 排序后的索引数组}。同一 fraction 下不同 seed 抽不同子集。
    """
    file_ids = np.asarray(file_ids)
    out: dict[str, np.ndarray] = {}
    for f in fractions:
        f = float(f)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"fraction={f} 须在 (0, 1]")
        rng = np.random.default_rng(int(seed) * 10007 + int(round(f * 10000)))
        picked = []
        for fid in np.unique(file_ids):
            idx = np.where(file_ids == fid)[0]
            n_take = min(len(idx), max(int(min_per_file),
                                       int(round(f * len(idx)))))
            if n_take > 0:
                picked.append(rng.choice(idx, size=n_take, replace=False))
        out[str(f)] = np.sort(np.concatenate(picked)) if picked else np.empty(0, int)
    return out


def dose_curve_summary(rows: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """{剂量: [各次重复的指标]} → 均值/标准差/极值。"""
    out: dict[str, dict[str, float]] = {}
    for k, vals in rows.items():
        v = np.asarray(vals, dtype=np.float64)
        if v.size == 0:
            continue
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                  "min": float(v.min()), "max": float(v.max()),
                  "n_repeats": int(v.size)}
    return out


def marginal_gain(curve: dict[str, float]) -> dict[str, float]:
    """学习曲线相邻剂量的边际增益（按剂量字符串转 float 排序）。"""
    ks = sorted(curve, key=lambda s: float(s))
    return {ks[i]: float(curve[ks[i]] - curve[ks[i - 1]]) for i in range(1, len(ks))}
