"""域适应：特征级边际对齐（CORAL / KMM / 子空间收缩）+ 结构诊断。

v1.2 的迁移矩阵（报告 §3.12）测出 0→3 = 0.739 而 3→0 = 0.948 的不对称，
§3.12-C 又证明特征里仍残留强载荷信息（全特征载荷探针 0.895）。按常规思路，
"把源域边际分布对齐到目标域"应当能补上缺口。本模块提供三种**特征级**闭式对齐，
与 N2 v1.3 的深度表征对齐损失（CORAL/MMD 加在隐层上）互补：这里变换直接作用于
48 维物理特征，可直接喂给 SVM/RF，且变换矩阵可检查、可单测。

  CORAL  二阶统计白化-重着色（Sun et al. 2016 的闭式解，精确匹配协方差）
  KMM    核均值匹配（Huang et al. 2006 的无约束闭式解 + clip/归一化近似约束）
  shrink 部分对齐插值：W = (1−s)·W_CORAL + s·I，用于扫对齐强度

诊断（只用目标域**事后**标签，不参与任何训练/变换拟合）：
  fisher_ratio           类间/类内散布比，衡量边际对齐是否毁掉类条件结构
  class_conditional_drift 对齐前后，各类源中心到目标同标签中心的距离

所有变换函数都是纯 numpy，输入 (n, d) 浮点数组。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- 对称矩阵幂


def _sym_power(a: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    """对称矩阵的特征分解幂：A^p = U diag(max(λ,eps)^p) Uᵀ。"""
    vals, vecs = np.linalg.eigh(np.asarray(a, dtype=np.float64))
    vals = np.clip(vals, eps, None)
    return (vecs * (vals ** power)) @ vecs.T


def _cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("协方差需要至少 2 行样本")
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / x.shape[0]


# ---------------------------------------------------------------- 分布距离


def coral_distance(xs: np.ndarray, xt: np.ndarray) -> float:
    """CORAL 距离 ‖Cs − Ct‖_F² / (4 d²)。"""
    xs, xt = np.asarray(xs, float), np.asarray(xt, float)
    if xs.shape[1] != xt.shape[1]:
        raise ValueError("源/目标特征维数不一致")
    cs, ct = _cov(xs), _cov(xt)
    d = xs.shape[1]
    return float(((cs - ct) ** 2).sum() / (4.0 * d * d))


def _rbf(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def median_bandwidth(xs: np.ndarray, xt: np.ndarray) -> float:
    """中位数启发式带宽尺度：拼接后各维标准差的中位数（下界 1e-6）。"""
    x = np.vstack([np.asarray(xs, float), np.asarray(xt, float)])
    return float(max(np.median(np.std(x, axis=0)), 1e-6))


def mmd_rbf(xs: np.ndarray, xt: np.ndarray,
            bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> float:
    """多带宽 RBF-MMD²（有偏估计，非负）。"""
    xs, xt = np.asarray(xs, float), np.asarray(xt, float)
    if xs.shape[1] != xt.shape[1]:
        raise ValueError("源/目标特征维数不一致")
    if len(xs) < 2 or len(xt) < 2:
        return 0.0
    scale = median_bandwidth(xs, xt)
    total = 0.0
    for bw in bandwidths:
        k = _rbf(xs, xt, bw * scale)
        total += (_rbf(xs, xs, bw * scale).mean()
                  + _rbf(xt, xt, bw * scale).mean() - 2.0 * k.mean())
    return float(max(total / len(bandwidths), 0.0))


# ---------------------------------------------------------------- CORAL 变换


@dataclass
class CoralTransform:
    """仿射变换 x' = x @ W + b；shrink=1 时 W=I、b=0，精确退化为恒等。"""

    W: np.ndarray
    b: np.ndarray
    shrink: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.W + self.b


def feature_scale(xs: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """拼接后各维标准差（下界 1e-6），用于把病态量纲拉回同一尺度。"""
    x = np.vstack([np.asarray(xs, float), np.asarray(xt, float)])
    return np.clip(np.std(x, axis=0), 1e-6, None)


def fit_coral(xs: np.ndarray, xt: np.ndarray, eps: float = 1e-6,
              shrink: float = 0.0, scale: np.ndarray | None = None) -> CoralTransform:
    """拟合 CORAL：W = Cs^(−1/2) @ Ct^(1/2)，精确使 Wᵀ Cs W = Ct。

    **先按各维标准差归一化再拟合**：真实物理特征里 td_peak、env_*_snr 等量纲差
    好几个数量级，直接在原始空间做矩阵幂会因条件数爆炸而把协方差放大几个数量级
    （v1.3 E32 首轮实测踩到）。设 D=diag(scale)，在标准化空间得到 W' 后，
    原空间的变换是相似变换 W = D⁻¹ W' D，仍有 Wᵀ Cs W = Ct 精确成立。

    shrink ∈ [0,1] 是**样本层面的凸组合**强度：
      x' = (1−s)·[(x−mu_s)@W_full + mu_t] + s·x
    因此 s=1 精确恒等（W=I, b=0），s=0 完全对齐；中间是凸组合而非矩阵幂插值，
    数值稳定且"唯一变量是对齐强度"成立（矩阵插值配合均值平移会让 s=1 不恒等）。
    """
    xs, xt = np.asarray(xs, dtype=np.float64), np.asarray(xt, dtype=np.float64)
    if xs.ndim != 2 or xt.ndim != 2 or xs.shape[1] != xt.shape[1]:
        raise ValueError("xs/xt 须为同列数的 (n, d) 数组")
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink 须在 [0,1]")
    if eps <= 0.0:
        raise ValueError("eps 须为正")
    s = feature_scale(xs, xt) if scale is None else np.clip(np.asarray(scale, float), 1e-6, None)
    if s.shape != (xs.shape[1],):
        raise ValueError("scale 长度须等于特征维数")
    inv = 1.0 / s
    xs_n = (xs - xs.mean(axis=0)) * inv
    xt_n = (xt - xt.mean(axis=0)) * inv
    cs, ct = _cov(xs_n), _cov(xt_n)
    w_n = _sym_power(cs, -0.5, eps) @ _sym_power(ct, 0.5, eps)
    # 相似变换回原空间：W = D⁻¹ W' D
    w_full = (inv[:, None] * w_n) * s[None, :]
    d = xs.shape[1]
    mu_s, mu_t = xs.mean(axis=0), xt.mean(axis=0)
    # 完全对齐的仿射偏置；样本凸组合后乘 (1−s)
    b_full = mu_t - mu_s @ w_full
    w = (1.0 - shrink) * w_full + shrink * np.eye(d)
    b = (1.0 - shrink) * b_full
    return CoralTransform(W=w, b=b, shrink=float(shrink))


def cov_match_error(xs_aligned: np.ndarray, xt: np.ndarray) -> float:
    """对齐后源协方差与目标协方差的归一化 Frobenius 差。"""
    return coral_distance(xs_aligned, xt)


# ---------------------------------------------------------------- KMM 权重


def kernel_mean_weights(xs: np.ndarray, xt: np.ndarray,
                        bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                        lam: float = 1e-3, clip: tuple[float, float] = (0.2, 3.0)) -> np.ndarray:
    """KMM：解 (Kss + λ n_s I) w = n_s · mean_t k(x_t, x_s)，再 clip + 归一化到均值 1。

    这是 Huang et al. 2006 的无约束闭式解；原始问题的 w ≥ 0 与 Σw = n_s
    用 clip 与归一化近似（避免引入 QP 依赖）。Xs == Xt 时 w ≈ 1。
    权重用于 sample_weight，不改动特征空间。
    """
    xs, xt = np.asarray(xs, dtype=np.float64), np.asarray(xt, dtype=np.float64)
    if xs.ndim != 2 or xt.ndim != 2 or xs.shape[1] != xt.shape[1]:
        raise ValueError("xs/xt 须为同列数的 (n, d) 数组")
    ns = xs.shape[0]
    if ns < 2 or xt.shape[0] < 2:
        return np.ones(ns, dtype=np.float64)
    # 与 CORAL 同一坑：物理特征量纲跨数量级时，RBF 距离被最大尺度特征主宰，
    # 核矩阵退化成单位阵 → 权重全被 clip 后投影回 1（看似"无需加权"的假象）。
    # 先按各维标准差归一化，再做中位数启发式带宽。
    s = feature_scale(xs, xt)
    xs_n, xt_n = xs / s, xt / s
    scale = median_bandwidth(xs_n, xt_n)
    kss = np.zeros((ns, ns), dtype=np.float64)
    kts = np.zeros(ns, dtype=np.float64)
    for b in bandwidths:
        kss += _rbf(xs_n, xs_n, b * scale) / len(bandwidths)
        kts += _rbf(xt_n, xs_n, b * scale).mean(axis=0) / len(bandwidths)
    rhs = ns * kts
    a = kss + lam * ns * np.eye(ns)
    try:
        w = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(a, rhs, rcond=None)[0]
    lo, hi = clip
    # 两个凸约束（box 与 Σw=n_s）同时满足：交替投影，而不是「clip 后除以均值」——
    # 后者会把刚被截断的上界重新放大出界。两集合交集非空时交替投影必收敛。
    w = np.clip(np.nan_to_num(w, nan=1.0, posinf=hi, neginf=lo), lo, hi)
    for _ in range(200):
        gap = 1.0 - float(w.mean())
        if abs(gap) < 1e-12:
            break
        w = np.clip(w + gap, lo, hi)
    return w


# ---------------------------------------------------------------- 结构诊断


def fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """类间散布 / 类内散布（trace 口径），越大表示类条件结构越清晰。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if len(x) != len(y) or x.ndim != 2:
        raise ValueError("x/y 长度不一致或 x 不是二维")
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    mu = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        xc = x[y == c]
        mc = xc.mean(axis=0)
        between += len(xc) * float(((mc - mu) ** 2).sum())
        within += float(((xc - mc) ** 2).sum())
    denom = max(within, 1e-12)
    return float(between / denom)


def class_conditional_drift(xs: np.ndarray, ys: np.ndarray,
                            xt: np.ndarray, yt: np.ndarray,
                            scale: np.ndarray | None = None) -> dict:
    """每类源中心 → 目标同标签中心的距离（诊断用；yt 绝不参与变换拟合）。

    scale 为空时用源/目标拼接后的各维标准差做无量纲化，避免量纲大的特征主导。
    返回 {"per_class": {label: distance}, "mean": 平均距离}。
    """
    xs, xt = np.asarray(xs, float), np.asarray(xt, float)
    ys, yt = np.asarray(ys), np.asarray(yt)
    if scale is None:
        scale = np.std(np.vstack([xs, xt]), axis=0)
    scale = np.clip(np.asarray(scale, float), 1e-6, None)
    classes = sorted(set(ys.tolist()) & set(yt.tolist()))
    per: dict[str, float] = {}
    for c in classes:
        ms = (xs[ys == c] / scale).mean(axis=0)
        mt = (xt[yt == c] / scale).mean(axis=0)
        per[str(c)] = float(np.linalg.norm(ms - mt))
    return {"per_class": per,
            "mean": float(np.mean(list(per.values()))) if per else 0.0}
