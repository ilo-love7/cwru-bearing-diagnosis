"""特征工程：时域统计 + 频域描述 + 包络阶次（物理特征）。

三组特征的定位：
  td_*   时域统计 —— 对冲击敏感（峭度、峰值因子），但随载荷漂移
  fd_*   频域描述 —— 能量分布形状，对工况有一定鲁棒性
  env_*  包络阶次 —— 用实测转速把故障特征频率换算成“阶次”，再在 ±2% 带内取峰。
          这一组是跨载荷泛化的关键：频率随转速缩放，阶次不随转速变化。

所有统计量按 (n_windows, window) 批量计算，无 Python 级循环（包络谱的频带取峰除外）。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .kinematics import BearingGeometry, fault_frequencies

TIME_FEATURES = [
    "td_mean", "td_std", "td_rms", "td_skew", "td_kurtosis", "td_peak",
    "td_p2p", "td_crest", "td_shape", "td_impulse", "td_clearance", "td_abs_mean",
]
FREQ_FEATURES = [
    "fd_centroid", "fd_spread", "fd_entropy", "fd_rolloff85",
    "fd_band_0_1k", "fd_band_1_2k", "fd_band_2_3k", "fd_band_3_4k", "fd_band_4_5k",
    "fd_hf_ratio",
]
ENV_ORDERS = ["BPFO", "BPFI", "BSF2", "FTF"]
ENV_HARMONICS = (1, 2, 3)


def _env_feature_names() -> list[str]:
    names = []
    for order in ENV_ORDERS:
        for h in ENV_HARMONICS:
            names.append(f"env_{order.lower()}_h{h}_snr")
            names.append(f"env_{order.lower()}_h{h}_rel")
    names.append("env_kurtosis")
    names.append("env_energy_ratio")
    return names


FEATURE_NAMES = TIME_FEATURES + FREQ_FEATURES + _env_feature_names()
FEATURE_GROUPS = {
    "time": TIME_FEATURES,
    "freq": FREQ_FEATURES,
    "env": _env_feature_names(),
}


def time_features(X: np.ndarray) -> np.ndarray:
    """(n, L) → (n, 12)"""
    x = np.asarray(X, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    std = xc.std(axis=1)
    rms = np.sqrt((xc ** 2).mean(axis=1))
    absx = np.abs(xc)
    mean_abs = absx.mean(axis=1)
    peak = absx.max(axis=1)
    p2p = x.max(axis=1) - x.min(axis=1)
    var = std ** 2 + 1e-18
    skew = (xc ** 3).mean(axis=1) / var ** 1.5
    kurt = (xc ** 4).mean(axis=1) / var ** 2
    eps = 1e-18
    crest = peak / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = peak / (mean_abs + eps)
    clearance = peak / (np.sqrt(absx).mean(axis=1) ** 2 + eps)
    out = np.column_stack([
        mu.ravel(), std, rms, skew, kurt, peak, p2p,
        crest, shape, impulse, clearance, mean_abs,
    ])
    return out


def _spectrum(X: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[1]
    amp = 2.0 * np.abs(np.fft.rfft(X, axis=1)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return amp, freqs


def freq_features(X: np.ndarray, fs: float) -> np.ndarray:
    """(n, L) → (n, 10)"""
    amp, freqs = _spectrum(X, fs)
    total = amp.sum(axis=1) + 1e-18
    centroid = (amp * freqs).sum(axis=1) / total
    spread = np.sqrt((amp * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
    p = amp / total[:, None]
    entropy = -(p * np.log(p + 1e-18)).sum(axis=1) / np.log(p.shape[1])
    cum = np.cumsum(amp, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]

    bands = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
    band_e = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        band_e.append(amp[:, m].sum(axis=1) / total * 1e3)  # 千分比，便于阅读
    hf = amp[:, freqs >= 3000].sum(axis=1) / total
    return np.column_stack([centroid, spread, entropy, rolloff, *band_e, hf])


def envelope_features(X: np.ndarray, fs: float, rpm: np.ndarray,
                      geom: BearingGeometry | None = None, tol: float = 0.02) -> np.ndarray:
    """包络阶次特征：把理论故障频率按实测转速换算，取带内峰值。

    带半宽 = max(tol·fc, Δf)，Δf = fs/N 为一个频率分辨单元。
    这样保证任何转速/窗长下带内至少落进一个 bin —— 否则当 2%·fc < Δf 时
    （如 N=1024、fs=12k、BPFI≈160 Hz 时 2% 仅 3.2 Hz，而 Δf=11.7 Hz）
    带内无 bin，特征会静默退化为 0。

    返回 (n, 26)：4 个阶次 × 3 次谐波 × (log1p 信噪比、占总包络能量比)
                 + 包络峭度 + 包络能量比
    """
    x = np.asarray(X, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    env = np.abs(hilbert(x, axis=1))
    env = env - env.mean(axis=1, keepdims=True)
    n = x.shape[1]
    E = 2.0 * np.abs(np.fft.rfft(env, axis=1)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    df = fs / n
    floor_mask = (freqs >= 500) & (freqs <= fs / 2 * 0.98)
    floor = np.median(E[:, floor_mask], axis=1) + 1e-12
    total = E.sum(axis=1) + 1e-12

    fr = np.asarray(rpm, dtype=np.float64) / 60.0
    geom = geom or BearingGeometry()
    mult = geom.multipliers()
    n_bins = E.shape[1]

    feats = []
    for order in ENV_ORDERS:
        base = mult[order] * fr
        for h in ENV_HARMONICS:
            fc = base * h
            half = np.maximum(tol * fc, df)
            lo = np.clip(np.floor((fc - half) / df).astype(int), 0, n_bins - 1)
            hi = np.clip(np.ceil((fc + half) / df).astype(int), 0, n_bins - 1)
            peak_amp = np.empty(len(x))
            # 同一文件内转速恒定 → (lo, hi) 只有少数几种组合，按组合批量取峰
            for lo_i, hi_i in {(int(a), int(b)) for a, b in zip(lo, hi)}:
                rows = np.where((lo == lo_i) & (hi == hi_i))[0]
                peak_amp[rows] = E[rows][:, lo_i:hi_i + 1].max(axis=1)
            feats.append(np.log1p(peak_amp / floor))
            feats.append(peak_amp / total)

    env_kurt = ((env - env.mean(axis=1, keepdims=True)) ** 4).mean(axis=1) / (
        env.std(axis=1) ** 4 + 1e-18)
    env_ratio = env.std(axis=1) / (x.std(axis=1) + 1e-18)
    feats.extend([env_kurt, env_ratio])
    return np.column_stack(feats)


def extract(X: np.ndarray, fs: float, rpm: np.ndarray,
            geom: BearingGeometry | None = None, groups: tuple[str, ...] = ("time", "freq", "env")) -> tuple[np.ndarray, list[str]]:
    """按组拼特征，返回 (矩阵, 列名)。"""
    parts, names = [], []
    if "time" in groups:
        parts.append(time_features(X)); names += TIME_FEATURES
    if "freq" in groups:
        parts.append(freq_features(X, fs)); names += FREQ_FEATURES
    if "env" in groups:
        parts.append(envelope_features(X, fs, rpm, geom)); names += _env_feature_names()
    if not parts:
        raise ValueError("至少选一组特征")
    return np.hstack(parts), names
