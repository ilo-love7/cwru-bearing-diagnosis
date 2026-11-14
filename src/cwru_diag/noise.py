"""噪声模型：白噪 / 有色噪（转频谐波 + 结构共振）/ 脉冲噪，按逐窗 SNR 精确归一。

报告 §5.3 明确把"只做了高斯白噪声"列为局限：真实现场噪声有色（转速谐波、
结构共振、电磁干扰），且 §3.9 的噪声增广里 SVM 用固定噪声副本而 CNN 逐窗重抽，
公平性不完全对等。本模块统一三类噪声生成与**逐窗随机**增广接口：

  white    加性高斯白噪声（与 08 脚本 add_noise 完全同口径，可回归对照）
  colored  频域塑形：低频 1/f^β 倾斜 + 转频谐波窄带峰 + 结构共振窄带峰
  impulse  稀疏双极性脉冲（电磁/开关干扰），密度可控

所有噪声都先按自身 RMS 归一，再按每窗信号功率换算到目标 SNR（dB），
因此三类噪声在同一 SNR 网格上的功率严格一致，只有**谱/时域形状**不同。
"""
from __future__ import annotations

import numpy as np

NOISE_KINDS = ("white", "colored", "impulse")


def _colored_noise(shape: tuple[int, int], seed: int, fs: float,
                   rpm: float | np.ndarray) -> np.ndarray:
    """频域塑形的有色噪声：rFFT 白噪 × 目标幅度谱 → irFFT。

    幅度谱 = 低频倾斜 1/(f+f0)^β + 转频谐波（1×/2×/3× f_r）窄带峰
           + 结构共振（1.2 kHz / 3.4 kHz）窄带峰。随机相位由白噪自带。
    """
    rng = np.random.default_rng(seed)
    n, l = shape
    w = rng.normal(size=shape)
    spec = np.fft.rfft(w, axis=1)
    freqs = np.fft.rfftfreq(l, d=1.0 / fs)
    # 低频倾斜
    amp = 1.0 / np.power(freqs + 60.0, 0.7)
    fr = float(np.mean(rpm)) / 60.0 if np.ndim(rpm) else float(rpm) / 60.0
    peaks = [(h * fr, 6.0, 8.0) for h in (1, 2, 3)] + [(1200.0, 8.0, 45.0),
                                                          (3400.0, 5.0, 70.0)]
    for center, gain, width in peaks:
        amp = amp + gain * np.exp(-0.5 * ((freqs - center) / width) ** 2)
    out = np.fft.irfft(spec * amp[None, :], n=l, axis=1)
    return out


def _impulse_noise(shape: tuple[int, int], seed: int,
                   density: float = 0.004) -> np.ndarray:
    """稀疏双极性脉冲 + 极小底噪（避免全零导致 RMS 归一除零）。"""
    rng = np.random.default_rng(seed)
    hit = rng.random(shape) < density
    out = rng.normal(size=shape) * hit * rng.choice([-1.0, 1.0], size=shape)
    out = out + 1e-3 * rng.normal(size=shape)
    return out


def make_noise(shape: tuple[int, int], kind: str, seed: int = 0,
               fs: float = 12000.0, rpm: float | np.ndarray = 1750.0,
               density: float = 0.004) -> np.ndarray:
    """生成未归一的原始噪声 (n, L)。"""
    if kind == "white":
        return np.random.default_rng(seed).normal(size=shape)
    if kind == "colored":
        return _colored_noise(shape, seed, fs, rpm)
    if kind == "impulse":
        return _impulse_noise(shape, seed, density)
    raise ValueError(f"未知噪声类型 {kind!r}，可用 {NOISE_KINDS}")


def add_noise_at_snr(x: np.ndarray, kind: str, snr_db: float | None,
                     seed: int = 0, fs: float = 12000.0,
                     rpm: float | np.ndarray = 1750.0,
                     density: float = 0.004) -> np.ndarray:
    """在逐窗自身功率的口径下叠加噪声，返回与 x 同 dtype 的数组。

    snr_db=None 表示保持干净。噪声先按每行 RMS 归一，再乘 sqrt(p_sig/10^(SNR/10))，
    故实际 SNR 与请求值一致（白噪情形与 08 脚本的 add_noise 逐点等价）。
    """
    xf = np.asarray(x, dtype=np.float64)
    if xf.ndim != 2:
        raise ValueError("x 须为 (n, L) 波形窗口数组")
    if snr_db is None:
        return np.asarray(x).copy()
    n = make_noise(xf.shape, kind, seed, fs, rpm, density)
    rms = np.sqrt(np.mean(n * n, axis=1, keepdims=True))
    rms = np.maximum(rms, 1e-12)
    p_sig = np.mean(xf * xf, axis=1, keepdims=True)
    p_noise = p_sig / (10.0 ** (float(snr_db) / 10.0))
    out = xf + (n / rms) * np.sqrt(p_noise)
    return out.astype(np.asarray(x).dtype if np.issubdtype(np.asarray(x).dtype,
                                                           np.floating) else np.float32)


def effective_snr_db(x: np.ndarray, x_noisy: np.ndarray) -> np.ndarray:
    """实测逐窗 SNR：10·log10(p_sig / p_noise)。用于单测与协议核对。"""
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(x_noisy, dtype=np.float64) - x
    p_sig = np.mean(x * x, axis=1)
    p_noise = np.maximum(np.mean(n * n, axis=1), 1e-30)
    return 10.0 * np.log10(p_sig / p_noise)


def mixed_augment(x: np.ndarray, kinds: tuple[str, ...],
                  snrs: tuple[float | None, ...], seed: int = 0,
                  fs: float = 12000.0, rpm: float | np.ndarray = 1750.0) -> np.ndarray:
    """逐窗独立随机抽 (噪声族, SNR) 的增广；None 表示该窗保持干净。

    用于给 SVM/RF 也做"每轮重抽"的公平增广：多次调用换 seed 即得多个增广副本，
    与 CNN 每个 epoch 重抽的口径一致。
    """
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    k_pick = rng.integers(0, len(kinds), size=len(x))
    s_pick = rng.integers(0, len(snrs), size=len(x))
    out = np.array(x, copy=True)
    for i, kind in enumerate(kinds):
        for j, snr in enumerate(snrs):
            sel = (k_pick == i) & (s_pick == j)
            if not sel.any() or snr is None:
                continue
            out[sel] = add_noise_at_snr(x[sel], kind, snr,
                                        seed=int(seed * 100003 + i * 97 + j),
                                        fs=fs, rpm=rpm)
    return out


def noise_shape_stats(x: np.ndarray, x_noisy: np.ndarray,
                      fs: float = 12000.0) -> dict:
    """噪声形状诊断：低频能量占比、峭度、脉冲因子（纯描述，不参与训练）。"""
    n = np.asarray(x_noisy, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(n, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n.shape[1], d=1.0 / fs)
    total = max(float(spec.sum()), 1e-30)
    low = float(spec[:, freqs < 500.0].sum()) / total
    flat = n.ravel()
    sd = max(float(flat.std()), 1e-30)
    kurt = float(np.mean(((flat - flat.mean()) / sd) ** 4))
    crest = float(np.max(np.abs(n), axis=1).mean() /
                  max(float(np.sqrt(np.mean(n * n))), 1e-30))
    return {"low_freq_energy_share": low, "kurtosis": kurt, "crest_factor": crest}
