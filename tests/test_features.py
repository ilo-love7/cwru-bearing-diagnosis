"""特征正确性：用解析信号验证时域/频域统计，用合成轴承冲击验证包络阶次特征。"""
import numpy as np
import pytest
from scipy.signal import hilbert

from cwru_diag.features import (FEATURE_GROUPS, FEATURE_NAMES, envelope_features,
                                extract, freq_features, time_features)
from cwru_diag.kinematics import BearingGeometry


def test_time_features_on_sine():
    """正弦：RMS = A/√2，峭度 ≈ 1.5，峰值因子 ≈ √2，偏度 ≈ 0。"""
    n, fs, f, A = 4096, 12000.0, 500.0, 1.0
    t = np.arange(n) / fs
    X = (A * np.sin(2 * np.pi * f * t))[None, :]
    f_names = FEATURE_NAMES[:12]
    v = dict(zip(f_names, time_features(X)[0]))
    assert v["td_rms"] == pytest.approx(A / np.sqrt(2), rel=1e-3)
    assert v["td_kurtosis"] == pytest.approx(1.5, rel=2e-2)
    assert v["td_crest"] == pytest.approx(np.sqrt(2), rel=2e-2)
    assert abs(v["td_skew"]) < 1e-2
    assert v["td_p2p"] == pytest.approx(2 * A, rel=1e-3)


def test_kurtosis_gaussian_is_three():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 2048))
    k = time_features(X)[:, FEATURE_NAMES.index("td_kurtosis")]
    assert k.mean() == pytest.approx(3.0, abs=0.06)


def test_impulsive_signal_has_higher_kurtosis():
    rng = np.random.default_rng(1)
    base = rng.normal(size=2048)
    impulse = base.copy()
    impulse[::256] += 12.0
    X = np.vstack([base, impulse])
    td = time_features(X)
    i_kurt = FEATURE_NAMES.index("td_kurtosis")
    assert td[1, i_kurt] > td[0, i_kurt]


def test_spectral_centroid_of_tone():
    n, fs, f = 4096, 12000.0, 1500.0
    t = np.arange(n) / fs
    X = np.sin(2 * np.pi * f * t)[None, :]
    fd = freq_features(X, fs)[0]
    centroid = fd[FEATURE_NAMES.index("fd_centroid") - 12]  # 相对 freq 组的偏移
    assert centroid == pytest.approx(f, abs=5.0)


def _bearing_like_signal(fs=12000, seconds=1.0, fr=1797 / 60, bpfo_mult=3.5848,
                         resonance=3000.0, snr_db=6.0, seed=0):
    """合成外圈故障：周期冲击 → 激励结构共振 → 加噪声。"""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    bpfo = bpfo_mult * fr
    period = 1.0 / bpfo
    sig = np.zeros(n)
    k = 0
    while k * period < seconds:
        i = int(k * period * fs)
        m = min(n - i, int(0.02 * fs))
        tt = np.arange(m) / fs
        sig[i:i + m] += np.exp(-900 * tt) * np.sin(2 * np.pi * resonance * tt)
        k += 1
    sig += rng.normal(scale=np.std(sig) / (10 ** (snr_db / 20)), size=n)
    return sig


def test_envelope_peaks_at_bpfo():
    """外圈合成信号：BPFO 处的包络信噪比应显著高于 BPFI，且 > 无故障平坦信号。"""
    fs = 12000.0
    rpm = 1797.0
    x = _bearing_like_signal(fs=fs, fr=rpm / 60)
    flat = np.random.default_rng(7).normal(size=len(x))

    X = np.vstack([x, flat])
    env = envelope_features(X, fs, np.array([rpm, rpm]), BearingGeometry())

    env_names = FEATURE_GROUPS["env"]
    i_bpfo = env_names.index("env_bpfo_h1_snr")
    i_bpfi = env_names.index("env_bpfi_h1_snr")
    assert env[0, i_bpfo] > env[0, i_bpfi] + 0.5
    assert env[0, i_bpfo] > env[1, i_bpfo] + 0.5


def test_envelope_harmonics_present():
    """冲击串的包络谱在 1×/2× BPFO 都应出现峰。"""
    fs = 12000.0
    rpm = 1797.0
    x = _bearing_like_signal(fs=fs, fr=rpm / 60)[None, :]
    env = envelope_features(x, fs, np.array([rpm]))
    env_names = FEATURE_GROUPS["env"]
    h1 = env[0, env_names.index("env_bpfo_h1_snr")]
    h2 = env[0, env_names.index("env_bpfo_h2_snr")]
    h3 = env[0, env_names.index("env_bpfo_h3_snr")]
    assert h1 > 1.0 and h2 > 1.0, "基频与二次谐波都应可检出"
    assert max(h1, h2, h3) > 1.0


def test_extract_groups_and_names():
    fs = 12000.0
    x = _bearing_like_signal(fs=fs)[None, :]
    M, names = extract(x, fs, np.array([1797.0]), groups=("time", "freq", "env"))
    assert M.shape == (1, len(names))
    assert len(names) == len(set(names)), "特征名不能重复"
    M2, names2 = extract(x, fs, np.array([1797.0]), groups=("time",))
    assert len(names2) == 12 and M2.shape == (1, 12)
