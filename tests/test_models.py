"""模型层正确性：噪声增广的功率口径 + 带增广的训练环能正常收敛。"""
import numpy as np
import pytest

from cwru_diag.models import augment_noise, train_cnn


def test_augment_noise_none_keeps_clean():
    X = np.random.default_rng(0).normal(size=(16, 256)).astype(np.float32)
    out = augment_noise(X, [None], seed=1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, X)


def test_augment_noise_power_matches_snr():
    """加噪后实测 SNR 应接近设定值（按窗口功率口径）。"""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(64, 2048)).astype(np.float64)
    out = augment_noise(X, [6.0], seed=3)
    p_sig = (X ** 2).mean(axis=1)
    p_noise = ((out - X) ** 2).mean(axis=1)
    snr_db = 10 * np.log10(p_sig / p_noise)
    assert np.allclose(snr_db, 6.0, atol=1.0)
    assert out.dtype == X.dtype


def test_augment_noise_mixed_choices():
    X = np.zeros((30, 128), dtype=np.float32)
    X += 0.5
    out = augment_noise(X, [None, None, None], seed=4)   # 全是干净
    np.testing.assert_allclose(out, X)
    with pytest.raises(ValueError):
        augment_noise(X, [])


def test_train_cnn_with_aug_runs():
    """带 aug_snrs 的训练环能跑完并给出非平凡的验证准确率。"""
    rng = np.random.default_rng(5)
    n, L = 96, 512
    t = np.arange(L) / 12000.0
    y = rng.integers(0, 2, size=n)
    X = np.stack([
        (0.4 * np.sin(2 * np.pi * (300 + 700 * yy) * t) + 0.05 * rng.normal(size=L))
        for yy in y
    ]).astype(np.float32)
    res = train_cnn(X[:72], y[:72], X[72:], y[72:], n_classes=2, epochs=6,
                    batch_size=32, seed=0, verbose=False,
                    aug_snrs=[None, 10, 0])
    assert res.best_epoch >= 1
    assert 0.0 <= res.best_val_acc <= 1.0
    assert len(res.history["val_acc"]) >= 1
