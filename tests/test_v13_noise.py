"""v1.3 E33：三类噪声的 SNR 精确性、形状差异与逐窗增广语义。"""
import numpy as np
import pytest

from cwru_diag.noise import (NOISE_KINDS, add_noise_at_snr, effective_snr_db,
                             make_noise, mixed_augment, noise_shape_stats)


def _signal(n=48, l=2048, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(l)[None, :] / 12000.0
    return (np.sin(2 * np.pi * 530 * t + rng.uniform(0, 6.28, (n, 1)))
            + 0.4 * rng.normal(size=(n, l))).astype(np.float32)


@pytest.mark.parametrize("kind", NOISE_KINDS)
@pytest.mark.parametrize("snr", [10.0, 2.0, 0.0, -4.0])
def test_requested_snr_is_exact_for_every_kind(kind, snr):
    x = _signal()
    xn = add_noise_at_snr(x, kind, snr, seed=11, rpm=1750.0)
    got = effective_snr_db(x, xn)
    assert np.all(np.isfinite(got))
    # 归一到逐窗自身功率，误差只来自 float32 存储
    assert np.abs(got - snr).max() < 0.02


def test_none_snr_returns_untouched_copy():
    x = _signal()
    out = add_noise_at_snr(x, "white", None, seed=3)
    np.testing.assert_array_equal(out, x)
    assert out is not x
    assert out.dtype == x.dtype


def test_dtype_and_shape_preserved():
    x = _signal()
    out = add_noise_at_snr(x, "colored", 6.0, seed=1)
    assert out.shape == x.shape and out.dtype == x.dtype


def test_same_seed_reproducible_different_seed_changes():
    x = _signal(n=8)
    a = add_noise_at_snr(x, "white", 2.0, seed=7)
    b = add_noise_at_snr(x, "white", 2.0, seed=7)
    c = add_noise_at_snr(x, "white", 2.0, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_colored_noise_has_more_low_frequency_energy_than_white():
    x = _signal()
    sw = noise_shape_stats(x, add_noise_at_snr(x, "white", 0.0, seed=2))
    sc = noise_shape_stats(x, add_noise_at_snr(x, "colored", 0.0, seed=2, rpm=1750.0))
    assert sc["low_freq_energy_share"] > sw["low_freq_energy_share"] + 0.05


def test_impulse_noise_is_spiky():
    x = _signal()
    sw = noise_shape_stats(x, add_noise_at_snr(x, "white", 0.0, seed=2))
    si = noise_shape_stats(x, add_noise_at_snr(x, "impulse", 0.0, seed=2))
    assert si["kurtosis"] > sw["kurtosis"] * 5.0
    assert si["crest_factor"] > sw["crest_factor"] * 1.5


def test_all_kinds_have_same_power_at_same_snr():
    x = _signal()
    pows = []
    for kind in NOISE_KINDS:
        n = add_noise_at_snr(x, kind, 3.0, seed=5) - x
        pows.append(float(np.mean(n.astype(np.float64) ** 2)))
    assert max(pows) / min(pows) < 1.01  # float32 量化误差以内


def test_make_noise_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_noise((4, 128), "pink")
    with pytest.raises(ValueError):
        add_noise_at_snr(np.zeros((4, 128)), "pink", 0.0)


def test_mixed_augment_draws_per_window_and_keeps_clean_windows():
    x = _signal(n=120)
    out = mixed_augment(x, NOISE_KINDS, (None, 10.0, 0.0, -4.0), seed=4)
    assert out.shape == x.shape and out.dtype == x.dtype
    unchanged = np.all(out == x, axis=1)
    assert 0 < unchanged.sum() < len(x)      # 部分窗干净、部分窗带噪
    # 带噪窗的实际 SNR 落在请求档次附近（不精确，因为同窗只加一种）
    snr = effective_snr_db(x[~unchanged], out[~unchanged])
    assert np.all(snr > -4.05) and np.all(snr < 10.05)


def test_mixed_augment_reproducible():
    x = _signal(n=24)
    a = mixed_augment(x, ("white", "colored"), (2.0, -2.0), seed=9)
    b = mixed_augment(x, ("white", "colored"), (2.0, -2.0), seed=9)
    np.testing.assert_array_equal(a, b)


def test_mixed_augment_no_none_keeps_no_window_clean():
    x = _signal(n=64)
    out = mixed_augment(x, ("white", "impulse"), (6.0, 0.0), seed=1)
    assert np.all(np.any(out != x, axis=1))


def test_train_cnn_aug_fn_is_called_each_epoch_and_backward_compatible():
    """aug_fn 注入路径：每 epoch 调用一次、形状校验；不传时行为不变。"""
    import torch

    from cwru_diag.models import BearingCNN, predict_cnn, train_cnn

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 256)).astype(np.float32)
    y = np.array([0, 1, 2, 3] * 8)
    calls = []

    def fn(xtr, epoch):
        calls.append(epoch)
        return mixed_augment(xtr, ("white", "impulse"), (10.0, 0.0), seed=epoch)

    res = train_cnn(x, y, x[:8], y[:8], n_classes=4, epochs=2, batch_size=16,
                    lr=1e-3, seed=0, patience=5, verbose=False, aug_fn=fn)
    assert calls == [1, 2]
    assert isinstance(res.model, BearingCNN)
    assert predict_cnn(res.model, x[:4], batch_size=4).shape == (4,)
    assert torch.isfinite(torch.tensor(res.history["val_acc"])).all()

    # 未提供 aug_fn 时仍可正常训练（与 v1.1/v1.2 调用方式兼容）
    res2 = train_cnn(x, y, x[:8], y[:8], n_classes=4, epochs=1, batch_size=16,
                     seed=0, patience=5, verbose=False)
    assert len(res2.history["train_loss"]) == 1
