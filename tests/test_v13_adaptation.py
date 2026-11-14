"""v1.3 E32：特征级域适应的纯函数语义（含数值稳定性与退化情形）。"""
import numpy as np
import pytest

from cwru_diag.adaptation import (CoralTransform, class_conditional_drift,
                                  coral_distance, cov_match_error,
                                  fisher_ratio, fit_coral, kernel_mean_weights,
                                  median_bandwidth, mmd_rbf)


def _gauss(n, mu, cov, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n)


def test_coral_distance_self_zero_and_shift_positive():
    xs = _gauss(600, np.zeros(3), np.eye(3), 0)
    assert coral_distance(xs, xs.copy()) == pytest.approx(0.0, abs=1e-12)
    xt = xs * np.array([3.0, 1.0, 0.5])
    assert coral_distance(xs, xt) > 0.0


def test_fit_coral_exactly_matches_target_covariance():
    cs = np.array([[4.0, 1.2, 0.0], [1.2, 2.0, 0.3], [0.0, 0.3, 1.0]])
    ct = np.array([[1.0, -0.4, 0.2], [-0.4, 3.0, 0.1], [0.2, 0.1, 0.6]])
    xs = _gauss(2000, np.array([2.0, -1.0, 0.5]), cs, 1)
    xt = _gauss(2000, np.array([-3.0, 4.0, 1.0]), ct, 2)
    d0 = coral_distance(xs, xt)
    tr = fit_coral(xs, xt)
    zs = tr(xs)
    assert cov_match_error(zs, xt) < 1e-6
    assert coral_distance(zs, xt) < d0 * 0.01
    # 均值也对齐到目标域
    assert np.abs(zs.mean(axis=0) - xt.mean(axis=0)).max() < 0.15


def test_shrink_one_is_exact_identity():
    rng = np.random.default_rng(3)
    xs, xt = rng.normal(size=(400, 4)), rng.normal(size=(400, 4)) * 2.0 + 5.0
    tr = fit_coral(xs, xt, shrink=1.0)
    assert isinstance(tr, CoralTransform)
    np.testing.assert_allclose(tr.W, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(tr.b, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(tr(xs), xs, atol=1e-10)


def test_shrink_is_sample_level_convex_combination():
    rng = np.random.default_rng(31)
    xs = rng.normal(size=(400, 3)) @ np.diag([3.0, 1.0, 0.5])
    xt = rng.normal(size=(400, 3)) @ np.diag([0.5, 2.0, 1.5]) + 4.0
    full = fit_coral(xs, xt, shrink=0.0)(xs)
    for s in (0.25, 0.5, 0.75):
        got = fit_coral(xs, xt, shrink=s)(xs)
        np.testing.assert_allclose(got, (1 - s) * full + s * xs, atol=1e-8)


def test_partial_shrink_is_interpolated():
    rng = np.random.default_rng(4)
    xs = rng.normal(size=(500, 3)) @ np.diag([3.0, 1.0, 0.5])
    xt = rng.normal(size=(500, 3)) @ np.diag([0.5, 2.0, 1.5])
    w0 = fit_coral(xs, xt, shrink=0.0).W
    w1 = fit_coral(xs, xt, shrink=1.0).W
    wh = fit_coral(xs, xt, shrink=0.5).W
    np.testing.assert_allclose(wh, 0.5 * w0 + 0.5 * w1, atol=1e-12)


def test_ill_scaled_features_do_not_blow_up_covariance():
    """真实特征量纲跨数量级时，未标准化的 CORAL 会把协方差放大 1e2 倍以上。"""
    rng = np.random.default_rng(21)
    scales = np.array([1e3, 1.0, 1e-3, 10.0])
    base_s = rng.normal(size=(1200, 4))
    base_t = rng.normal(size=(1200, 4)) @ np.array(
        [[1.0, 0.4, 0.0, 0.0], [0.4, 1.8, 0.0, 0.0],
         [0.0, 0.0, 0.5, 0.2], [0.0, 0.0, 0.2, 2.2]])
    xs = base_s * scales + np.array([10.0, 2.0, 0.0, -5.0])
    xt = base_t * scales + np.array([-8.0, 4.0, 1.0, 7.0])
    d0 = coral_distance(xs, xt)
    tr = fit_coral(xs, xt)
    zs = tr(xs)
    assert np.all(np.isfinite(zs))
    # 归一化到各维标准差后比较，避免大尺度特征主导 Frobenius 差
    s = np.std(np.vstack([xs, xt]), axis=0)
    err = coral_distance(zs / s, xt / s)
    assert err < 1e-4, f"对齐后归一化协方差差仍为 {err}"
    assert coral_distance(xs / s, xt / s) > err * 10.0
    # 变换不能把样本整体放大（旧实现的典型症状）
    assert np.abs(zs / s).mean() < 20.0 * np.abs(xs / s).mean()
    assert d0 > 0.0


def test_singular_covariance_is_regularized_not_nan():
    xs = np.tile(np.array([[1.0, 2.0, 3.0]]), (50, 1))
    xt = np.tile(np.array([[2.0, 0.0, 5.0]]), (50, 1))
    tr = fit_coral(xs, xt, eps=1e-3)
    assert np.all(np.isfinite(tr.W))
    assert np.all(np.isfinite(tr(xs)))


def test_mmd_self_zero_positive_and_nonnegative():
    xs = _gauss(300, np.zeros(3), np.eye(3), 5)
    xt = _gauss(300, np.array([2.0, 0.0, 0.0]), np.eye(3), 6)
    assert mmd_rbf(xs, xs.copy()) < 1e-8
    assert mmd_rbf(xs, xt) > 0.0
    assert median_bandwidth(xs, xt) > 0.0
    assert mmd_rbf(xs[:1], xt) == 0.0  # 样本不足时安全返回 0


def test_kernel_mean_weights_degenerate_to_one_when_domains_equal():
    xs = _gauss(120, np.zeros(3), np.eye(3), 7)
    w = kernel_mean_weights(xs, xs.copy(), lam=1e-3)
    assert len(w) == len(xs)
    assert w.mean() == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w, np.ones(len(xs)), atol=0.15)


def test_kernel_mean_weights_shifted_domain_stays_bounded_and_normalized():
    xs = _gauss(120, np.zeros(3), np.eye(3), 8)
    xt = _gauss(120, np.array([1.5, 0.0, 0.0]), np.eye(3), 9)
    w = kernel_mean_weights(xs, xt, clip=(0.2, 3.0))
    assert w.min() >= 0.2 - 1e-12 and w.max() <= 3.0 + 1e-12
    assert w.mean() == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(w, np.ones(len(xs)))  # 确实起了加权作用


def test_kmm_does_not_collapse_to_one_on_ill_scaled_features():
    """量纲病态时核不能退化：否则权重全 1，KMM 看似无作用实则根本没跑起来。"""
    rng = np.random.default_rng(25)
    scale = np.array([1e3, 1.0, 1e-3])
    xs = rng.normal(size=(150, 3)) * scale
    xt = rng.normal(size=(150, 3)) * scale + np.array([500.0, 0.0, 0.0])
    w = kernel_mean_weights(xs, xt)
    assert w.std() > 0.05
    assert not np.allclose(w, np.ones(len(xs)), atol=0.1)
    assert w.mean() == pytest.approx(1.0, abs=1e-6)


def test_fisher_ratio_separated_beats_mixed():
    rng = np.random.default_rng(10)
    x_sep = np.vstack([rng.normal(-4.0, 0.2, (100, 2)), rng.normal(4.0, 0.2, (100, 2))])
    x_mix = np.vstack([rng.normal(0.0, 2.0, (100, 2)), rng.normal(0.1, 2.0, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    assert fisher_ratio(x_sep, y) > 10.0 * fisher_ratio(x_mix, y)
    assert fisher_ratio(x_sep, np.zeros(len(x_sep), int)) == 0.0


def test_class_conditional_drift_decreases_after_manual_alignment():
    xs = np.array([[0.0, 0.0], [0.2, 0.0], [4.0, 0.0], [4.2, 0.0]])
    xt = xs + np.array([2.0, 1.0])
    ys = yt = np.array([0, 0, 1, 1])
    before = class_conditional_drift(xs, ys, xt, yt, scale=np.ones(2))
    after = class_conditional_drift(xs + np.array([2.0, 1.0]), ys, xt, yt,
                                    scale=np.ones(2))
    assert after["mean"] < before["mean"]
    assert after["mean"] == pytest.approx(0.0, abs=1e-12)
    assert set(after["per_class"]) == {"0", "1"}


def test_input_validation():
    a = np.zeros((10, 3))
    with pytest.raises(ValueError):
        coral_distance(a, np.zeros((10, 2)))
    with pytest.raises(ValueError):
        mmd_rbf(a, np.zeros((10, 2)))
    with pytest.raises(ValueError):
        fit_coral(a, a, shrink=1.5)
    with pytest.raises(ValueError):
        fit_coral(a, a, eps=0.0)
    with pytest.raises(ValueError):
        kernel_mean_weights(a, np.zeros((10, 2)))
    with pytest.raises(ValueError):
        fisher_ratio(a, np.zeros(5, int))
