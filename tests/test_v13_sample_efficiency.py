"""v1.3 E34：剂量子集与文件分层学习曲线的抽样语义。"""
import numpy as np
import pytest

from cwru_diag.sample_efficiency import (dose_curve_summary, dose_subsets,
                                         marginal_gain, stratified_file_subsets)


def test_dose_subsets_sizes_and_membership():
    pool = np.arange(478)
    s = dose_subsets(pool, (0, 10, 116, 478), seed=0, n_repeats=4)
    for k, reps in s.items():
        assert len(reps) == 4, k
        for r in reps:
            assert len(r) == int(k)
            assert len(np.unique(r)) == len(r)          # 无放回
            assert np.all(np.isin(r, pool))
            assert np.all(np.diff(r) >= 0)              # 排序后稳定


def test_dose_zero_and_full_edges():
    pool = np.arange(50) + 100
    s = dose_subsets(pool, (0, 50), seed=1, n_repeats=2)
    assert all(len(r) == 0 for r in s["0"])
    for r in s["50"]:
        np.testing.assert_array_equal(r, pool)


def test_dose_prefix_mode_matches_v12_protocol():
    pool = np.arange(478)
    s = dose_subsets(pool, (116, 200), seed=0, prefix=True)
    np.testing.assert_array_equal(s["116"][0], pool[:116])
    np.testing.assert_array_equal(s["200"][0], pool[:200])
    assert all(len(reps) == 1 for reps in s.values())


def test_dose_reproducible_and_repeats_are_distinct():
    pool = np.arange(478)
    a = dose_subsets(pool, (116,), seed=3, n_repeats=3)["116"]
    b = dose_subsets(pool, (116,), seed=3, n_repeats=3)["116"]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], a[1])


def test_dose_out_of_range_raises():
    with pytest.raises(ValueError):
        dose_subsets(np.arange(10), (11,))
    with pytest.raises(ValueError):
        dose_subsets(np.arange(10), (-1,))


def _file_ids():
    return np.concatenate([np.full(100, 1), np.full(80, 2), np.full(60, 3)])


def test_stratified_subsets_respect_fraction_per_file():
    fids = _file_ids()
    s = stratified_file_subsets(fids, (0.1, 0.5, 1.0), seed=0, min_per_file=1)
    for frac, counts in (("0.1", {1: 10, 2: 8, 3: 6}),
                         ("0.5", {1: 50, 2: 40, 3: 30}),
                         ("1.0", {1: 100, 2: 80, 3: 60})):
        idx = s[frac]
        assert len(np.unique(idx)) == len(idx)
        for fid, n in counts.items():
            assert int(np.sum(fids[idx] == fid)) == n, (frac, fid)
    np.testing.assert_array_equal(s["1.0"], np.arange(len(fids)))


def test_stratified_min_per_file_and_reproducibility():
    fids = _file_ids()
    s = stratified_file_subsets(fids, (0.01,), seed=5, min_per_file=3)["0.01"]
    assert len(s) == 9 and all(int(np.sum(fids[s] == fid)) == 3 for fid in (1, 2, 3))
    t = stratified_file_subsets(fids, (0.01,), seed=5, min_per_file=3)["0.01"]
    np.testing.assert_array_equal(s, t)


def test_stratified_invalid_fraction_raises():
    with pytest.raises(ValueError):
        stratified_file_subsets(_file_ids(), (0.0,))
    with pytest.raises(ValueError):
        stratified_file_subsets(_file_ids(), (1.5,))


def test_dose_curve_summary_values():
    s = dose_curve_summary({"10": [0.6, 0.8], "50": [0.9, 0.9, 0.9]})
    assert s["10"]["mean"] == pytest.approx(0.7)
    assert s["10"]["std"] == pytest.approx(0.1)
    assert s["10"]["min"] == pytest.approx(0.6)
    assert s["10"]["max"] == pytest.approx(0.8)
    assert s["50"]["std"] == pytest.approx(0.0)
    assert s["50"]["n_repeats"] == 3


def test_marginal_gain_sorts_by_numeric_dose():
    g = marginal_gain({"2": 0.7, "10": 0.9, "100": 0.95})
    assert list(g) == ["10", "100"]
    assert g["10"] == pytest.approx(0.2)
    assert g["100"] == pytest.approx(0.05)
