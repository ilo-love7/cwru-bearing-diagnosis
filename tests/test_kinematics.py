"""轴承运动学：与文献标称倍数对照。"""
import numpy as np
import pytest

from cwru_diag.kinematics import BearingGeometry, fault_frequencies


def test_multipliers_match_literature():
    """SKF 6205-2RS：BPFO 3.5848、BPFI 5.4152、BSF 2.3566、2BSF 4.7133、FTF 0.3983。"""
    m = BearingGeometry().multipliers()
    assert m["BPFO"] == pytest.approx(3.5848, abs=5e-4)
    assert m["BPFI"] == pytest.approx(5.4152, abs=5e-4)
    assert m["BSF"] == pytest.approx(2.3566, abs=5e-4)
    assert m["BSF2"] == pytest.approx(2 * m["BSF"], rel=1e-12)
    assert m["BSF2"] == pytest.approx(4.7133, abs=5e-4)
    assert m["FTF"] == pytest.approx(0.3983, abs=5e-4)


def test_sum_relation():
    """内外圈特征频率之和恰为 n_balls × fr —— 运动学恒等式。"""
    m = BearingGeometry().multipliers()
    assert m["BPFO"] + m["BPFI"] == pytest.approx(9.0, abs=1e-9)


def test_scales_with_shaft_speed():
    fr = 1797 / 60
    f = fault_frequencies(fr)
    assert f["BPFO"] == pytest.approx(3.5848 * fr, rel=1e-4)
    assert f["BPFI"] == pytest.approx(5.4152 * fr, rel=1e-4)


def test_contact_angle_effect():
    """接触角增大 → d/D·cosα 减小 → BPFO 变大、BPFI 变小。"""
    flat = BearingGeometry().multipliers()
    angled = BearingGeometry(contact_angle_deg=15).multipliers()
    assert angled["BPFO"] > flat["BPFO"]
    assert angled["BPFI"] < flat["BPFI"]
