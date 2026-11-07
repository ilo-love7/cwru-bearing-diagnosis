"""轴承故障特征频率（运动学）。

驱动端轴承为 SKF 6205-2RS JEM：9 个滚动体，球径 0.3126 in，节径 1.537 in，接触角 0。
公式（Randall & Antoni 2011, MSSP）：
    BPFO = (n/2)·fr·(1 − d/D·cosα)
    BPFI = (n/2)·fr·(1 + d/D·cosα)
    BSF  = (D/2d)·fr·(1 − (d/D·cosα)²)          ← 单个滚动体自转频率
    2·BSF                                        ← 滚动体缺陷冲击的重复频率（文献常用）
    FTF  = (fr/2)·(1 − d/D·cosα)                 ← 保持架频率
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    n_balls: int = 9
    ball_dia_in: float = 0.3126
    pitch_dia_in: float = 1.537
    contact_angle_deg: float = 0.0

    @property
    def ratio(self) -> float:
        """d/D·cosα"""
        import math

        return (self.ball_dia_in / self.pitch_dia_in) * math.cos(
            math.radians(self.contact_angle_deg)
        )

    def multipliers(self) -> dict[str, float]:
        """各故障特征频率相对转频 fr 的倍数（与转速无关，可直接用于阶次分析）。"""
        r = self.ratio
        n = self.n_balls
        bsf = (1.0 / (2.0 * r)) * (1.0 - r * r)
        return {
            "BPFO": n / 2.0 * (1.0 - r),
            "BPFI": n / 2.0 * (1.0 + r),
            "BSF": bsf,
            "BSF2": 2.0 * bsf,
            "FTF": 0.5 * (1.0 - r),
        }


def fault_frequencies(fr_hz: float, geom: BearingGeometry | None = None) -> dict[str, float]:
    """给定转频（Hz），返回各故障特征频率（Hz）。"""
    geom = geom or BearingGeometry()
    return {k: v * fr_hz for k, v in geom.multipliers().items()}


# 文献常用标称值（fr=1 时），用于测试与文档对照：
# BPFO 3.5848, BPFI 5.4152, BSF 2.3566, 2·BSF 4.7133, FTF 0.3983
