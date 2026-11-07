"""CWRU 轴承故障诊断 · 可复用模块。"""
from .config import load_config
from .kinematics import BearingGeometry, fault_frequencies
from .segmentation import WindowSet, segment_record

__all__ = [
    "load_config",
    "BearingGeometry",
    "fault_frequencies",
    "WindowSet",
    "segment_record",
]
__version__ = "1.0.0"
