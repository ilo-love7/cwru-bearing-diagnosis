"""CWRU .mat 读取与元数据解析。

CWRU 每个文件含若干通道变量，命名形如 X105_DE_time / X105_FE_time / X105_BA_time，
以及转速变量 X105RPM（部分文件无下划线）。本模块只依赖 scipy.io.loadmat。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_DE_RE = re.compile(r"^X\d+_DE_time$")
_ANY_RE = re.compile(r"^X\d+_(DE|FE|BA)_time$")
_RPM_RE = re.compile(r"^X\d+_?RPM$")


@dataclass
class Record:
    file_id: int
    path: Path
    de: np.ndarray          # 驱动端加速度，g
    rpm: float              # 实测转速（无转速变量时为 nan）
    fs: float
    keys: tuple[str, ...]

    @property
    def fr(self) -> float:
        return self.rpm / 60.0

    @property
    def duration_s(self) -> float:
        return len(self.de) / self.fs


def _as_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).ravel()


def load_record(path: str | Path, fs: float = 12000.0) -> Record:
    """读取一个 CWRU .mat，返回驱动端通道与转速。"""
    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = tuple(k for k in mat.keys() if not k.startswith("__"))

    de_key = next((k for k in keys if _DE_RE.match(k)), None)
    if de_key is None:  # 兜底：取第一个 *_DE_time / 任意 _time
        de_key = next((k for k in keys if k.endswith("_DE_time")), None)
    if de_key is None:
        any_key = next((k for k in keys if _ANY_RE.match(k)), None)
        if any_key is None:
            raise KeyError(f"{path.name} 未找到振动通道，keys={keys}")
        de_key = any_key

    de = _as_1d(mat[de_key])
    de = de[np.isfinite(de)]

    rpm_key = next((k for k in keys if _RPM_RE.match(k)), None)
    rpm = float(np.ravel(mat[rpm_key])[0]) if rpm_key is not None else float("nan")
    if not np.isfinite(rpm):
        rpm = float("nan")

    file_id = int(re.match(r"^(\d+)", path.stem).group(1))
    return Record(file_id=file_id, path=path, de=de, rpm=rpm, fs=fs, keys=keys)
