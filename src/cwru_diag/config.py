"""配置读取：YAML → dict，路径相对仓库根解析，环境变量 CWRU_DATA_ROOT 可覆盖 data_root。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def _mini_yaml(text: str) -> dict[str, Any]:
    """只支持本项目用到的 YAML 子集：字典、列表、标量、一层嵌套。"""
    def cast(v: str) -> Any:
        v = v.strip()
        if v in ("null", "None", "~"):
            return None
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            return [cast(x) for x in inner.split(",")] if inner else []
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip():
            parent[key.strip()] = cast(val)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
    return root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    cfg = _mini_yaml(cfg_path.read_text(encoding="utf-8"))
    data_root = os.environ.get("CWRU_DATA_ROOT", cfg["data_root"])
    cfg["data_root"] = data_root
    cfg["raw_path"] = Path(data_root) / cfg.get("raw_dir", "raw")
    cfg["root"] = ROOT
    return cfg


if __name__ == "__main__":
    import json
    c = load_config()
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in c.items()},
                     ensure_ascii=False, indent=2))
