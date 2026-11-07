"""00 · 下载 CWRU 轴承数据（官方直链，无需登录）。

用法:
    python scripts/00_download_data.py [--roles main,position,severity] [--force]

数据落到 configs/config.yaml 的 data_root/raw/，文件名沿用官方编号（如 105.mat）。
每个文件下载后校验 MATLAB 5 头，并记录字节数与 sha256 到 results/metrics/download_log.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwru_diag.config import load_config  # noqa: E402

MAT_HEADER = b"MATLAB 5.0 MAT-file"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dst: Path, retries: int = 4, timeout: int = 90) -> int:
    """带重试的下载；返回字节数。先写 .part 再原子替换，避免半文件。"""
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cwru-diag/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None
                done = 0
                with tmp.open("wb") as fh:
                    while True:
                        block = resp.read(1 << 16)
                        if not block:
                            break
                        fh.write(block)
                        done += len(block)
            if total is not None and done != total:
                raise IOError(f"字节数不符: 期望 {total}, 实际 {done}")
            tmp.replace(dst)
            return done
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    重试 {attempt}/{retries - 1}（{exc}），{wait}s 后继续")
                time.sleep(wait)
    raise RuntimeError(f"下载失败: {url} ({last_err})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="main,position,severity",
                    help="要下载的子集，逗号分隔：main/position/severity")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    ap.add_argument("--workers", type=int, default=8, help="并发下载线程数")
    args = ap.parse_args()

    cfg = load_config()
    raw = Path(cfg["raw_path"])
    raw.mkdir(parents=True, exist_ok=True)

    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    manifest = pd.read_csv(ROOT / cfg["manifest"])
    subset = manifest[manifest["role"].isin(roles)].reset_index(drop=True)

    print(f"目标目录: {raw}")
    print(f"待下载 {len(subset)} 个文件（role={sorted(roles)}，并发 {args.workers}）")

    t0 = time.time()
    todo = []
    n_skip = 0
    for _, row in subset.iterrows():
        dst = raw / row["mat_file"]
        if dst.exists() and dst.stat().st_size > 0 and not args.force:
            n_skip += 1
        else:
            todo.append(row)
    if n_skip:
        print(f"已存在跳过 {n_skip} 个")

    done = {"n": 0}
    lock = __import__("threading").Lock()

    def work(row) -> None:
        name = row["mat_file"]
        url = f"{cfg['base_url']}/{name}"
        try:
            size = download(url, raw / name, timeout=300)
        except Exception as exc:  # noqa: BLE001
            with lock:
                print(f"  ✗ {name}: {exc}")
            raise
        with lock:
            done["n"] += 1
            print(f"  [{done['n']:02d}/{len(todo)}] {name:>11s} {size / 1e6:5.1f} MB  "
                  f"{row['label']:<6s} sev={row['severity_inch']}\" load={row['load_hp']}HP")

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, row) for row in todo]
            for f in as_completed(futures):
                f.result()

    log = []
    for _, row in subset.iterrows():
        dst = raw / row["mat_file"]
        head = dst.open("rb").read(len(MAT_HEADER))
        log.append({
            "file_id": int(row["file_id"]),
            "mat_file": row["mat_file"],
            "bytes": dst.stat().st_size,
            "sha256": sha256_of(dst),
            "matlab5_header": bool(head.startswith(MAT_HEADER)),
        })

    out = ROOT / "results" / "metrics" / "download_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_files": len(log),
        "n_new": len(todo),
        "n_skipped": n_skip,
        "elapsed_s": round(time.time() - t0, 1),
        "all_matlab5": all(x["matlab5_header"] for x in log),
        "files": log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    total_mb = sum(x["bytes"] for x in log) / 1e6
    print(f"\n完成：新增 {len(todo)}，跳过 {n_skip}，合计 {len(log)} 文件 / {total_mb:.0f} MB"
          f"，用时 {time.time() - t0:.0f}s")
    print(f"校验日志: {out}")
    return 0 if all(x["matlab5_header"] for x in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
