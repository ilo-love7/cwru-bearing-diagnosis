#!/usr/bin/env bash
# 一键复现全部指标与图。数据不全时会先下载。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}

echo "[0/6] 下载数据（已存在的文件会跳过）"
$PY scripts/00_download_data.py --workers 10

echo "[1/6] 数据体检"
$PY scripts/01_eda.py

echo "[2/6] 经典模型基线"
$PY scripts/02_train_classical.py

echo "[3/6] CNN 对照"
$PY scripts/03_train_cnn.py

echo "[4/6] 跨载荷泛化"
$PY scripts/04_cross_load.py --cnn-seeds 1 --cnn-epochs 20

echo "[5/6] 消融 / 未见损伤 / 位置泛化"
$PY scripts/05_ablation_position_severity.py

echo "[6/6] 汇总图表"
$PY scripts/06_make_figures.py

echo
echo "完成。指标见 results/metrics/*.json，图见 results/figures/*.png"
