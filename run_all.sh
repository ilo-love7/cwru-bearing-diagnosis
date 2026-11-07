#!/usr/bin/env bash
# 一键复现全部指标与图。数据不全时会先下载。
# 完整链 00–11 全部串起来；只要同工况基线可跑 00–06，深化实验 07–11 可单独跑。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}

echo "[0/9] 下载数据（已存在的文件会跳过）"
$PY scripts/00_download_data.py --workers 10

echo "[1/9] 数据体检"
$PY scripts/01_eda.py

echo "[2/9] 经典模型基线"
$PY scripts/02_train_classical.py

echo "[3/9] CNN 对照"
$PY scripts/03_train_cnn.py

echo "[4/9] 跨载荷泛化"
$PY scripts/04_cross_load.py --cnn-seeds 1 --cnn-epochs 20

echo "[5/9] 消融 / 未见损伤 / 位置泛化"
$PY scripts/05_ablation_position_severity.py

echo "[6/9] 类型×程度 10 分类"
$PY scripts/07_severity_10class.py

echo "[7/9] 加噪鲁棒性（零适配）"
$PY scripts/08_noise_robustness.py

echo "[8/9] 深化：CNN 分布外口径 + 噪声增广 + 位置 few-shot"
$PY scripts/09_cnn_ood.py
$PY scripts/10_noise_aug.py
$PY scripts/11_position_fewshot.py

echo "[9/9] 汇总图表"
$PY scripts/06_make_figures.py

echo
echo "完成。指标见 results/metrics/*.json，图见 results/figures/*.png"
