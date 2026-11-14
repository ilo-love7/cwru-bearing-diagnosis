#!/usr/bin/env bash
# 一键复现全部指标与图。数据不全时会先下载。
# 完整链 00–18 全部串起来；v1.1 深化 09–11、v1.2 深化 12–14、v1.3 深化 15–17 可单独跑。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}

echo "[0/11] 下载数据（已存在的文件会跳过）"
$PY scripts/00_download_data.py --workers 10

echo "[1/11] 数据体检"
$PY scripts/01_eda.py

echo "[2/11] 经典模型基线"
$PY scripts/02_train_classical.py

echo "[3/11] CNN 对照"
$PY scripts/03_train_cnn.py

echo "[4/11] 跨载荷泛化"
$PY scripts/04_cross_load.py --cnn-seeds 1 --cnn-epochs 20

echo "[5/11] 消融 / 未见损伤 / 位置泛化"
$PY scripts/05_ablation_position_severity.py

echo "[6/11] 类型×程度 10 分类"
$PY scripts/07_severity_10class.py

echo "[7/11] 加噪鲁棒性（零适配）"
$PY scripts/08_noise_robustness.py

echo "[8/11] 深化：CNN 分布外口径 + 噪声增广 + 位置 few-shot"
$PY scripts/09_cnn_ood.py
$PY scripts/10_noise_aug.py
$PY scripts/11_position_fewshot.py

echo "[9/11] v1.2 深化：217.mat 双口径敏感性"
$PY scripts/12_sensitivity_217.py

echo "[10/11] v1.2 深化：跨载荷迁移矩阵 + 载荷探针"
$PY scripts/13_transfer_matrix.py

echo "[11/14] v1.2 深化：10 分类序数口径 + 消融×协议"
$PY scripts/14_10class_ordinal.py

echo "[12/14] v1.3 深化：特征级域适应（CORAL / KMM / few-shot）"
$PY scripts/15_domain_adaptation.py

echo "[13/14] v1.3 深化：噪声族与增广公平性（白/有色/脉冲）"
$PY scripts/16_noise_families.py

echo "[14/14] v1.3 深化：217 剂量曲线 + 样本效率学习曲线"
$PY scripts/17_dose_learning_curve.py

echo "汇总图表"
$PY scripts/06_make_figures.py
$PY scripts/18_summary_v13.py

echo
echo "完成。指标见 results/metrics/*.json，图见 results/figures/*.png"
