"""18 · v1.3 汇总：把 E32/E33/E34 的结果 JSON 转成报告可直接粘贴的 Markdown 表。

输入  results/metrics/domain_adaptation.json
      results/metrics/noise_families.json
      results/metrics/dose_learning_curve.json
输出  results/v13_summary_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"


def load(name: str) -> dict:
    p = M / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f(v, n=3) -> str:
    return "—" if v is None else f"{float(v):.{n}f}"


def emit(lines: list[str]) -> None:
    out = ROOT / "results" / "v13_summary_tables.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"完成 → {out.relative_to(ROOT)}  ({len(lines)} 行)")


def main() -> int:
    da = load("domain_adaptation.json")
    nf = load("noise_families.json")
    dl = load("dose_learning_curve.json")
    L: list[str] = ["# N1 v1.3 汇总表（由 `scripts/18_summary_v13.py` 自动生成）", ""]

    # ---------------- 表 A：E32 方法对比 ----------------
    L += ["## 表 A · E32 域适应方法对比（SVM 全特征，5 seed 均值）", ""]
    if da:
        L += ["| 有向对 | 口径 | baseline | CORAL | CORAL½ | KMM | few-shot k=5 | 全目标域参考 |",
              "|---|---|---|---|---|---|---|---|"]
        for pname, row in da["pairs"].items():
            s, d = pname.split("_to_")
            for proto in ("with_217", "without_217"):
                m = row["protocols"][proto]["methods"]
                L.append(f"| {s}→{d} HP | {'含 217' if proto == 'with_217' else '剔 217'} | "
                         f"{f(m['baseline']['accuracy'],4)} | {f(m['coral']['accuracy'],4)} | "
                         f"{f(m['coral_half']['accuracy'],4)} | {f(m['kmm']['accuracy'],4)} | "
                         f"{f(m['fewshot5']['accuracy'],4)} | "
                         f"{f(row['baseline_full_target_reference'],4)} |")
        L += ["", "> 全目标域参考 = 与 v1.2 迁移矩阵同口径（目标载荷全量测试）；"
              "本表主口径为「目标域 10% 时间块划作适配池、其余 90% 共用测试集」。", ""]

        sw = da["D_shrink_sweep"]
        L += ["## 表 B · E32 对齐强度扫描（0→3 HP，含 217）", "",
              "| shrink s | 准确率 | 归一化 CORAL 距离 | 源域 Fisher 比 | 类条件中心漂移 |",
              "|---|---|---|---|---|"]
        for s in sw["shrinks"]:
            r = sw["rows"][str(s)]
            L.append(f"| {float(s):.2f} | {f(r['accuracy'],4)} | "
                     f"{f(r['coral_dist_after'],6)} | {f(r['fisher_source'],3)} | "
                     f"{f(r['class_drift'],3)} |")
        L += ["", f"> 对齐前：CORAL 距离 {f(sw['coral_dist_before'],6)}，"
              f"源域 Fisher {f(sw['fisher_source_before'],3)}，"
              f"类条件漂移 {f(sw['class_drift_before'],3)}。s=1 精确退化为恒等变换。", ""]

        L += ["### 表 B2 · 结构诊断（含 217 口径，seed 0）", "",
              "| 有向对 | 方法 | 准确率 | CORAL 距离 | MMD² | 源域 Fisher | 类条件漂移 | 217 窗口主预测类 |",
              "|---|---|---|---|---|---|---|---|"]
        for pname, row in da["pairs"].items():
            s, d = pname.split("_to_")
            for meth in ("baseline", "coral", "kmm", "fewshot5"):
                e = row["protocols"]["with_217"]["methods"][meth]
                g = e["diag"]
                pred = e["pred_217_distribution"] or {}
                main_cls = max(pred, key=pred.get) if pred else "—"
                main_val = f(pred.get(main_cls), 2) if pred else "—"
                L.append(f"| {s}→{d} HP | {meth} | {f(e['accuracy'],4)} | "
                         f"{f(g['coral_dist'],5)} | {f(g['mmd'],5)} | "
                         f"{f(g['fisher_source'],3)} | {f(g['class_drift'],3)} | "
                         f"{main_cls} {main_val} |")
        L += [""]
    else:
        L += ["（缺 domain_adaptation.json）", ""]

    # ---------------- 表 C/D：E33 噪声族 ----------------
    L += ["## 表 C · E33 零适配：三族噪声 SNR 曲线（SVM 全特征 / CNN）", ""]
    if nf:
        snr_keys = nf["protocol"]["snr_levels"]
        kinds = nf["protocol"]["kinds"]
        L += ["| 模型 | 噪声族 | " + " | ".join(snr_keys) + " |",
              "|---|---|" + "---|" * len(snr_keys)]
        for cfg_name in ("clean", "cnn_clean", "cnn_white", "cnn_mix"):
            for k in kinds:
                r = nf["results"][cfg_name]
                vals = [r["SVM_all"][k][s] if "SVM_all" in r else r[k][s]
                        for s in snr_keys]
                L.append(f"| {cfg_name} | {k} | " + " | ".join(f(v,3) for v in vals) + " |")
        L += [""]

        L += ["## 表 D · E33 增广配置的 0 dB 跨族准确率（SVM 全特征）", "",
              "| 训练配置 | white | colored | impulse | 三族均值 |",
              "|---|---|---|---|---|"]
        for cfg_name in ("clean", "white_fixed", "white_fair", "mix_fair"):
            r = nf["results"][cfg_name]["SVM_all"]
            vals = [r[k]["0dB"] for k in kinds]
            L.append(f"| {cfg_name} | " + " | ".join(f(v,3) for v in vals)
                     + f" | {f(float(sum(vals)/len(vals)),3)} |")
        L += ["", "### 表 D2 · 相对零适配的增益（百分点）", "",
              "| 配置 | 噪声族 | clean | 10 dB | 4 dB | 0 dB | −4 dB | −8 dB |",
              "|---|---|---|---|---|---|---|---|"]
        for cfg_name, rows in nf["delta_vs_clean_train"].items():
            if cfg_name.startswith("fairness"):
                continue
            for k in kinds:
                L.append(f"| {cfg_name} | {k} | " +
                         " | ".join(f(100 * rows[k].get(s, 0), 1) for s in snr_keys) + " |")
        fair = nf["delta_vs_clean_train"].get("fairness_white_fair_minus_fixed", {})
        if fair:
            L += ["", "逐窗重抽 − 固定副本（白噪测试，百分点）：" +
                  ", ".join(f"{s}={100*fair[s]:+.1f}" for s in snr_keys), ""]
        L += ["### 表 D3 · 0 dB 噪声形状诊断", "",
              "| 噪声族 | 低频能量占比 | 峭度 | 峰均比 |",
              "|---|---|---|---|"]
        for k, s in nf["noise_shape_at_0db"].items():
            L.append(f"| {k} | {f(s['low_freq_energy_share'],3)} | "
                     f"{f(s['kurtosis'],1)} | {f(s['crest_factor'],1)} |")
        L += [""]
    else:
        L += ["（缺 noise_families.json）", ""]

    # ---------------- 表 E/F/G：E34 ----------------
    L += ["## 表 E · E34 217.mat 完整剂量曲线（3 目标载荷均值，5 随机子集）", ""]
    if dl:
        doses = [str(k) for k in dl["protocol"]["doses"]]
        conds = ("217_random", "upsample", "borrow_real", "prefix")
        L += ["| 条件 | " + " | ".join(doses) + " |",
              "|---|" + "---|" * len(doses)]
        for cond in conds:
            vals = []
            for k in doses:
                per_dst = [dl["A_dose_curve"][d]["conditions"][cond][k]["mean"]
                           for d in ("0", "1", "2")]
                vals.append(float(sum(per_dst) / 3.0))
            L.append(f"| {cond} | " + " | ".join(f(v,4) for v in vals) + " |")
        L += ["", "### 表 E2 · 各目标载荷的峰值剂量与 v1.2 单点对照", "",
              "| 目标 | 峰值剂量 k | 峰值准确率 | k=0 | k=478 | 前缀 116 | 随机 116 均值±std | 借真实样本 116 |",
              "|---|---|---|---|---|---|---|---|"]
        for d, p in dl["D_peaks"].items():
            L.append(f"| {d} HP | {p['peak_dose']} | {f(p['peak_accuracy'],4)} | "
                     f"{f(p['zero_accuracy'],4)} | {f(p['full_accuracy'],4)} | "
                     f"{f(p['prefix116'],4)} | {f(p['random116_mean'],4)}±{f(p['random116_std'],4)} | "
                     f"{f(p['borrow116'],4)} |")
        L += [""]

        L += ["## 表 F · E34 样本效率学习曲线（文件分层抽样，5 seed 均值）", "",
              "| 协议 | 模型 | " + " | ".join(f"{int(float(x)*100)}%" for x in dl["protocol"]["fractions"]) + " |",
              "|---|---|" + "---|" * len(dl["protocol"]["fractions"])]
        for pname, row in dl["B_learning_curve"].items():
            for m, e in row["models"].items():
                cv = e["curve"]
                L.append(f"| {pname} | {m} | " +
                         " | ".join(f(cv[str(x)],4) for x in dl["protocol"]["fractions"]) + " |")
        L += [""]

        L += ["## 表 G · E34 k=116 去混杂（准确率，5 次重复均值）", "",
              "| 目标 | 模型 | without | 217 前缀 | 217 随机 | 等量上采样 | 借真实样本 |",
              "|---|---|---|---|---|---|---|"]
        for d, row in dl["C_deconfound"].items():
            for m in ("SVM", "RF"):
                L.append(f"| {d} HP | {m} | " +
                         " | ".join(f(row[c][m]['mean'],4) for c in
                                   ("without", "217_prefix", "217_random",
                                    "upsample", "borrow_real")) + " |")
        L += [""]
    else:
        L += ["（缺 dose_learning_curve.json）", ""]

    emit(L)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
