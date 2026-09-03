from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import pandas as pd
try:
    from scipy import stats
except ImportError:  # Core summaries remain usable in a minimal Phase B environment.
    stats = None

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, fusion_root, load_config, outer_m2_root,
    output_root, phase_b_output_root, read_json, write_json,
)


METRICS = (
    "node_accuracy", "node_macro_f1", "node_weakest_recall",
    "tier3_accuracy", "tier3_macro_f1",
)


def result_root(config: dict, condition: str, participant: str, seed: int) -> Path:
    phase_b = phase_b_output_root(config)
    if condition == "B0":
        return phase_b / "B0_phase_a" / "A2" / f"{participant}_as_test" / f"seed_{seed}" / "test_results"
    if condition == "B1":
        return phase_b / "B1" / f"{participant}_as_test" / f"seed_{seed}" / "test_results"
    if condition == "B1_IMU_M2":
        return fusion_root(config, participant, seed) / "test_results"
    if condition == "IMU_Direct":
        return (
            phase_b / "outer_experts" / f"{participant}_as_test" / f"seed_{seed}"
            / "imu_direct_node" / "test_results"
        )
    if condition == "IMU_M2":
        return outer_m2_root(config, participant, seed) / "test_results"
    raise ValueError(condition)


def flatten_metrics(value: dict) -> dict[str, float | int]:
    return {
        "samples": int(value["samples"]),
        "node_accuracy": float(value["node"]["accuracy"]),
        "node_macro_f1": float(value["node"]["macro_f1"]),
        "node_weakest_recall": float(value["node"]["weakest_class_recall"]),
        "tier3_accuracy": float(value["tier3"]["accuracy"]),
        "tier3_macro_f1": float(value["tier3"]["macro_f1"]),
    }


def paired_stats(left: pd.Series, right: pd.Series) -> dict[str, float | int | None]:
    delta = (left.astype(float) - right.astype(float)).to_numpy()
    count = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if count > 1 else 0.0
    if count > 1:
        critical = float(stats.t.ppf(0.975, count - 1)) if stats is not None else (2.200985 if count == 12 else 1.96)
        half = critical * sd / math.sqrt(count)
        t_p = float(stats.ttest_rel(left, right).pvalue) if stats is not None and sd > 0 else (1.0 if sd == 0 else None)
    else:
        half, t_p = 0.0, None
    try:
        w_p = 1.0 if not delta.any() else (float(stats.wilcoxon(delta).pvalue) if stats is not None else None)
    except ValueError:
        w_p = None
    return {
        "paired_runs": count, "mean_delta": mean, "sd_delta": sd,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()), "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def fmt(value: float | None, scale: float = 100.0) -> str:
    return "NA" if value is None else f"{value * scale:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize B1_IMU_M2 and paired Phase B references")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    conditions = ("B0", "B1", "B1_IMU_M2", "IMU_Direct", "IMU_M2")
    rows, missing = [], []
    for condition in conditions:
        for participant in config["participants"]:
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                for split in config["evaluation"]["splits"]:
                    path = result_root(config, condition, participant, seed) / f"{split}_metrics.json"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    rows.append({
                        "condition": condition, "participant": participant,
                        "seed": seed, "split": split, **flatten_metrics(read_json(path)),
                    })
    frame = pd.DataFrame(rows)
    summary_rows = []
    if not frame.empty:
        for (condition, split), group in frame.groupby(["condition", "split"], sort=False):
            summary_rows.append({
                "condition": condition, "split": split, "completed_runs": len(group),
                **{f"{metric}_mean": float(group[metric].mean()) for metric in METRICS},
                **{f"{metric}_sd": float(group[metric].std(ddof=1)) for metric in METRICS},
            })
    comparisons = []
    pairs = (("B1_IMU_M2", "B1"), ("B1_IMU_M2", "B0"), ("IMU_M2", "IMU_Direct"))
    for candidate, reference in pairs:
        for split in config["evaluation"]["splits"]:
            left = frame[(frame.condition == candidate) & (frame.split == split)]
            right = frame[(frame.condition == reference) & (frame.split == split)]
            merged = left.merge(right, on=["participant", "seed", "split"], suffixes=("_candidate", "_reference"))
            for metric in METRICS:
                if len(merged):
                    comparisons.append({
                        "candidate": candidate, "reference": reference,
                        "split": split, "metric": metric,
                        **paired_stats(merged[f"{metric}_candidate"], merged[f"{metric}_reference"]),
                    })

    weights = []
    for participant in config["participants"]:
        for seed_value in config["seeds"]:
            path = fusion_root(config, participant, int(seed_value)) / "fit_summary.json"
            if path.is_file():
                value = read_json(path)
                weights.append({
                    "participant": participant, "seed": int(seed_value),
                    **{f"weight_{name}": float(weight) for name, weight in zip(value["experts"], value["weights"])},
                    **{f"temperature_{name}": float(temp) for name, temp in zip(value["experts"], value["temperatures"])},
                })

    destination = output_root(config) / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "run_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(destination / "condition_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(destination / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weights).to_csv(destination / "fusion_parameters.csv", index=False, encoding="utf-8-sig")
    expected = len(conditions) * len(config["participants"]) * len(config["seeds"]) * len(config["evaluation"]["splits"])
    write_json(destination / "completeness.json", {
        "complete": not missing, "found_metric_files": len(rows),
        "expected_metric_files": expected, "missing": missing,
    })

    lines = [
        "# B1_IMU_M2 自动汇总", "",
        f"完整性：{'完整' if not missing else '未完整'}（{len(rows)}/{expected} 个指标文件）。", "",
        "## 主要结果", "",
        "| 方法 | split | Node Macro-F1 (%) | Node Acc (%) | 最弱类别 Recall (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["condition"] in ("B0", "B1", "B1_IMU_M2"):
            lines.append(
                f"| {row['condition']} | {row['split']} | {fmt(row['node_macro_f1_mean'])} "
                f"| {fmt(row['node_accuracy_mean'])} | {fmt(row['node_weakest_recall_mean'])} |"
            )
    lines.extend(["", "## 配对差异", "", "正值表示候选方法优于参照方法。", "",
                  "| 候选 − 参照 | split | 指标 | 均值差 (pp) | 95% CI (pp) | 胜/平/负 |",
                  "|---|---|---|---:|---:|---:|"])
    for row in comparisons:
        if row["metric"] in ("node_macro_f1", "node_accuracy", "node_weakest_recall"):
            lines.append(
                f"| {row['candidate']} − {row['reference']} | {row['split']} | {row['metric']} "
                f"| {fmt(row['mean_delta'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] "
                f"| {row['wins']}/{row['ties']}/{row['losses']} |"
            )
    if weights:
        weight_frame = pd.DataFrame(weights)
        lines.extend(["", "## 融合参数", ""])
        for name in config["fusion"]["experts"]:
            lines.append(
                f"- {name}: weight = {weight_frame[f'weight_{name}'].mean():.4f}, "
                f"temperature = {weight_frame[f'temperature_{name}'].mean():.4f}（12 个 outer×seed 平均）"
            )
    lines.extend(["", "## 解释注意", "",
                  "本实验的关键配对比较是 B1_IMU_M2 与 B1。两者使用相同的双摄像头 M2、外层划分、seed 和静态融合形式；差异是 IMU 专家由 Direct 头改为 M2 历史头。",
                  "IMU encoder 保持冻结且沿用 Phase B 已训练权重，因此结果应解释为“在固定 IMU 表示上加入历史头”的收益，而不是重新训练整个 IMU 网络的收益。", ""])
    (destination / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Found {len(rows)}/{expected} metrics; missing {len(missing)}. Saved to {destination}")


if __name__ == "__main__":
    main()
