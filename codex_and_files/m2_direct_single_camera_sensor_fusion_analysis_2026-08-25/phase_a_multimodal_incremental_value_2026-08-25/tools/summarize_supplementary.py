from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.io import read_json, write_csv, write_json
from phase_a.supplementary import (
    SUPPLEMENTARY_IDS,
    experiment_spec,
    load_supplementary_config,
    supplementary_model_dir,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate S1-S12 without best-seed selection")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--conditions", nargs="+", choices=SUPPLEMENTARY_IDS, default=list(SUPPLEMENTARY_IDS))
    args = parser.parse_args()
    config = load_supplementary_config(args.config)
    summary = Path(config["output_root"]) / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    per_node = []
    per_tier3 = []
    per_stage = []
    confusion_rows = []
    low_recall_rows = []
    missing = []
    metric_lookup = {}
    threshold = float(config["low_recall_threshold"])
    conditions = list(dict.fromkeys(args.conditions))
    for condition in conditions:
        spec = experiment_spec(config, condition)
        for participant in config["base"]["participants"]:
            for seed in config["base"]["seeds"]:
                result_dir = supplementary_model_dir(config, condition, participant, seed) / "test_results"
                for split in ("test_all", "test_normal", "test_fault"):
                    metrics_path = result_dir / f"{split}_metrics.json"
                    if not metrics_path.is_file():
                        missing.append(str(metrics_path))
                        continue
                    value = read_json(metrics_path)
                    metric_lookup[(condition, participant, seed, split)] = value
                    node = value.get("node")
                    tier3 = value["tier3"]
                    fold_rows.append({
                        "condition": condition, "participant": participant, "seed": seed,
                        "split": split, "task": spec["task"], "modality": spec["modality"],
                        "encoder": spec["encoder"], "samples": value["samples"],
                        "node_accuracy": node["accuracy"] if node else "",
                        "node_macro_f1": node["macro_f1"] if node else "",
                        "node_weakest_recall": node["weakest_class_recall"] if node else "",
                        "tier3_accuracy": tier3["accuracy"], "tier3_macro_f1": tier3["macro_f1"],
                        "tier3_weakest_recall": tier3["weakest_class_recall"],
                    })
                    if node:
                        per_node.extend({
                            "condition": condition, "participant": participant, "seed": seed,
                            "split": split, **row,
                        } for row in node["per_class"])
                    per_tier3.extend({
                        "condition": condition, "participant": participant, "seed": seed,
                        "split": split, **row,
                    } for row in tier3["per_class"])
                    for stage, stage_value in value["per_stage"].items():
                        stage_node = stage_value.get("node")
                        per_stage.append({
                            "condition": condition, "participant": participant, "seed": seed,
                            "split": split, "stage": stage, "samples": stage_value["samples"],
                            "node_accuracy": stage_node["accuracy"] if stage_node else "",
                            "node_macro_f1": stage_node["macro_f1"] if stage_node else "",
                            "tier3_accuracy": stage_value["tier3"]["accuracy"],
                            "tier3_macro_f1": stage_value["tier3"]["macro_f1"],
                        })
                    if split != "test_all":
                        continue
                    predictions = read_csv(result_dir / "test_all_predictions.csv")
                    if node:
                        counts = Counter((int(row["true_node_idx"]), int(row["pred_node_idx"]))
                                         for row in predictions if row["true_node_idx"] != row["pred_node_idx"])
                        confusion_rows.extend({
                            "condition": condition, "participant": participant, "seed": seed,
                            "label_space": "node", "rank": rank, "true_id": pair[0],
                            "pred_id": pair[1], "count": count,
                        } for rank, (pair, count) in enumerate(counts.most_common(12), 1))
                        low_ids = {int(row["class_id"]) + 1 for row in node["per_class"]
                                   if row["support"] and row["recall"] < threshold}
                        low_recall_rows.extend({
                            "condition": condition, "participant": participant, "seed": seed,
                            "label_space": "node", "true_id": int(row["true_node_idx"]),
                            "sample_name": row["sample_name"], "predicted_id": int(row["pred_node_idx"]),
                            "stage_id": int(row["stage_id"]), "run": row["run"],
                        } for row in predictions
                            if int(row["true_node_idx"]) in low_ids and row["true_node_idx"] != row["pred_node_idx"])
                    tier3_counts = Counter((int(row["true_tier3_id"]), int(row["pred_tier3_id"]))
                                           for row in predictions if row["true_tier3_id"] != row["pred_tier3_id"])
                    confusion_rows.extend({
                        "condition": condition, "participant": participant, "seed": seed,
                        "label_space": "tier3", "rank": rank, "true_id": pair[0],
                        "pred_id": pair[1], "count": count,
                    } for rank, (pair, count) in enumerate(tier3_counts.most_common(12), 1))
                    low_tier3 = {int(row["class_id"]) for row in tier3["per_class"]
                                 if row["support"] and row["recall"] < threshold}
                    low_recall_rows.extend({
                        "condition": condition, "participant": participant, "seed": seed,
                        "label_space": "tier3", "true_id": int(row["true_tier3_id"]),
                        "sample_name": row["sample_name"], "predicted_id": int(row["pred_tier3_id"]),
                        "stage_id": int(row["stage_id"]), "run": row["run"],
                    } for row in predictions
                        if int(row["true_tier3_id"]) in low_tier3 and row["true_tier3_id"] != row["pred_tier3_id"])
    write_csv(summary / "fold_seed_metrics.csv", fold_rows)
    write_csv(summary / "per_35_node.csv", per_node)
    write_csv(summary / "per_31_tier3.csv", per_tier3)
    write_csv(summary / "per_stage.csv", per_stage)
    write_csv(summary / "top_12_confusions.csv", confusion_rows)
    write_csv(summary / "low_recall_misclassified_samples.csv", low_recall_rows)

    condition_rows = []
    for condition in conditions:
        rows = [row for row in fold_rows if row["condition"] == condition]
        for split in ("test_all", "test_normal", "test_fault"):
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                continue
            numeric = ("node_accuracy", "node_macro_f1", "node_weakest_recall",
                       "tier3_accuracy", "tier3_macro_f1", "tier3_weakest_recall")
            averaged = {}
            for key in numeric:
                values = [float(row[key]) for row in selected if row[key] != ""]
                averaged[key] = sum(values) / len(values) if values else ""
            condition_rows.append({"condition": condition, "split": split,
                                   "runs": len(selected), **averaged})
    write_csv(summary / "condition_summary.csv", condition_rows)
    fold_delta_rows = []
    gates = {}
    majority = int(config["base"]["majority_positive_minimum"])
    fault_margin = float(config["base"]["fault_noninferiority_margin_pp"]) / 100.0
    for item in config["paired_comparisons"]:
        candidate = item["candidate"]
        baseline = item["baseline"]
        if candidate not in conditions or baseline not in conditions:
            continue
        candidate_node = experiment_spec(config, candidate)["task"] != "direct_tier3"
        baseline_node = experiment_spec(config, baseline)["task"] != "direct_tier3"
        if candidate_node != baseline_node:
            continue
        label_space = "node" if candidate_node else "tier3"
        macro_positive = weakest_positive = joint_positive = fault_noninferior = available = 0
        for participant in config["base"]["participants"]:
            for seed in config["base"]["seeds"]:
                candidate_all = metric_lookup.get((candidate, participant, seed, "test_all"))
                baseline_all = metric_lookup.get((baseline, participant, seed, "test_all"))
                candidate_fault = metric_lookup.get((candidate, participant, seed, "test_fault"))
                baseline_fault = metric_lookup.get((baseline, participant, seed, "test_fault"))
                if not all((candidate_all, baseline_all, candidate_fault, baseline_fault)):
                    continue
                c_all = candidate_all[label_space]
                b_all = baseline_all[label_space]
                c_fault = candidate_fault[label_space]
                b_fault = baseline_fault[label_space]
                macro_delta = float(c_all["macro_f1"]) - float(b_all["macro_f1"])
                weakest_delta = float(c_all["weakest_class_recall"]) - float(b_all["weakest_class_recall"])
                fault_delta = float(c_fault["macro_f1"]) - float(b_fault["macro_f1"])
                macro_gain = macro_delta > 0
                weakest_gain = weakest_delta > 0
                available += 1
                macro_positive += int(macro_gain)
                weakest_positive += int(weakest_gain)
                joint_positive += int(macro_gain and weakest_gain)
                fault_noninferior += int(fault_delta >= -fault_margin)
                fold_delta_rows.append({
                    "candidate": candidate, "baseline": baseline, "purpose": item["purpose"],
                    "participant": participant, "seed": seed, "label_space": label_space,
                    "macro_f1_delta": macro_delta, "weakest_recall_delta": weakest_delta,
                    "fault_macro_f1_delta": fault_delta, "macro_f1_positive": macro_gain,
                    "weakest_recall_positive": weakest_gain,
                    "joint_positive": macro_gain and weakest_gain,
                    "fault_noninferior": fault_delta >= -fault_margin,
                })
        key = f"{candidate}_vs_{baseline}"
        gates[key] = {
            "candidate": candidate, "baseline": baseline, "purpose": item["purpose"],
            "label_space": label_space, "available_fold_seed_pairs": available,
            "majority_threshold": majority, "macro_f1_positive": macro_positive,
            "weakest_recall_positive": weakest_positive, "joint_positive": joint_positive,
            "fault_noninferior_with_margin": fault_noninferior,
            "fault_margin_pp": config["base"]["fault_noninferiority_margin_pp"],
            "majority_macro_pass": macro_positive >= majority,
            "majority_weakest_pass": weakest_positive >= majority,
            "majority_joint_pass": joint_positive >= majority,
            "fault_majority_noninferior_pass": fault_noninferior >= majority,
            "bootstrap": "AVAILABLE" if (summary / f"paired_bootstrap_{candidate}_vs_{baseline}.json").is_file() else "PENDING",
        }
    write_csv(summary / "paired_fold_seed_deltas.csv", fold_delta_rows)
    write_json(summary / "incremental_value_gates.json", gates)
    comparison_status = []
    for item in config["paired_comparisons"]:
        if item["candidate"] not in conditions or item["baseline"] not in conditions:
            continue
        path = summary / f"paired_bootstrap_{item['candidate']}_vs_{item['baseline']}.json"
        comparison_status.append({**item, "bootstrap": "AVAILABLE" if path.is_file() else "PENDING",
                                  "path": str(path)})
    write_json(summary / "comparison_status.json", {"comparisons": comparison_status})
    completed = len(missing) == 0
    report = [
        "# 右手 EMG/IMU 补充实验汇总", "",
        f"状态：{'COMPLETE' if completed else 'PENDING'}。", "",
        f"本次汇总条件：{', '.join(conditions)}。", "",
        "本汇总不选择 best seed；统计单位保持四折 × 三 seed。", "",
        "## 条件", "",
        "- S1-S4：Tier3 预训练信号特征 → 冻结特征 → scratch M2 + Node head。",
        "- S5-S8：独立 scratch encoder → Direct Node。",
        "- S9-S12：独立 scratch encoder → Direct Tier3，同时作为 S1-S4 上游。", "",
        "## 完整性", "",
        f"- 已找到 fold×seed×split 指标：{len(fold_rows)}",
        f"- 尚缺指标：{len(missing)}", "",
        "## 文件", "",
        "- `condition_summary.csv`：总体、Normal、Fault。",
        "- `per_stage.csv`：Stage 分层。",
        "- `per_35_node.csv` / `per_31_tier3.csv`：逐类别。",
        "- `top_12_confusions.csv`：混淆对。",
        "- `low_recall_misclassified_samples.csv`：Recall<80% 类别的错误样本名。",
        "- `paired_fold_seed_deltas.csv`：每个预注册比较在 12 个 fold×seed 上的逐次增益。",
        "- `incremental_value_gates.json`：多数正增益、最弱 Recall 与 Fault 非劣门槛。",
        "- `comparison_status.json`：预注册配对比较与 bootstrap 状态。", "",
    ]
    if missing:
        report.extend(["## 首批缺失文件", ""] + [f"- `{path}`" for path in missing[:30]] + [""])
    (summary / "SUPPLEMENTARY_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"summary={summary}; missing={len(missing)}")


if __name__ == "__main__":
    main()
