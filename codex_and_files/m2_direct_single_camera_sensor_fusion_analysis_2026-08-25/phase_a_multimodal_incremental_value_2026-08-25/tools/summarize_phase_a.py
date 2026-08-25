from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.config import load_config
from phase_a.io import read_json, write_csv, write_json
from phase_a.paths import a0_result_dir, model_dir


def result_dir(config, condition, participant, seed):
    return a0_result_dir(config, participant, seed) if condition == "A0" else model_dir(config, condition, participant, seed) / "test_results"


def class_rows(metric):
    if "per_class" in metric:
        return metric["per_class"]
    return [{"class_id": index, "support": metric["support"][index],
             "precision": metric["per_class_precision"][index],
             "recall": metric["per_class_recall"][index], "f1": metric["per_class_f1"][index]}
            for index in range(len(metric["support"]))]


def weakest_recall(metric):
    return metric.get("weakest_class_recall", min(
        (row["recall"] for row in class_rows(metric) if row["support"]), default=0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase A without selecting best seeds")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    args = parser.parse_args(); config = load_config(args.config)
    summary_root = Path(config["output_root"]) / "summary"; summary_root.mkdir(parents=True, exist_ok=True)
    fold_rows, per_node, per_tier3, per_stage, missing = [], [], [], [], []
    prediction_rows = {condition: [] for condition in (f"A{i}" for i in range(8))}
    metric_lookup = {}
    for condition in prediction_rows:
        for participant in config["participants"]:
            for seed in config["seeds"]:
                for split in ("test_all", "test_normal", "test_fault"):
                    directory = result_dir(config, condition, participant, seed)
                    metrics_path = directory / f"{split}_metrics.json"
                    if not metrics_path.is_file():
                        missing.append(str(metrics_path)); continue
                    value = read_json(metrics_path); metric_lookup[(condition, participant, seed, split)] = value
                    fold_rows.append({
                        "condition": condition, "participant": participant, "seed": seed, "split": split,
                        "samples": value["samples"], "node_accuracy": value["node"]["accuracy"],
                        "node_macro_f1": value["node"]["macro_f1"],
                        "node_weakest_recall": weakest_recall(value["node"]),
                        "tier3_accuracy": value["tier3"]["accuracy"], "tier3_macro_f1": value["tier3"]["macro_f1"],
                    })
                    for row in class_rows(value["node"]):
                        per_node.append({"condition": condition, "participant": participant, "seed": seed,
                                         "split": split, **row})
                    for row in class_rows(value["tier3"]):
                        per_tier3.append({"condition": condition, "participant": participant, "seed": seed,
                                          "split": split, **row})
                    for stage, stage_value in value["per_stage"].items():
                        per_stage.append({"condition": condition, "participant": participant, "seed": seed,
                            "split": split, "stage": stage, "samples": stage_value["samples"],
                            "node_accuracy": stage_value["node"]["accuracy"], "node_macro_f1": stage_value["node"]["macro_f1"],
                            "tier3_accuracy": stage_value["tier3"]["accuracy"], "tier3_macro_f1": stage_value["tier3"]["macro_f1"]})
                    if split == "test_all":
                        with (directory / f"{split}_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
                            prediction_rows[condition].extend(csv.DictReader(handle))
    write_csv(summary_root / "fold_seed_metrics.csv", fold_rows)
    write_csv(summary_root / "per_35_node.csv", per_node)
    write_csv(summary_root / "per_31_tier3.csv", per_tier3)
    write_csv(summary_root / "per_stage.csv", per_stage)
    condition_rows, confusion_rows, gates = [], [], {}
    for condition in prediction_rows:
        rows = [row for row in fold_rows if row["condition"] == condition]
        for split in ("test_all", "test_normal", "test_fault"):
            selected = [row for row in rows if row["split"] == split]
            if selected:
                condition_rows.append({"condition": condition, "split": split, "runs": len(selected),
                    **{key: sum(row[key] for row in selected) / len(selected) for key in
                       ("node_accuracy", "node_macro_f1", "node_weakest_recall", "tier3_accuracy", "tier3_macro_f1")}})
        counts = Counter((int(row["true_node_idx"]), int(row["pred_node_idx"]))
                         for row in prediction_rows[condition] if row["true_node_idx"] != row["pred_node_idx"])
        confusion_rows.extend({"condition": condition, "label_space": "node", "rank": rank,
                               "true_id": pair[0], "pred_id": pair[1], "count": count}
                              for rank, (pair, count) in enumerate(counts.most_common(12), 1))
        tier3_counts = Counter((int(row["true_tier3_id"]), int(row["pred_tier3_id"]))
                               for row in prediction_rows[condition]
                               if row["true_tier3_id"] != row["pred_tier3_id"])
        confusion_rows.extend({"condition": condition, "label_space": "tier3", "rank": rank,
                               "true_id": pair[0], "pred_id": pair[1], "count": count}
                              for rank, (pair, count) in enumerate(tier3_counts.most_common(12), 1))
        if condition != "A0":
            positive_macro = positive_weak = joint_positive = fault_nonnegative = comparisons = 0
            for participant in config["participants"]:
                for seed in config["seeds"]:
                    candidate = metric_lookup.get((condition, participant, seed, "test_all"))
                    baseline = metric_lookup.get(("A0", participant, seed, "test_all"))
                    candidate_fault = metric_lookup.get((condition, participant, seed, "test_fault"))
                    baseline_fault = metric_lookup.get(("A0", participant, seed, "test_fault"))
                    if not all((candidate, baseline, candidate_fault, baseline_fault)): continue
                    comparisons += 1
                    macro_gain = candidate["node"]["macro_f1"] > baseline["node"]["macro_f1"]
                    positive_macro += macro_gain
                    cweak = weakest_recall(candidate["node"])
                    bweak = weakest_recall(baseline["node"])
                    weak_gain = cweak > bweak
                    positive_weak += weak_gain
                    joint_positive += macro_gain and weak_gain
                    margin = float(config["fault_noninferiority_margin_pp"]) / 100.0
                    fault_nonnegative += candidate_fault["node"]["macro_f1"] >= baseline_fault["node"]["macro_f1"] - margin
            threshold = int(config["majority_positive_minimum"])
            stress_deltas = []
            offset_deltas_by_scenario = {}
            if condition in {"A3", "A4", "A5", "A6", "A7"}:
                for participant in config["participants"]:
                    for seed in config["seeds"]:
                        path = model_dir(config, condition, participant, seed) / "stress_results" / "all_new_modalities_missing" / "test_all_metrics.json"
                        baseline = metric_lookup.get(("A0", participant, seed, "test_all"))
                        if path.is_file() and baseline:
                            stress_deltas.append(read_json(path)["node"]["macro_f1"] - baseline["node"]["macro_f1"])
                        stress_root = model_dir(config, condition, participant, seed) / "stress_results"
                        clean_path = stress_root / "clean" / "test_all_metrics.json"
                        if clean_path.is_file():
                            clean_value = read_json(clean_path)["node"]["macro_f1"]
                            for offset_path in stress_root.glob("*_offset_*/*metrics.json"):
                                if offset_path.name != "test_all_metrics.json":
                                    continue
                                scenario = offset_path.parent.name
                                offset_deltas_by_scenario.setdefault(scenario, []).append(
                                    read_json(offset_path)["node"]["macro_f1"] - clean_value
                                )
            fallback_tolerance = float(config["fallback_node_macro_f1_tolerance_pp"]) / 100.0
            latency_files = list((Path(config["output_root"]) / condition).glob("**/latency_cached_feature_scope.json"))
            offset_mean_deltas = {name: sum(values) / len(values)
                                  for name, values in offset_deltas_by_scenario.items()}
            gates[condition] = {"available_fold_seed_pairs": comparisons,
                "node_macro_f1_positive": positive_macro, "weakest_recall_positive": positive_weak,
                "node_macro_and_weakest_joint_positive": joint_positive,
                "fault_node_macro_f1_noninferior_with_margin": fault_nonnegative,
                "fault_margin_pp": config["fault_noninferiority_margin_pp"],
                "majority_threshold": threshold, "majority_macro_pass": positive_macro >= threshold,
                "majority_weakest_pass": positive_weak >= threshold,
                "majority_joint_pass": joint_positive >= threshold,
                "fault_majority_noninferior_pass": fault_nonnegative >= threshold,
                "bootstrap": "PENDING" if not (summary_root / f"paired_bootstrap_{condition}_vs_A0.json").is_file() else "AVAILABLE",
                "fallback_pairs": len(stress_deltas),
                "fallback_mean_node_macro_f1_delta": sum(stress_deltas) / len(stress_deltas) if stress_deltas else None,
                "fallback_pass": (sum(stress_deltas) / len(stress_deltas) >= -fallback_tolerance) if stress_deltas else None,
                "time_offset_scenarios": offset_mean_deltas,
                "worst_time_offset_mean_node_macro_f1_delta": min(offset_mean_deltas.values()) if offset_mean_deltas else None,
                "latency_files": len(latency_files),
                "latency": "UNSET_BUDGET" if config["latency"]["target_p95_ms"] is None else "CHECK latency JSON"}
    write_csv(summary_root / "condition_summary.csv", condition_rows)
    write_csv(summary_root / "top_12_confusions.csv", confusion_rows)
    write_csv(summary_root / "top_12_node_confusions.csv",
              [row for row in confusion_rows if row["label_space"] == "node"])
    write_json(summary_root / "incremental_value_gates.json", gates)
    completed = len(missing) == 0
    report = ["# Phase A 实验结果汇总", "", f"状态：{'COMPLETE' if completed else 'PENDING'}。",
              "", "本页由 `tools/summarize_phase_a.py` 自动生成；不做 best-seed 选择。",
              "", "## 完整性", "", f"- 已找到的 fold×seed×split 指标文件：{len(fold_rows)}",
              f"- 尚缺指标文件：{len(missing)}", "", "## 主要输出", "",
              "- `condition_summary.csv`：总体、Normal、Fault。",
              "- `per_stage.csv`：Stage 分层。",
              "- `per_31_tier3.csv`：31 Tier3 全类别。",
              "- `per_35_node.csv`：35 node 全类别。",
              "- `top_12_confusions.csv`：每条件当前前 12 个 node 与 Tier3 混淆对。",
              "- `paired_bootstrap_Ax_vs_A0.json`：配对 clip bootstrap CI。",
              "- `incremental_value_gates.json`：多数正增益、弱类、Fault、压力和延迟门槛。", ""]
    if missing:
        report.extend(["## 尚未运行", "", "实验代码包已经就绪；以下结果需要完成 GPU 任务后生成。",
                       "首批缺失示例：", ""] + [f"- `{path}`" for path in missing[:20]])
    (summary_root / "PHASE_A_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"summary={summary_root}; missing={len(missing)}")


if __name__ == "__main__":
    main()
