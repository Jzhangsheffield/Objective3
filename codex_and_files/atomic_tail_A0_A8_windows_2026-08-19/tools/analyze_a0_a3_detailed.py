from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENTS = ("A0", "A1", "A2", "A3")
PARTICIPANTS = ("A", "D", "J", "M")
SEEDS = (1, 2, 42)
SPLITS = ("test_all", "test_normal", "test_fault")
METRICS = ("node_accuracy", "node_macro_f1", "tier3_accuracy", "tier3_macro_f1")
COMPARISONS = (
    ("A1-A0", "A1", "A0"),
    ("A2-A0", "A2", "A0"),
    ("A3-A0", "A3", "A0"),
    ("A2-A1", "A2", "A1"),
    ("A3-A2", "A3", "A2"),
    ("A3-A1", "A3", "A1"),
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(metrics: dict) -> dict[str, float]:
    return {
        "node_accuracy": float(metrics["node"]["accuracy"]),
        "node_macro_f1": float(metrics["node"]["macro_f1"]),
        "node_balanced_accuracy": float(metrics["node"]["balanced_accuracy"]),
        "tier3_accuracy": float(metrics["tier3"]["accuracy"]),
        "tier3_macro_f1": float(metrics["tier3"]["macro_f1"]),
        "tier3_balanced_accuracy": float(metrics["tier3"]["balanced_accuracy"]),
        "samples": int(metrics["samples"]),
    }


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def paired_summary(values: list[float], epsilon: float = 1e-12) -> dict:
    result = summary(values)
    n = len(values)
    se = float(result["sd"]) / math.sqrt(n) if n > 1 else 0.0
    critical = T_CRITICAL_975.get(n - 1, 1.96)
    result.update(
        {
            "ci95_low": float(result["mean"]) - critical * se,
            "ci95_high": float(result["mean"]) + critical * se,
            "wins": sum(value > epsilon for value in values),
            "ties": sum(abs(value) <= epsilon for value in values),
            "losses": sum(value < -epsilon for value in values),
        }
    )
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_node_names(package_root: Path) -> dict[int, str]:
    graph = read_json(package_root / "assets" / "integrated_task_graph_latest.json")
    return {
        int(node["node_idx"]): str(node["node_id"]).replace(f"node_{node['node_idx']}_", "")
        for node in graph["nodes"]
        if 1 <= int(node["node_idx"]) <= 35
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    package_root = Path(args.package_root).resolve()
    output_root = package_root / "outputs"

    rows: dict[tuple[str, str, int, str], dict] = {}
    coverage = defaultdict(list)
    audits: dict[tuple[str, str, int], dict] = {}
    train_logs: dict[tuple[str, str, int], list[dict]] = {}
    for experiment in EXPERIMENTS:
        for participant in PARTICIPANTS:
            for seed in SEEDS:
                run_root = output_root / experiment / "all_runs" / f"{participant}_as_test" / f"seed_{seed}"
                completed = run_root / "completed.json"
                if completed.is_file():
                    coverage[experiment].append(f"{participant}:{seed}")
                audit_path = run_root / "augmentation_audit.json"
                if audit_path.is_file():
                    audits[(experiment, participant, seed)] = read_json(audit_path)
                train_path = run_root / "train_log.json"
                if train_path.is_file():
                    train_logs[(experiment, participant, seed)] = read_json(train_path)
                for split in SPLITS:
                    path = run_root / "test_results_actual_order" / f"{split}_metrics.json"
                    if path.is_file():
                        metrics = read_json(path)
                        row = metric_row(metrics)
                        row["per_stage"] = metrics["per_stage"]
                        rows[(experiment, participant, seed, split)] = row

    aggregate = {}
    participant_summary = {}
    seed_summary = {}
    paired = {}
    stages = {}
    for split in SPLITS:
        aggregate[split] = {}
        participant_summary[split] = {}
        seed_summary[split] = {}
        paired[split] = {}
        for experiment in EXPERIMENTS:
            aggregate[split][experiment] = {
                metric: summary([rows[(experiment, participant, seed, split)][metric] for participant in PARTICIPANTS for seed in SEEDS])
                for metric in METRICS
            }
            participant_summary[split][experiment] = {
                participant: {
                    metric: summary([rows[(experiment, participant, seed, split)][metric] for seed in SEEDS])
                    for metric in METRICS
                }
                for participant in PARTICIPANTS
            }
            seed_summary[split][experiment] = {
                str(seed): {
                    metric: summary([rows[(experiment, participant, seed, split)][metric] for participant in PARTICIPANTS])
                    for metric in METRICS
                }
                for seed in SEEDS
            }
        for label, left, right in COMPARISONS:
            paired[split][label] = {
                metric: paired_summary([
                    rows[(left, participant, seed, split)][metric] - rows[(right, participant, seed, split)][metric]
                    for participant in PARTICIPANTS for seed in SEEDS
                ])
                for metric in METRICS
            }

    for label, left, right in COMPARISONS:
        stages[label] = {}
        for stage in (1, 2, 3):
            values = []
            for participant in PARTICIPANTS:
                for seed in SEEDS:
                    left_value = rows[(left, participant, seed, "test_all")]["per_stage"][str(stage)]["node"]["accuracy"]
                    right_value = rows[(right, participant, seed, "test_all")]["per_stage"][str(stage)]["node"]["accuracy"]
                    values.append(float(left_value) - float(right_value))
            stages[label][str(stage)] = paired_summary(values)

    audit_summary = {}
    for experiment in EXPERIMENTS:
        audit_summary[experiment] = {}
        for key in (
            "samples", "augmentation_changed_fraction", "mean_normalized_kendall_distance",
            "tail_aux_eligible_fraction",
        ):
            audit_summary[experiment][key] = summary([
                float(audits[(experiment, participant, seed)][key])
                for participant in PARTICIPANTS for seed in SEEDS
            ])
        audit_summary[experiment]["active_tail_fraction"] = summary([
            float(audits[(experiment, participant, seed)]["tail_reason_counts"].get("active_incomplete_atomic_prefix", 0))
            / max(1, float(audits[(experiment, participant, seed)]["samples"]))
            for participant in PARTICIPANTS for seed in SEEDS
        ])

    training_summary = {}
    for experiment in EXPERIMENTS:
        nonempty = [train_logs[(experiment, participant, seed)] for participant in PARTICIPANTS for seed in SEEDS if train_logs.get((experiment, participant, seed))]
        training_summary[experiment] = {
            "trained_runs": len(nonempty),
            "epochs": summary([float(len(log)) for log in nonempty]) if nonempty else None,
            "first_epoch_accuracy_ge_0_99": summary([
                float(next((row["epoch"] for row in log if row["train_node_accuracy"] >= 0.99), len(log) + 1))
                for log in nonempty
            ]) if nonempty else None,
            "first_epoch_accuracy_ge_0_999": summary([
                float(next((row["epoch"] for row in log if row["train_node_accuracy"] >= 0.999), len(log) + 1))
                for log in nonempty
            ]) if nonempty else None,
            "final_accuracy": summary([float(log[-1]["train_node_accuracy"]) for log in nonempty]) if nonempty else None,
            "final_loss": summary([float(log[-1]["train_loss"]) for log in nonempty]) if nonempty else None,
            "total_seconds": summary([sum(float(row["seconds"]) for row in log) for log in nonempty]) if nonempty else None,
        }

    prediction_analysis = {}
    node_names = load_node_names(package_root)
    for split in SPLITS:
        outcomes = Counter()
        clusters: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        class_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for participant in PARTICIPANTS:
            for seed in SEEDS:
                prediction_by_experiment = {}
                for experiment in ("A0", "A3"):
                    path = output_root / experiment / "all_runs" / f"{participant}_as_test" / f"seed_{seed}" / "test_results_actual_order" / f"{split}_predictions.csv"
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        prediction_by_experiment[experiment] = {row["sample_name"]: row for row in csv.DictReader(handle)}
                if set(prediction_by_experiment["A0"]) != set(prediction_by_experiment["A3"]):
                    raise RuntimeError(f"Prediction sample mismatch: {participant} seed={seed} {split}")
                for sample_name, row0 in prediction_by_experiment["A0"].items():
                    row3 = prediction_by_experiment["A3"][sample_name]
                    truth = int(row0["true_node_idx"])
                    correct0 = int(row0["pred_node_idx"] == row0["true_node_idx"])
                    correct3 = int(row3["pred_node_idx"] == row3["true_node_idx"])
                    if correct0 and correct3:
                        outcomes["both_correct"] += 1
                    elif correct3:
                        outcomes["A3_only_correct"] += 1
                    elif correct0:
                        outcomes["A0_only_correct"] += 1
                    else:
                        outcomes["both_wrong"] += 1
                    outcomes["same_prediction"] += int(row0["pred_node_idx"] == row3["pred_node_idx"])
                    outcomes["total"] += 1
                    cluster = (participant, row0["run"])
                    clusters[cluster][0] += correct3 - correct0
                    clusters[cluster][1] += 1
                    class_counts[truth][0] += correct3
                    class_counts[truth][1] += correct0
                    class_counts[truth][2] += 1
        random_generator = random.Random(20260820)
        cluster_values = list(clusters.values())
        bootstrap = []
        for _ in range(int(args.bootstrap_repetitions)):
            sampled = [random_generator.choice(cluster_values) for _ in cluster_values]
            bootstrap.append(sum(item[0] for item in sampled) / max(1, sum(item[1] for item in sampled)))
        class_rows = [
            {
                "node_idx": node,
                "node_name": node_names.get(node, f"node_{node}"),
                "support_clip_seed": values[2],
                "A0_accuracy": values[1] / values[2],
                "A3_accuracy": values[0] / values[2],
                "delta": (values[0] - values[1]) / values[2],
            }
            for node, values in class_counts.items()
        ]
        prediction_analysis[split] = {
            **dict(outcomes),
            "same_prediction_fraction": outcomes["same_prediction"] / outcomes["total"],
            "A3_minus_A0_accuracy": (outcomes["A3_only_correct"] - outcomes["A0_only_correct"]) / outcomes["total"],
            "participant_run_clusters": len(cluster_values),
            "cluster_bootstrap_ci95_low": percentile(bootstrap, 0.025),
            "cluster_bootstrap_ci95_high": percentile(bootstrap, 0.975),
            "top_class_gains": sorted(class_rows, key=lambda row: (-row["delta"], -row["support_clip_seed"]))[:8],
            "top_class_losses": sorted(class_rows, key=lambda row: (row["delta"], -row["support_clip_seed"]))[:8],
        }

    report = {
        "coverage": {experiment: {"completed": len(coverage[experiment]), "keys": coverage[experiment]} for experiment in EXPERIMENTS},
        "aggregate": aggregate,
        "participant_summary": participant_summary,
        "seed_summary": seed_summary,
        "paired": paired,
        "test_all_stage_paired": stages,
        "augmentation_audit": audit_summary,
        "training": training_summary,
        "A3_vs_A0_predictions": prediction_analysis,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
