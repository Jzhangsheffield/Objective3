from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

from phase_a.io import write_json
from phase_a.supplementary import (
    base_protocol_dir,
    experiment_spec,
    load_supplementary_config,
    supplementary_model_dir,
    validate_supplementary_condition,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction_path(config: dict, condition: str, participant: str, seed: int) -> Path:
    return supplementary_model_dir(config, condition, participant, seed) / "test_results" / "test_all_predictions.csv"


def one_metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> tuple[float, float, float]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return float((truth == prediction).mean()), float(f1[present].mean()), float(recall[present].min())


def calculate_metrics(arrays: dict[str, np.ndarray], node_available: bool) -> dict[str, float]:
    result = {}
    tier3 = one_metrics(arrays["true_tier3"], arrays["pred_tier3"], 31)
    result.update(tier3_accuracy=tier3[0], tier3_macro_f1=tier3[1], tier3_weakest_recall=tier3[2])
    for label, mask in (
        ("normal", ~arrays["fault"]), ("fault", arrays["fault"]),
        ("stage1", arrays["stage"] == 1), ("stage2", arrays["stage"] == 2),
        ("stage3", arrays["stage"] == 3),
    ):
        value = one_metrics(arrays["true_tier3"][mask], arrays["pred_tier3"][mask], 31)
        result[f"{label}_tier3_macro_f1"] = value[1]
    if node_available:
        node = one_metrics(arrays["true_node"], arrays["pred_node"], 35)
        result.update(node_accuracy=node[0], node_macro_f1=node[1], node_weakest_recall=node[2])
        for label, mask in (
            ("normal", ~arrays["fault"]), ("fault", arrays["fault"]),
            ("stage1", arrays["stage"] == 1), ("stage2", arrays["stage"] == 2),
            ("stage3", arrays["stage"] == 3),
        ):
            value = one_metrics(arrays["true_node"][mask], arrays["pred_node"][mask], 35)
            result[f"{label}_node_macro_f1"] = value[1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired clip bootstrap for two supplementary S conditions")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repetitions", type=int, default=None)
    args = parser.parse_args()
    config = load_supplementary_config(args.config)
    candidate = validate_supplementary_condition(config, args.candidate)
    baseline = validate_supplementary_condition(config, args.baseline)
    candidate_node = experiment_spec(config, candidate)["task"] != "direct_tier3"
    baseline_node = experiment_spec(config, baseline)["task"] != "direct_tier3"
    if candidate_node != baseline_node:
        raise ValueError("A paired comparison must use the same label spaces; do not compare direct_tier3 with node-capable models")
    node_available = candidate_node and baseline_node

    records: dict[tuple[str, str], dict] = {}
    fault_names = set()
    participants = config["base"]["participants"]
    seeds = config["base"]["seeds"]
    for participant in participants:
        with (base_protocol_dir(config, participant) / "test_fault.jsonl").open("r", encoding="utf-8") as handle:
            fault_names.update(json.loads(line)["sample_name"] for line in handle if line.strip())
        for seed in seeds:
            base_rows = {row["sample_name"]: row for row in read_csv(prediction_path(config, baseline, participant, seed))}
            candidate_rows = {row["sample_name"]: row for row in read_csv(prediction_path(config, candidate, participant, seed))}
            if set(base_rows) != set(candidate_rows):
                raise ValueError(f"Unpaired predictions: {participant}, seed={seed}")
            for name, base_row in base_rows.items():
                candidate_row = candidate_rows[name]
                if (base_row["true_node_idx"], base_row["true_tier3_id"]) != (
                    candidate_row["true_node_idx"], candidate_row["true_tier3_id"]
                ):
                    raise ValueError(f"Ground-truth mismatch for {name}")
                item = records.setdefault((participant, name), {
                    "participant": participant,
                    "sample_name": name,
                    "true_node": int(base_row["true_node_idx"]) - 1,
                    "true_tier3": int(base_row["true_tier3_id"]),
                    "stage": int(base_row["stage_id"]),
                    "baseline": {}, "candidate": {},
                })
                item["baseline"][seed] = {
                    "node": int(base_row["pred_node_idx"]) - 1 if node_available else None,
                    "tier3": int(base_row["pred_tier3_id"]),
                }
                item["candidate"][seed] = {
                    "node": int(candidate_row["pred_node_idx"]) - 1 if node_available else None,
                    "tier3": int(candidate_row["pred_tier3_id"]),
                }
    items = list(records.values())
    participant_indices = {
        participant: np.array([index for index, row in enumerate(items) if row["participant"] == participant])
        for participant in participants
    }

    def arrays(indices: np.ndarray, model: str) -> dict[str, np.ndarray]:
        expanded = [(index, seed) for index in indices for seed in seeds]
        result = {
            "true_node": np.array([items[index]["true_node"] for index, _ in expanded]),
            "true_tier3": np.array([items[index]["true_tier3"] for index, _ in expanded]),
            "pred_tier3": np.array([items[index][model][seed]["tier3"] for index, seed in expanded]),
            "fault": np.array([items[index]["sample_name"] in fault_names for index, _ in expanded]),
            "stage": np.array([items[index]["stage"] for index, _ in expanded]),
        }
        if node_available:
            result["pred_node"] = np.array([items[index][model][seed]["node"] for index, seed in expanded])
        return result

    all_indices = np.arange(len(items))
    candidate_point = calculate_metrics(arrays(all_indices, "candidate"), node_available)
    baseline_point = calculate_metrics(arrays(all_indices, "baseline"), node_available)
    names = list(candidate_point)
    point = {name: candidate_point[name] - baseline_point[name] for name in names}
    repetitions = int(args.repetitions or config["base"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config["base"]["bootstrap_seed"]))
    samples = {name: np.empty(repetitions, dtype=float) for name in names}
    for repetition in range(repetitions):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in participant_indices.values()
        ])
        candidate_values = calculate_metrics(arrays(sampled, "candidate"), node_available)
        baseline_values = calculate_metrics(arrays(sampled, "baseline"), node_available)
        for name in names:
            samples[name][repetition] = candidate_values[name] - baseline_values[name]
    report = {
        "candidate": candidate,
        "baseline": baseline,
        "unit": "unique clip within participant; each sampled clip expands over all three seeds",
        "participant_stratified": True,
        "unique_clips": len(items),
        "repetitions": repetitions,
        "node_metrics_available": node_available,
        "metrics": {
            name: {
                "delta": float(point[name]),
                "ci95_low": float(np.percentile(samples[name], 2.5)),
                "ci95_high": float(np.percentile(samples[name], 97.5)),
                "probability_positive": float((samples[name] > 0).mean()),
            }
            for name in names
        },
    }
    summary = Path(config["output_root"]) / "summary"
    output = summary / f"paired_bootstrap_{candidate}_vs_{baseline}.json"
    write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
