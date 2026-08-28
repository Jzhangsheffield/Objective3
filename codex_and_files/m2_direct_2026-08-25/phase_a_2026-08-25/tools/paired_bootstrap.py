from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

from phase_a.config import load_config, validate_condition
from phase_a.io import write_json
from phase_a.paths import a0_result_dir, model_dir, protocol_dir


def prediction_path(config, condition, participant, seed, split="test_all") -> Path:
    if condition == "A0":
        return a0_result_dir(config, participant, seed) / f"{split}_predictions.csv"
    return model_dir(config, condition, participant, seed) / "test_results" / f"{split}_predictions.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(true_node, pred_node, true_tier3, pred_tier3, fault, stage):
    def one(truth, prediction, classes):
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (truth, prediction), 1)
        tp = np.diag(matrix).astype(float); support = matrix.sum(1); predicted = matrix.sum(0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        present = support > 0
        return float((truth == prediction).mean()), float(f1[present].mean()), float(recall[present].min())
    node = one(true_node, pred_node, 35); tier3 = one(true_tier3, pred_tier3, 31)
    normal_node = one(true_node[~fault], pred_node[~fault], 35)
    fault_node = one(true_node[fault], pred_node[fault], 35)
    stage_node = [one(true_node[stage == value], pred_node[stage == value], 35) for value in (1, 2, 3)]
    return np.array([node[0], node[1], node[2], tier3[0], tier3[1],
                     normal_node[1], fault_node[0], fault_node[1],
                     stage_node[0][1], stage_node[1][1], stage_node[2][1]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired clip bootstrap, with the same resampled clip across three seeds")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--repetitions", type=int, default=None)
    args = parser.parse_args()
    condition = validate_condition(args.condition)
    if condition == "A0": raise ValueError("Bootstrap compares A1-A7 against A0")
    config = load_config(args.config)
    records = {}
    fault_names = set()
    for participant in config["participants"]:
        with (protocol_dir(config, participant) / "test_fault.jsonl").open("r", encoding="utf-8") as handle:
            import json
            fault_names.update(json.loads(line)["sample_name"] for line in handle if line.strip())
        for seed in config["seeds"]:
            base = {row["sample_name"]: row for row in read_csv(prediction_path(config, "A0", participant, seed))}
            candidate = {row["sample_name"]: row for row in read_csv(prediction_path(config, condition, participant, seed))}
            if set(base) != set(candidate):
                raise ValueError(f"Unpaired predictions: {participant}, seed {seed}")
            for name in base:
                key = (participant, name)
                item = records.setdefault(key, {"participant": participant, "sample_name": name,
                    "true_node": int(base[name]["true_node_idx"]) - 1,
                    "true_tier3": int(base[name]["true_tier3_id"]), "stage": int(base[name]["stage_id"]),
                    "base": {}, "candidate": {}})
                item["base"][seed] = (int(base[name]["pred_node_idx"]) - 1, int(base[name]["pred_tier3_id"]))
                item["candidate"][seed] = (int(candidate[name]["pred_node_idx"]) - 1, int(candidate[name]["pred_tier3_id"]))
    items = list(records.values())
    participant_indices = {participant: np.array([i for i, row in enumerate(items) if row["participant"] == participant])
                           for participant in config["participants"]}
    def arrays(indices, model):
        expanded = [(index, seed) for index in indices for seed in config["seeds"]]
        true_node = np.array([items[i]["true_node"] for i, _ in expanded])
        true_tier3 = np.array([items[i]["true_tier3"] for i, _ in expanded])
        pred_node = np.array([items[i][model][seed][0] for i, seed in expanded])
        pred_tier3 = np.array([items[i][model][seed][1] for i, seed in expanded])
        fault = np.array([items[i]["sample_name"] in fault_names for i, _ in expanded])
        stage = np.array([items[i]["stage"] for i, _ in expanded])
        return metrics(true_node, pred_node, true_tier3, pred_tier3, fault, stage)
    all_indices = np.arange(len(items))
    point = arrays(all_indices, "candidate") - arrays(all_indices, "base")
    repetitions = args.repetitions or config["bootstrap_repetitions"]
    rng = np.random.default_rng(config["bootstrap_seed"])
    deltas = np.empty((repetitions, len(point)), dtype=float)
    for repetition in range(repetitions):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                  for indices in participant_indices.values()])
        deltas[repetition] = arrays(sampled, "candidate") - arrays(sampled, "base")
    names = ["node_accuracy", "node_macro_f1", "node_weakest_recall",
             "tier3_accuracy", "tier3_macro_f1", "normal_node_macro_f1",
             "fault_node_accuracy", "fault_node_macro_f1",
             "stage1_node_macro_f1", "stage2_node_macro_f1", "stage3_node_macro_f1"]
    report = {
        "condition": condition, "baseline": "A0", "unit": "clip",
        "seed_handling": "same resampled clip is expanded over seeds 1/2/42",
        "participant_stratified": True, "unique_clips": len(items), "repetitions": repetitions,
        "metrics": {name: {"delta": float(point[i]),
                           "ci95_low": float(np.percentile(deltas[:, i], 2.5)),
                           "ci95_high": float(np.percentile(deltas[:, i], 97.5)),
                           "probability_positive": float((deltas[:, i] > 0).mean())}
                    for i, name in enumerate(names)},
    }
    output = Path(config["output_root"]) / "summary" / f"paired_bootstrap_{condition}_vs_A0.json"
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
