from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.augmentation import TransitionModel, augment_history, stable_seed
from atomic_tail_exp.config import load_config, run_spec, write_json
from atomic_tail_exp.graph import TaskGraph, is_graph_valid


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit A0-A8 reorder policies without importing PyTorch.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participant", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scope", default="all_runs")
    parser.add_argument("--experiments", default="A1,A2,A3,A4,A5,A6,A7,A8")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_ids = [item.strip().upper() for item in args.experiments.split(",") if item.strip()]
    base_spec = run_spec(config, "A0", args.participant, args.seed, args.scope)
    paths = base_spec["paths"]
    graph = TaskGraph.load(paths["task_graph"], paths["relation_matrix"])
    rows = read_jsonl(Path(paths["protocol_root"]) / args.scope / "train.jsonl")
    transition = TransitionModel.fit(rows, graph.num_nodes, float(config["augmentation"]["transition_laplace"]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["participant"]), str(row["run"]))].append(row)
    histories = []
    for run_rows in grouped.values():
        run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
        histories.extend((current, run_rows[:index]) for index, current in enumerate(run_rows))
    report = {}
    for experiment_id in experiment_ids:
        experiment = config["experiments"][experiment_id]
        reasons = Counter()
        changed = 0
        invalid = 0
        distances = []
        for current, history in histories:
            result = augment_history(
                history,
                graph,
                stable_seed(args.seed, 0, str(current["sample_name"])),
                bool(experiment["active_tail_only"]),
                str(experiment["sampling"]),
                transition if experiment["sampling"] == "plausibility_weighted" else None,
                int(config["augmentation"]["candidate_count"]),
                float(config["augmentation"]["sampling_temperature"]),
                float(config["augmentation"]["max_normalized_kendall_distance"]),
                int(config["augmentation"]["min_changed_positions"]),
                int(config["augmentation"]["preserve_latest_non_tail"]),
            )
            reasons[result.decision.reason] += 1
            changed += int(result.changed)
            invalid += int(result.changed and not is_graph_valid(list(result.rows), graph))
            distances.append(result.normalized_kendall_distance)
        report[experiment_id] = {
            "samples": len(histories),
            "decision_reasons": dict(sorted(reasons.items())),
            "changed": changed,
            "changed_fraction": changed / max(1, len(histories)),
            "mean_normalized_kendall_distance": sum(distances) / max(1, len(distances)),
            "invalid_changed_graph_orders": invalid,
        }
    result = {"participant": args.participant, "seed": args.seed, "scope": args.scope, "experiments": report}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(item["invalid_changed_graph_orders"] for item in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
