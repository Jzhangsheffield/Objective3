from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sequence_disjoint_exp.common import (
    group_runs,
    load_package_config,
    read_json,
    read_jsonl,
    resolve_paths,
    write_json,
)


def history_prefixes(rows: list[dict[str, Any]], minimum_length: int) -> dict[tuple[int, ...], list[str]]:
    result: dict[tuple[int, ...], list[str]] = {}
    for (participant, run), run_rows in group_runs(rows).items():
        nodes = [int(row["node_idx"]) for row in run_rows]
        for current_index in range(max(1, minimum_length), len(nodes)):
            prefix = tuple(nodes[:current_index])
            result.setdefault(prefix, []).append(f"{participant}|{run}|target_position_{current_index + 1}")
    return result


def refresh_rounds(interval: str | int, epochs: int) -> list[int]:
    if str(interval).lower() == "once":
        return [0]
    numeric = max(1, int(interval))
    return list(range((max(1, epochs) - 1) // numeric + 1))


def audit_experiment(
    config: dict[str, Any], participant: str, definition: dict[str, Any]
) -> dict[str, Any]:
    paths = resolve_paths(config, participant, int(config["grid"]["seeds"][0]))
    legacy_root = Path(paths["legacy_atomic_package_root"])
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from atomic_tail_exp.augmentation import augment_history, stable_seed
    from atomic_tail_exp.graph import TaskGraph

    scope_root = Path(paths["protocol_root"]) / str(config["grid"]["train_scope"])
    train_rows = read_jsonl(scope_root / "train.jsonl")
    test_rows = read_jsonl(scope_root / "test_all.jsonl")
    minimum_length = int(config["augmentation_coverage_audit"]["minimum_history_length"])
    test_prefix_map = history_prefixes(test_rows, minimum_length)
    test_prefixes = set(test_prefix_map)
    train_runs = group_runs(train_rows)
    actual_prefixes: set[tuple[int, ...]] = set()
    augmented_prefixes: set[tuple[int, ...]] = set()
    generated = 0
    changed = 0
    reason_counts: Counter[str] = Counter()
    legacy_config = read_json(paths["legacy_atomic_config"])
    aug = legacy_config["augmentation"]
    base_experiment = legacy_config["experiments"][str(definition["base_experiment"])]
    interval = definition["refresh_interval"]
    graph = TaskGraph.load(paths["task_graph"], paths["relation_matrix"])
    rounds = refresh_rounds(interval, int(config["augmentation_coverage_audit"]["epochs"]))

    for run_rows in train_runs.values():
        for current_index in range(1, len(run_rows)):
            actual_history = list(run_rows[:current_index])
            actual_prefixes.add(tuple(int(row["node_idx"]) for row in actual_history))
            current_sample = str(run_rows[current_index]["sample_name"])
            for seed in config["grid"]["seeds"]:
                for refresh_round in rounds:
                    result = augment_history(
                        actual_history,
                        graph,
                        stable_seed(int(seed), int(refresh_round), current_sample),
                        bool(base_experiment["active_tail_only"]),
                        str(base_experiment["sampling"]),
                        None,
                        int(aug["candidate_count"]),
                        float(aug["sampling_temperature"]),
                        float(aug["max_normalized_kendall_distance"]),
                        int(aug["min_changed_positions"]),
                        int(aug["preserve_latest_non_tail"]),
                    )
                    generated += 1
                    changed += int(result.changed)
                    reason_counts[result.decision.reason] += 1
                    augmented_prefixes.add(tuple(int(row["node_idx"]) for row in result.rows))

    actual_covered = actual_prefixes.intersection(test_prefixes)
    augmented_covered = augmented_prefixes.intersection(test_prefixes)
    newly_covered = augmented_covered - actual_covered
    lost_actual_coverage = actual_covered - augmented_covered
    output = {
        "experiment_id": definition["id"],
        "source_experiment_id": definition["base_experiment"],
        "participant": participant,
        "refresh_interval": interval,
        "refresh_rounds": rounds,
        "active_tail_only": bool(base_experiment["active_tail_only"]),
        "position_mode": base_experiment["position_mode"],
        "seeds": list(config["grid"]["seeds"]),
        "comparison_unit": "node-order history prefix before each current clip",
        "important_interpretation": "The legacy augmenter reorders each sample history independently; it does not synthesize one coherent replacement run.",
        "test_guided_generation": False,
        "test_prefixes": len(test_prefixes),
        "actual_train_unique_prefixes": len(actual_prefixes),
        "augmented_train_unique_prefixes": len(augmented_prefixes),
        "generated_augmented_histories": generated,
        "changed_augmented_histories": changed,
        "changed_fraction": changed / max(1, generated),
        "actual_test_prefixes_covered": len(actual_covered),
        "actual_test_prefix_coverage": len(actual_covered) / max(1, len(test_prefixes)),
        "augmented_test_prefixes_covered": len(augmented_covered),
        "augmented_test_prefix_coverage": len(augmented_covered) / max(1, len(test_prefixes)),
        "test_prefixes_newly_covered_by_augmentation": len(newly_covered),
        "actual_test_prefixes_not_present_in_augmented_view": len(lost_actual_coverage),
        "tail_reason_counts": dict(sorted(reason_counts.items())),
        "covered_test_prefixes": [
            {
                "node_sequence": list(prefix),
                "test_occurrences": test_prefix_map[prefix],
                "already_in_actual_training_prefixes": prefix in actual_prefixes,
            }
            for prefix in sorted(augmented_covered, key=lambda value: (len(value), value))
        ],
    }
    safe_id = str(definition["id"]).replace("-", "_").lower()
    write_json(Path(paths["protocol_root"]) / f"augmentation_coverage_{safe_id}.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Posthoc audit of A1/A3-DualPos graph-valid history coverage")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participants", default=None)
    parser.add_argument("--experiments", default=None)
    args = parser.parse_args()
    config = load_package_config(args.config)
    participants = (
        [value.strip() for value in args.participants.split(",") if value.strip()]
        if args.participants else list(config["grid"]["participants"])
    )
    requested = (
        [value.strip() for value in args.experiments.split(",") if value.strip()]
        if args.experiments else list(config["augmentation_coverage_audit"]["experiment_ids"])
    )
    definitions = {str(item["id"]): item for item in config["experiments"]}
    unknown = [value for value in requested if value not in definitions]
    if unknown:
        raise ValueError(f"Unknown audit experiments: {unknown}")
    all_results = []
    for participant in participants:
        for experiment_id in requested:
            result = audit_experiment(config, participant, definitions[experiment_id])
            all_results.append(result)
            print(
                f"[{participant}/{experiment_id}] changed={result['changed_fraction']:.3f}, "
                f"test prefix coverage "
                f"actual={result['actual_test_prefix_coverage']:.3f}, "
                f"augmented={result['augmented_test_prefix_coverage']:.3f}, "
                f"new={result['test_prefixes_newly_covered_by_augmentation']}"
            )
    write_json(PACKAGE_ROOT / "inputs" / "augmentation_coverage_all_folds.json", {"results": all_results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
