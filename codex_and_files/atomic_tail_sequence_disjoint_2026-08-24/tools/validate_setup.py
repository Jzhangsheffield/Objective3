from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sequence_disjoint_exp.common import (
    group_runs,
    load_package_config,
    read_json,
    read_jsonl,
    resolve_paths,
    sequence_values,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sequence-disjoint package inputs without training")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    args = parser.parse_args()
    config = load_package_config(args.config)
    errors: list[str] = []
    warnings: list[str] = []
    expected_ids = set(config["grid"]["default_experiments"])
    configured_ids = {str(item["id"]) for item in config["experiments"]}
    if expected_ids - configured_ids:
        errors.append(f"Default experiments are undefined: {sorted(expected_ids - configured_ids)}")
    m2 = next(item for item in config["experiments"] if item["id"] == "M2-Direct-RealOrder")
    if m2["train_order"] != "actual" or m2.get("refresh_interval") is not None:
        errors.append("M2-Direct-RealOrder must use actual order and no refresh interval")
    for definition in config["experiments"]:
        if definition["id"] != "M2-Direct-RealOrder" and definition["base_experiment"] not in {"A1", "A3-DualPos"}:
            errors.append(f"Unexpected augmented base experiment: {definition}")

    for participant in config["grid"]["participants"]:
        for seed in config["grid"]["seeds"]:
            paths = resolve_paths(config, participant, int(seed))
            for key in (
                "legacy_atomic_config", "legacy_train_backbone", "legacy_extract_features",
                "task_graph", "relation_matrix",
            ):
                if not Path(paths[key]).is_file():
                    errors.append(f"{participant}/seed_{seed} missing {key}: {paths[key]}")
            for key in ("backbone_checkpoint", "train_cache", "test_cache"):
                if not Path(paths[key]).is_file():
                    warnings.append(
                        f"{participant}/seed_{seed} planned upstream artifact not generated yet: {paths[key]}"
                    )
        paths = resolve_paths(config, participant, int(config["grid"]["seeds"][0]))
        protocol_root = Path(paths["protocol_root"])
        scope_root = protocol_root / str(config["grid"]["train_scope"])
        required = [scope_root / "train.jsonl", *(scope_root / f"{name}.jsonl" for name in config["grid"]["test_splits"])]
        for path in required:
            if not path.is_file():
                errors.append(f"{participant} missing generated manifest: {path}")
        report_path = protocol_root / "sequence_disjoint_report.json"
        if not report_path.is_file():
            errors.append(f"{participant} missing isolation report: {report_path}")
            continue
        report = read_json(report_path)
        if int(report["sequence_counts"]["remaining_exact_node_overlap"]) != 0:
            errors.append(f"{participant} isolation report still has node sequence overlap")
        if all(path.is_file() for path in required):
            collapse = bool(config["sequence_isolation"]["collapse_consecutive_duplicates"])
            train_sequences = {
                sequence_values(rows, "node_idx", collapse)
                for rows in group_runs(read_jsonl(scope_root / "train.jsonl")).values()
            }
            test_sequences = {
                sequence_values(rows, "node_idx", collapse)
                for rows in group_runs(read_jsonl(scope_root / "test_all.jsonl")).values()
            }
            overlap = train_sequences.intersection(test_sequences)
            if overlap:
                errors.append(f"{participant} recomputed overlap={len(overlap)}")
            filtered = report["filtered_train"]
            if filtered["missing_node_idx"]:
                warnings.append(f"{participant} filtered train misses nodes {filtered['missing_node_idx']}")
            if filtered["missing_tier3_id"]:
                warnings.append(f"{participant} filtered train misses Tier-3 IDs {filtered['missing_tier3_id']}")

    first_paths = resolve_paths(
        config, str(config["grid"]["participants"][0]), int(config["grid"]["seeds"][0])
    )
    if not Path(first_paths["dataset_root"]).is_dir():
        warnings.append(
            f"Configured dataset_root is unavailable here; override it when running upstream: {first_paths['dataset_root']}"
        )

    print(f"Validation completed: errors={len(errors)}, warnings={len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
