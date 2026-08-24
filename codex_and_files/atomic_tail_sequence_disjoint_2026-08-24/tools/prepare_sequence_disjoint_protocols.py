from __future__ import annotations

import argparse
import csv
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
    manifest_summary,
    read_jsonl,
    resolve_paths,
    sequence_hash,
    sequence_values,
    write_json,
    write_jsonl,
)


def prepare_fold(config: dict[str, Any], participant: str, overwrite: bool) -> dict[str, Any]:
    paths = resolve_paths(config, participant=participant, seed=int(config["grid"]["seeds"][0]))
    source_root = Path(paths["source_protocol_root"])
    output_root = Path(paths["protocol_root"])
    scope = str(config["grid"]["train_scope"])
    source_scope = source_root / scope
    output_scope = output_root / scope
    expected_outputs = [output_scope / f"{name}.jsonl" for name in ("train", *config["grid"]["test_splits"])]
    report_path = output_root / "sequence_disjoint_report.json"
    if any(path.exists() for path in (*expected_outputs, report_path)) and not overwrite:
        raise FileExistsError(f"Fold {participant} already has generated protocols; pass --overwrite to replace them")

    train_rows = read_jsonl(source_scope / "train.jsonl")
    normal_train_manifest = source_root / "normal_only" / "train.jsonl"
    normal_train_keys = (
        set(group_runs(read_jsonl(normal_train_manifest)))
        if normal_train_manifest.is_file() else set()
    )
    test_rows_by_split = {
        name: read_jsonl(source_scope / f"{name}.jsonl")
        for name in config["grid"]["test_splits"]
    }
    test_all_rows = test_rows_by_split[str(config["sequence_isolation"]["source_test_split"])]
    train_runs = group_runs(train_rows)
    test_runs = group_runs(test_all_rows)
    field = str(config["sequence_isolation"]["signature_field"])
    collapse = bool(config["sequence_isolation"]["collapse_consecutive_duplicates"])
    train_signatures = {
        key: sequence_values(rows, field, collapse) for key, rows in train_runs.items()
    }
    test_signatures = {
        key: sequence_values(rows, field, collapse) for key, rows in test_runs.items()
    }
    test_signature_set = set(test_signatures.values())
    removed_keys = {key for key, signature in train_signatures.items() if signature in test_signature_set}
    retained_train_rows = [row for row in train_rows if (str(row["participant"]), str(row["run"])) not in removed_keys]
    retained_runs = group_runs(retained_train_rows)
    retained_signatures = {
        sequence_values(rows, field, collapse) for rows in retained_runs.values()
    }
    remaining_overlap = retained_signatures.intersection(test_signature_set)
    if remaining_overlap:
        raise RuntimeError(f"Sequence isolation failed for fold {participant}: {len(remaining_overlap)} signatures remain")

    output_scope.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_scope / "train.jsonl", retained_train_rows)
    for split_name, rows in test_rows_by_split.items():
        write_jsonl(output_scope / f"{split_name}.jsonl", rows)

    tier3_train_signatures = {
        key: sequence_values(rows, "tier3_id", collapse) for key, rows in train_runs.items()
    }
    tier3_test_signature_set = {
        sequence_values(rows, "tier3_id", collapse) for rows in test_runs.values()
    }
    test_fault_keys = set(group_runs(test_rows_by_split["test_fault"]))
    index_rows: list[dict[str, Any]] = []
    for role, grouped, signatures in (
        ("train", train_runs, train_signatures),
        ("test", test_runs, test_signatures),
    ):
        for key, rows in grouped.items():
            signature = signatures[key]
            tier3_signature = sequence_values(rows, "tier3_id", collapse)
            run_type = (
                ("normal" if key in normal_train_keys else "fault")
                if role == "train" else
                ("fault" if key in test_fault_keys else "normal")
            )
            index_rows.append({
                "role": role,
                "participant": key[0],
                "run": key[1],
                "samples": len(rows),
                "run_type": run_type,
                "node_signature_hash": sequence_hash(signature),
                "node_sequence": " ".join(map(str, signature)),
                "tier3_signature_hash": sequence_hash(tier3_signature),
                "tier3_sequence": " ".join(map(str, tier3_signature)),
                "exact_node_overlap": role == "train" and signature in test_signature_set,
                "exact_tier3_overlap": role == "train" and tier3_train_signatures[key] in tier3_test_signature_set,
                "retained_for_training": role == "test" or key not in removed_keys,
            })
    index_path = output_root / "run_sequence_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    removed_by_participant = Counter(key[0] for key in removed_keys)
    removed_by_type = Counter("normal" if key in normal_train_keys else "fault" for key in removed_keys)
    retained_by_type = Counter("normal" if key in normal_train_keys else "fault" for key in retained_runs)
    report = {
        "test_participant": participant,
        "camera_id": config["grid"]["camera_id"],
        "scope": scope,
        "sequence_definition": {
            "field": field,
            "sort_key": "annotation_row_index",
            "collapse_consecutive_duplicates": collapse,
            "matching": "exact_full_run_sequence",
        },
        "augmentation_policy": {
            "may_naturally_generate_test_order": True,
            "test_sequence_is_not_read_by_sampler": True,
            "test_guided_resampling": False,
        },
        "source_train": manifest_summary(train_rows),
        "filtered_train": manifest_summary(retained_train_rows),
        "test_splits": {name: manifest_summary(rows) for name, rows in test_rows_by_split.items()},
        "sequence_counts": {
            "train_runs_before": len(train_runs),
            "train_unique_node_sequences_before": len(set(train_signatures.values())),
            "test_runs": len(test_runs),
            "test_unique_node_sequences": len(test_signature_set),
            "overlapping_unique_node_sequences_before": len(set(train_signatures.values()).intersection(test_signature_set)),
            "removed_train_runs": len(removed_keys),
            "removed_train_runs_by_participant": dict(sorted(removed_by_participant.items())),
            "removed_train_runs_by_type": dict(sorted(removed_by_type.items())),
            "retained_train_runs_by_type": dict(sorted(retained_by_type.items())),
            "retained_train_runs": len(retained_runs),
            "remaining_exact_node_overlap": len(remaining_overlap),
            "tier3_overlap_train_runs_before": sum(
                signature in tier3_test_signature_set for signature in tier3_train_signatures.values()
            ),
        },
        "removed_train_runs": [f"{person}|{run}" for person, run in sorted(removed_keys)],
        "outputs": {
            "protocol_root": str(output_root),
            "run_sequence_index": str(index_path),
        },
    }
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build run-level sequence-disjoint LOSO manifests")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participants", default=None, help="Comma-separated folds; defaults to config grid")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_package_config(args.config)
    participants = (
        [value.strip() for value in args.participants.split(",") if value.strip()]
        if args.participants else list(config["grid"]["participants"])
    )
    reports = []
    for participant in participants:
        report = prepare_fold(config, participant, args.overwrite)
        reports.append(report)
        counts = report["sequence_counts"]
        print(
            f"[{participant}] train runs {counts['train_runs_before']} -> {counts['retained_train_runs']}; "
            f"removed={counts['removed_train_runs']}; remaining_overlap={counts['remaining_exact_node_overlap']}"
        )
    write_json(PACKAGE_ROOT / "inputs" / "sequence_disjoint_all_folds_report.json", {"folds": reports})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
