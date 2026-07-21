from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_jsonl, run_key, write_json, write_jsonl


PARTICIPANTS = ("A", "D", "J", "M")


def find_fault_manifest(dataset_root: str | Path, participant: str) -> Path:
    folder = Path(dataset_root) / f"{participant}_as_test"
    candidates = (
        folder / "fault_run_test_manifest.jsonl",
        folder / "falut_run_test_manifest.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No fault-run manifest found in {folder}")


def load_global_fault_run_keys(dataset_root: str | Path) -> set[tuple[str, str]]:
    fault_keys: set[tuple[str, str]] = set()
    for participant in PARTICIPANTS:
        for row in read_jsonl(find_fault_manifest(dataset_root, participant)):
            fault_keys.add(run_key(row))
    return fault_keys


def _validate_rows(rows: list[dict[str, Any]], source: str) -> None:
    required = {
        "sample_name",
        "participant",
        "run",
        "annotation_row_index",
        "tier3_id",
        "node_id",
        "node_idx",
        "stage_id",
    }
    missing_examples: list[str] = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            missing_examples.append(f"{row.get('sample_name', '<unknown>')}: {missing}")
            if len(missing_examples) >= 5:
                break
    if missing_examples:
        raise ValueError(f"Manifest {source} is missing graph fields: {missing_examples}")


def _sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["participant"]),
            str(row["run"]),
            int(row["annotation_row_index"]),
        ),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_keys = {run_key(row) for row in rows}
    return {
        "samples": len(rows),
        "runs": len(run_keys),
        "participants": sorted({str(row["participant"]) for row in rows}),
    }


def prepare_protocols(
    dataset_root: str | Path,
    output_root: str | Path,
    test_participant: str = "J",
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_root = ensure_dir(output_root)
    fold_folder = dataset_root / f"{test_participant}_as_test"
    train_all = read_jsonl(fold_folder / "train_manifest.jsonl")
    test_all = read_jsonl(fold_folder / "test_manifest.jsonl")
    fault_rows = read_jsonl(find_fault_manifest(dataset_root, test_participant))
    _validate_rows(train_all, str(fold_folder / "train_manifest.jsonl"))
    _validate_rows(test_all, str(fold_folder / "test_manifest.jsonl"))

    global_fault_keys = load_global_fault_run_keys(dataset_root)
    heldout_fault_keys = {run_key(row) for row in fault_rows}
    train_normal = [row for row in train_all if run_key(row) not in global_fault_keys]
    test_normal = [row for row in test_all if run_key(row) not in heldout_fault_keys]
    test_fault = [row for row in test_all if run_key(row) in heldout_fault_keys]

    if {row["sample_name"] for row in test_fault} != {row["sample_name"] for row in fault_rows}:
        raise ValueError("Held-out fault manifest is not an exact subset of test_manifest")

    protocol_rows = {
        "normal_only": {
            "train": _sorted(train_normal),
            "test_all": _sorted(test_all),
            "test_normal": _sorted(test_normal),
            "test_fault": _sorted(test_fault),
        },
        "all_runs": {
            "train": _sorted(train_all),
            "test_all": _sorted(test_all),
            "test_normal": _sorted(test_normal),
            "test_fault": _sorted(test_fault),
        },
    }

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "test_participant": test_participant,
        "global_fault_runs": sorted([f"{p}|{run}" for p, run in global_fault_keys]),
        "protocols": {},
    }
    for scope, splits in protocol_rows.items():
        scope_root = ensure_dir(output_root / scope)
        report["protocols"][scope] = {}
        for split_name, rows in splits.items():
            path = scope_root / f"{split_name}.jsonl"
            write_jsonl(path, rows)
            report["protocols"][scope][split_name] = {
                "path": str(path),
                **_summary(rows),
            }

    write_json(output_root / "protocol_report.json", report)
    return report

