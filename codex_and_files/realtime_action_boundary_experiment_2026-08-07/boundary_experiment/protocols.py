from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .annotations import RunInfo
from .utils import read_jsonl, write_json, write_jsonl


def _run_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["participant"]), str(row.get("run", row.get("source_run")))) for row in rows}


def prepare_boundary_protocols(
    run_index: dict[str, RunInfo],
    atomic_project_root: str | Path,
    output_root: str | Path,
    camera_id: str,
    participants: list[str],
) -> dict[str, Any]:
    atomic_project_root = Path(atomic_project_root)
    output_root = Path(output_root)
    by_key = {(info.participant, info.source_run): info for info in run_index.values()}
    report: dict[str, Any] = {"folds": {}}
    for heldout in participants:
        source_root = atomic_project_root / "outputs" / f"{heldout}_as_test" / f"cam_{camera_id}" / "protocols"
        report["folds"][heldout] = {}
        for scope in ("normal_only", "all_runs"):
            scope_report: dict[str, Any] = {}
            split_keys: dict[str, set[tuple[str, str]]] = {}
            for split in ("train", "test_normal", "test_fault", "test_all"):
                source = source_root / scope / f"{split}.jsonl"
                keys = _run_keys(read_jsonl(source))
                split_keys[split] = keys
                missing = sorted(keys - set(by_key))
                if missing:
                    raise KeyError(f"Structured dataset is missing run keys from {source}: {missing}")
                rows = [
                    {
                        "sample_name": by_key[key].sample_name,
                        "participant": key[0],
                        "source_run": key[1],
                        "split": split,
                        "train_scope": scope,
                        "heldout_participant": heldout,
                    }
                    for key in sorted(keys)
                ]
                target = output_root / f"{heldout}_as_test" / scope / f"{split}.jsonl"
                write_jsonl(target, rows)
                scope_report[split] = {"runs": len(rows), "path": str(target), "source": str(source)}
            if any(participant == heldout for participant, _ in split_keys["train"]):
                raise ValueError(f"LOSO leakage: held-out {heldout} appears in {scope} train")
            if any(participant != heldout for participant, _ in split_keys["test_all"]):
                raise ValueError(f"Non-held-out run appears in {heldout}/{scope} test_all")
            if split_keys["train"] & split_keys["test_all"]:
                raise ValueError(f"Train/test run overlap in {heldout}/{scope}")
            if split_keys["test_normal"] & split_keys["test_fault"]:
                raise ValueError(f"Normal/fault test overlap in {heldout}/{scope}")
            if split_keys["test_normal"] | split_keys["test_fault"] != split_keys["test_all"]:
                raise ValueError(f"Normal+fault does not equal test_all in {heldout}/{scope}")
            report["folds"][heldout][scope] = scope_report
    write_json(output_root / "protocol_report.json", report)
    return report


def load_protocol_runs(protocol_path: str | Path) -> list[str]:
    return [str(row["sample_name"]) for row in read_jsonl(protocol_path)]
