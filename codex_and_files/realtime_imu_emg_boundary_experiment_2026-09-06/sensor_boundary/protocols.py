from __future__ import annotations

from pathlib import Path
from typing import Any

from .annotations import RunInfo
from .utils import read_jsonl, write_json, write_jsonl


SPLITS = ("train", "test_normal", "test_fault", "test_all")
SCOPES = ("normal_only", "all_runs")


def prepare_protocols(run_index: dict[str, RunInfo], source_root: str | Path, output_root: str | Path, participants: list[str]) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    report: dict[str, Any] = {"source": str(source_root), "folds": {}}
    for heldout in participants:
        report["folds"][heldout] = {}
        for scope in SCOPES:
            sets: dict[str, set[str]] = {}
            report["folds"][heldout][scope] = {}
            for split in SPLITS:
                source = source_root / f"{heldout}_as_test" / scope / f"{split}.jsonl"
                rows = read_jsonl(source)
                names = [str(row["sample_name"]) for row in rows]
                missing = sorted(set(names) - set(run_index))
                if missing:
                    raise KeyError(f"Runs missing from dataset for {source}: {missing}")
                normalized = [
                    {
                        "sample_name": name,
                        "participant": run_index[name].participant,
                        "source_run": run_index[name].source_run,
                        "split": split,
                        "train_scope": scope,
                        "heldout_participant": heldout,
                    }
                    for name in names
                ]
                target = output_root / f"{heldout}_as_test" / scope / f"{split}.jsonl"
                write_jsonl(target, normalized)
                sets[split] = set(names)
                report["folds"][heldout][scope][split] = len(names)
            if any(run_index[name].participant == heldout for name in sets["train"]):
                raise ValueError(f"LOSO leakage in {heldout}/{scope} train")
            if any(run_index[name].participant != heldout for name in sets["test_all"]):
                raise ValueError(f"Non-heldout run in {heldout}/{scope} test")
            if sets["train"] & sets["test_all"]:
                raise ValueError(f"Train/test overlap in {heldout}/{scope}")
            if sets["test_normal"] | sets["test_fault"] != sets["test_all"]:
                raise ValueError(f"normal + fault != all in {heldout}/{scope}")
    write_json(output_root / "protocol_report.json", report)
    return report


def load_protocol_runs(path: str | Path, limit: int = 0) -> list[str]:
    names = [str(row["sample_name"]) for row in read_jsonl(path)]
    return names if limit <= 0 else names[:limit]
