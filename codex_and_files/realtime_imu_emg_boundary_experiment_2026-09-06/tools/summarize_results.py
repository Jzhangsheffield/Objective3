from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from sensor_boundary.utils import read_json


def nested(row: dict, *keys, default=None):
    value = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LOSO metrics into one CSV")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.outputs_root)
    rows = []
    for path in root.glob("**/evaluation/*/metrics.json"):
        result = read_json(path)
        macro = result["macro"]
        parts = path.parts
        rows.append({
            "modality": next((x for x in ("imu", "emg") if x in parts), "unknown"),
            "heldout": next((x.removesuffix("_as_test") for x in parts if x.endswith("_as_test")), ""),
            "scope": next((x for x in ("normal_only", "all_runs") if x in parts), ""),
            "seed": next((x.removeprefix("seed_") for x in parts if x.startswith("seed_")), ""),
            "test_split": path.parent.name,
            "frame_accuracy": nested(macro, "frame_state", "accuracy"),
            "frame_f1": nested(macro, "frame_state", "f1"),
            "boundary_start_f1_200ms": nested(macro, "boundary", "200", "start", "f1"),
            "boundary_end_f1_200ms": nested(macro, "boundary", "200", "end", "f1"),
            "segmental_f1_50": nested(macro, "segmental_f1", "50", "f1"),
            "edit_score": nested(macro, "edit_score"),
            "emission_delay_ms": nested(macro, "emission_delay_ms"),
            "real_time_factor": result.get("overall_real_time_factor"),
        })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["modality"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["modality"], x["heldout"], x["seed"], x["scope"], x["test_split"])))
    print(f"Wrote {len(rows)} rows to {target}")


if __name__ == "__main__":
    main()
