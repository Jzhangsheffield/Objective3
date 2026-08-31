from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.io import read_json, write_csv, write_json
from phase_b.paths import b0_condition_root, experiment_root


def result_root(config: dict, condition: str, participant: str, seed: int) -> Path:
    if condition == "B0":
        return b0_condition_root(config, "A2", participant, seed) / "test_results"
    return experiment_root(config, condition, participant, seed) / "test_results"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize B0-B5 without best-seed selection")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    conditions = list(config["experiments"])
    run_rows, missing = [], []
    for condition in conditions:
        for participant in config["participants"]:
            for seed in config["seeds"]:
                for split in config["evaluation"]["splits"]:
                    path = result_root(config, condition, participant, seed) / f"{split}_metrics.json"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    value = read_json(path)
                    run_rows.append({
                        "condition": condition, "participant": participant, "seed": seed, "split": split,
                        "samples": value["samples"], "node_accuracy": value["node"]["accuracy"],
                        "node_macro_f1": value["node"]["macro_f1"],
                        "node_weakest_recall": value["node"]["weakest_class_recall"],
                        "tier3_accuracy": value["tier3"]["accuracy"],
                        "tier3_macro_f1": value["tier3"]["macro_f1"],
                    })
    summary = []
    for condition in conditions:
        for split in config["evaluation"]["splits"]:
            selected = [row for row in run_rows if row["condition"] == condition and row["split"] == split]
            summary.append({
                "condition": condition, "split": split, "completed_runs": len(selected),
                "expected_runs": len(config["participants"]) * len(config["seeds"]),
                **{key: mean([float(row[key]) for row in selected]) for key in (
                    "node_accuracy", "node_macro_f1", "node_weakest_recall",
                    "tier3_accuracy", "tier3_macro_f1",
                )},
            })
    output = Path(config["output_root"]) / "summary"
    write_csv(output / "fold_seed_metrics.csv", run_rows)
    write_csv(output / "condition_summary.csv", summary)
    write_json(output / "completeness.json", {
        "complete": not missing, "found_metric_files": len(run_rows),
        "expected_metric_files": len(conditions) * len(config["participants"]) * len(config["seeds"])
        * len(config["evaluation"]["splits"]), "missing": missing,
    })
    print(f"Found {len(run_rows)} metrics; missing {len(missing)}. Saved to {output}")


if __name__ == "__main__":
    main()
