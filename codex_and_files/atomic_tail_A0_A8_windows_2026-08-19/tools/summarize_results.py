from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.config import load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize completed A0-A8 runs.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--split", default="test_all", choices=["test_normal", "test_fault", "test_all"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(str(config["paths"]["output_root"]).format(package_root=config["_package_root"]))
    summary_root = Path(args.output) if args.output else output_root / "summary"
    rows = []
    for path in output_root.glob("A*/**/completed.json"):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        metrics = item.get("metrics", {}).get(args.split)
        if not metrics:
            continue
        rows.append({
            "experiment": item["experiment_id"],
            "scope": item["scope"],
            "participant": item["participant"],
            "seed": int(item["seed"]),
            "split": args.split,
            "node_accuracy": metrics["node"]["accuracy"],
            "node_macro_f1": metrics["node"]["macro_f1"],
            "tier3_accuracy": metrics["tier3"]["accuracy"],
            "tier3_macro_f1": metrics["tier3"]["macro_f1"],
            "completed_json": str(path),
        })
    summary_root.mkdir(parents=True, exist_ok=True)
    raw_path = summary_root / f"{args.split}_runs.csv"
    if rows:
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["scope"], row["experiment"], row["participant"], row["seed"])))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["experiment"])].append(row)
    aggregates = []
    for (scope, experiment), group in sorted(grouped.items()):
        entry = {"scope": scope, "experiment": experiment, "runs": len(group)}
        for metric in ("node_accuracy", "node_macro_f1", "tier3_accuracy", "tier3_macro_f1"):
            values = [float(row[metric]) for row in group]
            entry[f"{metric}_mean"] = statistics.fmean(values)
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(entry)
    baseline = {(row["scope"], row["participant"], row["seed"]): row for row in rows if row["experiment"] == "A0"}
    paired = defaultdict(list)
    for row in rows:
        if row["experiment"] == "A0":
            continue
        base = baseline.get((row["scope"], row["participant"], row["seed"]))
        if base:
            paired[(row["scope"], row["experiment"])].append(float(row["node_accuracy"]) - float(base["node_accuracy"]))
    paired_deltas = [
        {
            "scope": scope,
            "experiment": experiment,
            "paired_runs": len(values),
            "node_accuracy_delta_vs_A0_mean": statistics.fmean(values),
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        for (scope, experiment), values in sorted(paired.items())
    ]
    report = {"split": args.split, "completed_runs": len(rows), "aggregates": aggregates, "paired_deltas": paired_deltas}
    write_json(summary_root / f"{args.split}_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

