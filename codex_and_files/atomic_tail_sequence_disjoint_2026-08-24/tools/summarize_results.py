from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize completed sequence-disjoint experiments")
    parser.add_argument("--output-root", default=str(PACKAGE_ROOT / "outputs" / "history_models"))
    parser.add_argument("--summary-root", default=str(PACKAGE_ROOT / "outputs" / "summary"))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    records: list[dict[str, Any]] = []
    for completed_path in output_root.glob("*/all_runs/*_as_test/seed_*/completed.json"):
        completed = read_json(completed_path)
        for split, metrics in completed["metrics"].items():
            records.append({
                "experiment": completed["experiment_id"],
                "participant": completed["participant"],
                "seed": completed["seed"],
                "split": split.replace("test_", ""),
                "samples": metrics["samples"],
                "node_accuracy": metrics["node"]["accuracy"],
                "node_macro_f1": metrics["node"]["macro_f1"],
                "node_balanced_accuracy": metrics["node"]["balanced_accuracy"],
                "tier3_accuracy": metrics["tier3"]["accuracy"],
                "tier3_macro_f1": metrics["tier3"]["macro_f1"],
                "tier3_balanced_accuracy": metrics["tier3"]["balanced_accuracy"],
                "completed_json": str(completed_path),
            })
    summary_root = Path(args.summary_root)
    write_csv(summary_root / "fold_seed_metrics.csv", records)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["experiment"]), str(record["split"]))].append(record)
    metric_names = [
        "node_accuracy", "node_macro_f1", "node_balanced_accuracy",
        "tier3_accuracy", "tier3_macro_f1", "tier3_balanced_accuracy",
    ]
    aggregates: list[dict[str, Any]] = []
    for (experiment, split), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "experiment": experiment,
            "split": split,
            "fold_seed_records": len(rows),
            "participants": ",".join(sorted({str(row["participant"]) for row in rows})),
            "seeds": ",".join(map(str, sorted({int(row["seed"]) for row in rows}))),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    write_csv(summary_root / "aggregate_metrics.csv", aggregates)

    baseline = {
        (str(row["participant"]), int(row["seed"]), str(row["split"])): row
        for row in records if row["experiment"] == "M2-Direct-RealOrder"
    }
    deltas: list[dict[str, Any]] = []
    for row in records:
        if row["experiment"] == "M2-Direct-RealOrder":
            continue
        key = (str(row["participant"]), int(row["seed"]), str(row["split"]))
        if key not in baseline:
            continue
        control = baseline[key]
        deltas.append({
            "experiment": row["experiment"],
            "participant": row["participant"],
            "seed": row["seed"],
            "split": row["split"],
            "delta_node_accuracy_vs_M2": float(row["node_accuracy"]) - float(control["node_accuracy"]),
            "delta_tier3_accuracy_vs_M2": float(row["tier3_accuracy"]) - float(control["tier3_accuracy"]),
        })
    write_csv(summary_root / "paired_deltas_vs_M2.csv", deltas)
    with (summary_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": records, "aggregates": aggregates, "paired_deltas": deltas}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Found {len(records)} split-level records; wrote summaries to {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
