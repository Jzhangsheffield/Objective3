from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


METRIC_FIELDS = (
    "node_accuracy",
    "node_macro_f1",
    "tier3_accuracy",
    "tier3_macro_f1",
    "tier3_balanced_accuracy",
)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def collect_rows(outputs_root: Path, participants: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "node" not in metrics or "tier3" not in metrics:
            continue
        parts = path.relative_to(outputs_root).parts
        if len(parts) < 9 or not parts[0].endswith("_as_test"):
            continue
        participant = parts[0].removesuffix("_as_test")
        if participant not in participants:
            continue
        seed_part = next((part for part in parts if part.startswith("seed_")), "seed_unknown")
        try:
            history_index = parts.index("history_models")
            train_scope = parts[history_index + 2]
            model = parts[history_index + 3]
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "participant": participant,
                "seed": seed_part.removeprefix("seed_"),
                "train_scope": train_scope,
                "model": model,
                "split": metrics.get("split", path.stem.removesuffix("_metrics")),
                "samples": int(metrics.get("samples", 0)),
                "node_accuracy": float(metrics["node"]["accuracy"]),
                "node_macro_f1": float(metrics["node"]["macro_f1"]),
                "tier3_accuracy": float(metrics["tier3"]["accuracy"]),
                "tier3_macro_f1": float(metrics["tier3"]["macro_f1"]),
                "tier3_balanced_accuracy": float(metrics["tier3"]["balanced_accuracy"]),
                "metrics_path": str(path),
            }
        )
    return rows


def build_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (row["participant"], row["seed"], row["train_scope"], row["split"]): row
        for row in rows
        if row["model"] == "m0"
    }
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["model"] == "m0":
            continue
        key = (row["participant"], row["seed"], row["train_scope"], row["split"])
        if key not in baseline:
            continue
        base = baseline[key]
        delta = {
            "participant": row["participant"],
            "seed": row["seed"],
            "train_scope": row["train_scope"],
            "model": row["model"],
            "split": row["split"],
        }
        for field in METRIC_FIELDS:
            delta[f"delta_{field}"] = float(row[field]) - float(base[field])
        deltas.append(delta)
    return deltas


def build_aggregate(
    rows: list[dict[str, Any]], deltas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # Average repeated seeds inside each held-out participant first, then summarize people.
    person_metrics: dict[tuple[str, str, str, str], dict[str, float]] = {}
    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["participant"], row["train_scope"], row["model"], row["split"])].append(row)
    for key, group in grouped_rows.items():
        person_metrics[key] = {field: mean([float(row[field]) for row in group]) for field in METRIC_FIELDS}

    person_deltas: dict[tuple[str, str, str, str], dict[str, float]] = {}
    grouped_deltas: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped_deltas[(row["participant"], row["train_scope"], row["model"], row["split"])].append(row)
    for key, group in grouped_deltas.items():
        person_deltas[key] = {
            field: mean([float(row[f"delta_{field}"]) for row in group]) for field in METRIC_FIELDS
        }

    experiment_keys = sorted({(key[1], key[2], key[3]) for key in person_metrics})
    aggregate: list[dict[str, Any]] = []
    for train_scope, model, split in experiment_keys:
        people = sorted(
            participant
            for participant, scope, current_model, current_split in person_metrics
            if (scope, current_model, current_split) == (train_scope, model, split)
        )
        row: dict[str, Any] = {
            "train_scope": train_scope,
            "model": model,
            "split": split,
            "participant_count": len(people),
            "participants": ",".join(people),
        }
        for field in METRIC_FIELDS:
            values = [person_metrics[(person, train_scope, model, split)][field] for person in people]
            row[f"mean_{field}"] = mean(values)
            row[f"std_{field}"] = sample_std(values)
            delta_values = [
                person_deltas[(person, train_scope, model, split)][field]
                for person in people
                if (person, train_scope, model, split) in person_deltas
            ]
            row[f"mean_delta_vs_m0_{field}"] = mean(delta_values) if delta_values else ""
            row[f"std_delta_vs_m0_{field}"] = sample_std(delta_values) if delta_values else ""
        aggregate.append(row)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A/D/M cross-person M0-M6 results")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", nargs="+", default=["A", "D", "M"])
    args = parser.parse_args()
    rows = collect_rows(Path(args.outputs_root), set(args.participants))
    if not rows:
        raise FileNotFoundError(f"No model metric files found under {args.outputs_root}")
    deltas = build_deltas(rows)
    aggregate = build_aggregate(rows, deltas)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "cross_person_metrics.csv", rows)
    write_csv(output_dir / "cross_person_deltas_vs_m0.csv", deltas)
    write_csv(output_dir / "cross_person_aggregate.csv", aggregate)
    print(
        f"participants={sorted(set(row['participant'] for row in rows))} "
        f"metric_rows={len(rows)} delta_rows={len(deltas)} aggregate_rows={len(aggregate)}"
    )
    print(f"Saved cross-person summaries to {output_dir}")


if __name__ == "__main__":
    main()
