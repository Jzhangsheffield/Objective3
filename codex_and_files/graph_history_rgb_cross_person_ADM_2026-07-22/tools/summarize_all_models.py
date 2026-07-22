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
    "node_balanced_accuracy",
    "tier3_accuracy",
    "tier3_macro_f1",
    "tier3_balanced_accuracy",
)
REFERENCE_MODELS = (
    "m0",
    "e2e_node_scratch",
    "e2e_node_from_tier3",
    "e2e_tier3_scratch",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def metric_value(metrics: dict[str, Any], space: str, name: str) -> float | None:
    block = metrics.get(space)
    if not isinstance(block, dict) or name not in block:
        return None
    return float(block[name])


def parse_location(path: Path, outputs_root: Path) -> dict[str, str] | None:
    parts = path.relative_to(outputs_root).parts
    if len(parts) < 8 or not parts[0].endswith("_as_test"):
        return None
    participant = parts[0].removesuffix("_as_test")
    seed_part = next((part for part in parts if part.startswith("seed_")), None)
    if seed_part is None:
        return None
    if "history_models" in parts:
        index = parts.index("history_models")
        try:
            feature_source = parts[index + 1]
            representation_scope = feature_source.removeprefix("retrained_")
            return {
                "participant": participant,
                "seed": seed_part.removeprefix("seed_"),
                "representation_scope": representation_scope,
                "train_scope": parts[index + 2],
                "model": parts[index + 3],
                "family": "history_feature",
            }
        except IndexError:
            return None
    if "e2e_baselines" in parts:
        index = parts.index("e2e_baselines")
        try:
            return {
                "participant": participant,
                "seed": seed_part.removeprefix("seed_"),
                "representation_scope": parts[index + 1],
                "train_scope": parts[index + 1],
                "model": parts[index + 2],
                "family": "e2e_video",
            }
        except IndexError:
            return None
    return None


def collect_rows(
    outputs_root: Path,
    participants: set[str],
    seeds: set[str] | None,
    train_scopes: set[str] | None = None,
    representation_scopes: set[str] | None = None,
    matched_scope_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("*_metrics.json")):
        location = parse_location(path, outputs_root)
        if location is None or location["participant"] not in participants:
            continue
        if seeds is not None and location["seed"] not in seeds:
            continue
        if train_scopes is not None and location["train_scope"] not in train_scopes:
            continue
        if (
            representation_scopes is not None
            and location["representation_scope"] not in representation_scopes
        ):
            continue
        if matched_scope_only and location["representation_scope"] != location["train_scope"]:
            continue
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics.get("tier3"), dict):
            continue
        node = metrics.get("node") if isinstance(metrics.get("node"), dict) else None
        tier3 = metrics["tier3"]
        rows.append(
            {
                **location,
                "target_space": metrics.get(
                    "target_space", "node" if node is not None else "tier3"
                ),
                "split": metrics.get("split", path.stem.removesuffix("_metrics")),
                "samples": int(metrics.get("samples", 0)),
                "node_present_classes": node.get("present_class_count") if node else None,
                "tier3_present_classes": tier3.get("present_class_count"),
                "node_accuracy": metric_value(metrics, "node", "accuracy"),
                "node_macro_f1": metric_value(metrics, "node", "macro_f1"),
                "node_balanced_accuracy": metric_value(metrics, "node", "balanced_accuracy"),
                "tier3_accuracy": metric_value(metrics, "tier3", "accuracy"),
                "tier3_macro_f1": metric_value(metrics, "tier3", "macro_f1"),
                "tier3_balanced_accuracy": metric_value(metrics, "tier3", "balanced_accuracy"),
                "metrics_path": str(path),
            }
        )
    return rows


def collect_stage_rows(
    outputs_root: Path,
    participants: set[str],
    seeds: set[str] | None,
    train_scopes: set[str] | None = None,
    representation_scopes: set[str] | None = None,
    matched_scope_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("*_metrics.json")):
        location = parse_location(path, outputs_root)
        if location is None or location["participant"] not in participants:
            continue
        if seeds is not None and location["seed"] not in seeds:
            continue
        if train_scopes is not None and location["train_scope"] not in train_scopes:
            continue
        if (
            representation_scopes is not None
            and location["representation_scope"] not in representation_scopes
        ):
            continue
        if matched_scope_only and location["representation_scope"] != location["train_scope"]:
            continue
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        for stage_id, stage_block in metrics.get("per_stage", {}).items():
            if not isinstance(stage_block, dict):
                continue
            node = stage_block.get("node") if isinstance(stage_block.get("node"), dict) else None
            tier3 = stage_block.get("tier3") if isinstance(stage_block.get("tier3"), dict) else None
            if tier3 is None and metrics.get("target_space") == "tier3":
                tier3 = stage_block.get("metrics") if isinstance(stage_block.get("metrics"), dict) else None
            if tier3 is None:
                continue
            rows.append(
                {
                    **location,
                    "target_space": metrics.get(
                        "target_space", "node" if node is not None else "tier3"
                    ),
                    "split": metrics.get("split", path.stem.removesuffix("_metrics")),
                    "stage_id": str(stage_id),
                    "samples": int(stage_block.get("samples", 0)),
                    "node_present_classes": node.get("present_class_count") if node else None,
                    "tier3_present_classes": tier3.get("present_class_count"),
                    "node_accuracy": float(node["accuracy"]) if node else None,
                    "node_macro_f1": float(node["macro_f1"]) if node else None,
                    "node_balanced_accuracy": float(node["balanced_accuracy"]) if node else None,
                    "tier3_accuracy": float(tier3["accuracy"]),
                    "tier3_macro_f1": float(tier3["macro_f1"]),
                    "tier3_balanced_accuracy": float(tier3["balanced_accuracy"]),
                    "metrics_path": str(path),
                }
            )
    return rows


def build_pairwise_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["participant"],
            row["seed"],
            row["representation_scope"],
            row["train_scope"],
            row["split"],
            row["model"],
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        for reference_model in REFERENCE_MODELS:
            if row["model"] == reference_model:
                continue
            reference = lookup.get(
                (
                    row["participant"],
                    row["seed"],
                    row["representation_scope"],
                    row["train_scope"],
                    row["split"],
                    reference_model,
                )
            )
            if reference is None:
                continue
            delta_row: dict[str, Any] = {
                "participant": row["participant"],
                "seed": row["seed"],
                "representation_scope": row["representation_scope"],
                "train_scope": row["train_scope"],
                "model": row["model"],
                "reference_model": reference_model,
                "split": row["split"],
            }
            comparable = False
            for field in METRIC_FIELDS:
                current_value = row[field]
                reference_value = reference[field]
                if current_value is None or reference_value is None:
                    delta_row[f"delta_{field}"] = None
                else:
                    delta_row[f"delta_{field}"] = float(current_value) - float(reference_value)
                    comparable = True
            if comparable:
                output.append(delta_row)
    return output


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def std_or_none(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)


def aggregate_across_people(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_person: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_person[
            (
                row["participant"],
                row["representation_scope"],
                row["train_scope"],
                row["model"],
                row["split"],
            )
        ].append(row)
    person_means: dict[tuple[str, str, str, str, str], dict[str, float | None]] = {}
    for key, group in by_person.items():
        person_means[key] = {
            field: mean_or_none([float(row[field]) for row in group if row[field] is not None])
            for field in METRIC_FIELDS
        }
    experiment_keys = sorted({(key[1], key[2], key[3], key[4]) for key in person_means})
    output: list[dict[str, Any]] = []
    for representation_scope, train_scope, model, split in experiment_keys:
        people = sorted(
            participant
            for participant, representation, scope, current_model, current_split in person_means
            if (representation, scope, current_model, current_split)
            == (representation_scope, train_scope, model, split)
        )
        row: dict[str, Any] = {
            "representation_scope": representation_scope,
            "train_scope": train_scope,
            "model": model,
            "split": split,
            "participant_count": len(people),
            "participants": ",".join(people),
        }
        for field in METRIC_FIELDS:
            values = [
                person_means[(person, representation_scope, train_scope, model, split)][field]
                for person in people
                if person_means[(person, representation_scope, train_scope, model, split)][field]
                is not None
            ]
            row[f"mean_{field}"] = mean_or_none([float(value) for value in values])
            row[f"std_{field}"] = std_or_none([float(value) for value in values])
        output.append(row)
    return output


def aggregate_stages_across_people(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed = [
        {**row, "split": f"{row['split']}|||{row['stage_id']}"}
        for row in rows
    ]
    aggregate = aggregate_across_people(transformed)
    for row in aggregate:
        split, stage_id = str(row["split"]).rsplit("|||", 1)
        row["split"] = split
        row["stage_id"] = stage_id
    return aggregate


def aggregate_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_person: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_person[
            (
                row["participant"],
                row["representation_scope"],
                row["train_scope"],
                row["model"],
                row["reference_model"],
                row["split"],
            )
        ].append(row)
    person_means: dict[tuple[str, str, str, str, str, str], dict[str, float | None]] = {}
    for key, group in by_person.items():
        person_means[key] = {
            field: mean_or_none(
                [float(row[f"delta_{field}"]) for row in group if row[f"delta_{field}"] is not None]
            )
            for field in METRIC_FIELDS
        }
    experiment_keys = sorted({(key[1], key[2], key[3], key[4], key[5]) for key in person_means})
    output: list[dict[str, Any]] = []
    for representation_scope, train_scope, model, reference_model, split in experiment_keys:
        people = sorted(
            participant
            for participant, representation, scope, current_model, reference, current_split in person_means
            if (representation, scope, current_model, reference, current_split)
            == (representation_scope, train_scope, model, reference_model, split)
        )
        row: dict[str, Any] = {
            "representation_scope": representation_scope,
            "train_scope": train_scope,
            "model": model,
            "reference_model": reference_model,
            "split": split,
            "participant_count": len(people),
            "participants": ",".join(people),
        }
        for field in METRIC_FIELDS:
            values = [
                person_means[
                    (person, representation_scope, train_scope, model, reference_model, split)
                ][field]
                for person in people
                if person_means[
                    (person, representation_scope, train_scope, model, reference_model, split)
                ][field]
                is not None
            ]
            row[f"mean_delta_{field}"] = mean_or_none([float(value) for value in values])
            row[f"std_delta_{field}"] = std_or_none([float(value) for value in values])
        output.append(row)
    return output


def build_training_scope_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare complete all-runs and normal-only pipelines for the same fold/model/split."""
    matched = [row for row in rows if row["representation_scope"] == row["train_scope"]]
    lookup = {
        (row["participant"], row["seed"], row["train_scope"], row["model"], row["split"]): row
        for row in matched
    }
    output: list[dict[str, Any]] = []
    for row in matched:
        if row["train_scope"] != "all_runs":
            continue
        normal = lookup.get(
            (row["participant"], row["seed"], "normal_only", row["model"], row["split"])
        )
        if normal is None:
            continue
        delta: dict[str, Any] = {
            "participant": row["participant"],
            "seed": row["seed"],
            "model": row["model"],
            "split": row["split"],
            "comparison": "all_runs_minus_normal_only",
        }
        comparable = False
        for field in METRIC_FIELDS:
            if row[field] is None or normal[field] is None:
                delta[f"delta_{field}"] = None
            else:
                delta[f"delta_{field}"] = float(row[field]) - float(normal[field])
                comparable = True
        if comparable:
            output.append(delta)
    return output


def aggregate_training_scope_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["participant"], row["model"], row["split"])].append(row)
    person_means = {
        key: {
            field: mean_or_none(
                [float(row[f"delta_{field}"]) for row in group if row[f"delta_{field}"] is not None]
            )
            for field in METRIC_FIELDS
        }
        for key, group in grouped.items()
    }
    output: list[dict[str, Any]] = []
    for model, split in sorted({(key[1], key[2]) for key in person_means}):
        people = sorted(
            participant
            for participant, current_model, current_split in person_means
            if (current_model, current_split) == (model, split)
        )
        row: dict[str, Any] = {
            "model": model,
            "split": split,
            "comparison": "all_runs_minus_normal_only",
            "participant_count": len(people),
            "participants": ",".join(people),
        }
        for field in METRIC_FIELDS:
            values = [
                person_means[(person, model, split)][field]
                for person in people
                if person_means[(person, model, split)][field] is not None
            ]
            row[f"mean_delta_{field}"] = mean_or_none([float(value) for value in values])
            row[f"std_delta_{field}"] = std_or_none([float(value) for value in values])
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified statistics for E2E Tier-3, E2E Node, and history M0-M6"
    )
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", nargs="+", default=["A", "D", "M"])
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--train-scopes", nargs="+", default=None)
    parser.add_argument("--representation-scopes", nargs="+", default=None)
    parser.add_argument("--matched-scope-only", action="store_true")
    args = parser.parse_args()
    rows = collect_rows(
        Path(args.outputs_root),
        set(args.participants),
        set(args.seeds) if args.seeds else None,
        set(args.train_scopes) if args.train_scopes else None,
        set(args.representation_scopes) if args.representation_scopes else None,
        args.matched_scope_only,
    )
    if not rows:
        raise FileNotFoundError(f"No unified metric files found under {args.outputs_root}")
    stage_rows = collect_stage_rows(
        Path(args.outputs_root),
        set(args.participants),
        set(args.seeds) if args.seeds else None,
        set(args.train_scopes) if args.train_scopes else None,
        set(args.representation_scopes) if args.representation_scopes else None,
        args.matched_scope_only,
    )
    deltas = build_pairwise_deltas(rows)
    aggregate = aggregate_across_people(rows)
    delta_aggregate = aggregate_deltas(deltas)
    stage_aggregate = aggregate_stages_across_people(stage_rows)
    training_scope_deltas = build_training_scope_deltas(rows)
    training_scope_delta_aggregate = aggregate_training_scope_deltas(training_scope_deltas)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "all_model_metrics.csv", rows)
    write_csv(output_dir / "all_model_pairwise_deltas.csv", deltas)
    write_csv(output_dir / "all_model_cross_person_aggregate.csv", aggregate)
    write_csv(output_dir / "all_model_delta_aggregate.csv", delta_aggregate)
    write_csv(output_dir / "all_model_per_stage_metrics.csv", stage_rows)
    write_csv(
        output_dir / "all_model_per_stage_cross_person_aggregate.csv", stage_aggregate
    )
    write_csv(output_dir / "all_model_training_scope_deltas.csv", training_scope_deltas)
    write_csv(
        output_dir / "all_model_training_scope_delta_aggregate.csv",
        training_scope_delta_aggregate,
    )
    print(
        f"models={sorted(set(row['model'] for row in rows))} rows={len(rows)} "
        f"deltas={len(deltas)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
