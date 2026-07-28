from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DIRECT_MODELS = ("m1_direct", "m2_direct", "m3_direct")
SPLITS = ("test_normal", "test_fault", "test_all")
METRIC_FIELDS = (
    "node_accuracy",
    "node_macro_f1",
    "node_balanced_accuracy",
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


def read_metric(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "samples": int(payload.get("samples", 0)),
        "node_accuracy": float(payload["node"]["accuracy"]),
        "node_macro_f1": float(payload["node"]["macro_f1"]),
        "node_balanced_accuracy": float(payload["node"]["balanced_accuracy"]),
        "tier3_accuracy": float(payload["tier3"]["accuracy"]),
        "tier3_macro_f1": float(payload["tier3"]["macro_f1"]),
        "tier3_balanced_accuracy": float(payload["tier3"]["balanced_accuracy"]),
    }


def legacy_model_root(seed_root: Path, train_scope: str) -> Path:
    representation = (
        "retrained_normal_only" if train_scope == "normal_only" else "retrained_all_runs"
    )
    return seed_root / "history_models" / representation / train_scope


def collect_rows(
    outputs_root: Path,
    camera_id: str,
    participants: list[str],
    seeds: list[int],
    train_scopes: list[str],
    require_complete_grid: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for participant in participants:
        for seed in seeds:
            seed_root = (
                outputs_root
                / f"{participant}_as_test"
                / f"cam_{camera_id}"
                / f"seed_{seed}"
            )
            for train_scope in train_scopes:
                direct_root = (
                    seed_root / "history_models" / "direct_head_fusion" / train_scope
                )
                reference_root = legacy_model_root(seed_root, train_scope)
                for model in DIRECT_MODELS:
                    legacy_model = model.removesuffix("_direct")
                    for split in SPLITS:
                        direct_path = direct_root / model / "test_results" / f"{split}_metrics.json"
                        m0_path = reference_root / "m0" / "test_results" / f"{split}_metrics.json"
                        delta_path = (
                            reference_root
                            / legacy_model
                            / "test_results"
                            / f"{split}_metrics.json"
                        )
                        required_paths = (direct_path, m0_path, delta_path)
                        absent = [str(path) for path in required_paths if not path.is_file()]
                        if absent:
                            missing.extend(absent)
                            continue
                        direct_metric = read_metric(direct_path)
                        m0_metric = read_metric(m0_path)
                        legacy_metric = read_metric(delta_path)
                        row: dict[str, Any] = {
                            "participant": participant,
                            "seed": seed,
                            "train_scope": train_scope,
                            "model": model,
                            "split": split,
                            **direct_metric,
                            "metrics_path": str(direct_path),
                        }
                        direct_rows.append(row)
                        for reference_name, reference_metric, reference_path in (
                            ("m0", m0_metric, m0_path),
                            (legacy_model, legacy_metric, delta_path),
                        ):
                            comparison: dict[str, Any] = {
                                "participant": participant,
                                "seed": seed,
                                "train_scope": train_scope,
                                "model": model,
                                "reference_model": reference_name,
                                "split": split,
                                "direct_metrics_path": str(direct_path),
                                "reference_metrics_path": str(reference_path),
                            }
                            for field in METRIC_FIELDS:
                                comparison[f"delta_{field}"] = (
                                    float(direct_metric[field]) - float(reference_metric[field])
                                )
                            delta_rows.append(comparison)
    if missing and require_complete_grid:
        preview = "\n".join(sorted(set(missing))[:20])
        raise FileNotFoundError(
            f"Direct-head strict grid is incomplete; missing {len(set(missing))} files. "
            f"First missing paths:\n{preview}"
        )
    return direct_rows, delta_rows


def build_aggregate(
    direct_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_metrics: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in direct_rows:
        grouped_metrics[
            (row["participant"], row["train_scope"], row["model"], row["split"])
        ].append(row)
    participant_metrics = {
        key: {field: mean([float(row[field]) for row in rows]) for field in METRIC_FIELDS}
        for key, rows in grouped_metrics.items()
    }

    grouped_deltas: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in delta_rows:
        grouped_deltas[
            (
                row["participant"],
                row["train_scope"],
                row["model"],
                row["reference_model"],
                row["split"],
            )
        ].append(row)
    participant_deltas = {
        key: {
            field: mean([float(row[f"delta_{field}"]) for row in rows])
            for field in METRIC_FIELDS
        }
        for key, rows in grouped_deltas.items()
    }

    experiment_keys = sorted(
        {
            (train_scope, model, reference_model, split)
            for _, train_scope, model, reference_model, split in participant_deltas
        }
    )
    aggregate: list[dict[str, Any]] = []
    for train_scope, model, reference_model, split in experiment_keys:
        people = sorted(
            participant
            for participant, scope, current_model, current_split in participant_metrics
            if (scope, current_model, current_split) == (train_scope, model, split)
            and (
                participant,
                train_scope,
                model,
                reference_model,
                split,
            )
            in participant_deltas
        )
        row: dict[str, Any] = {
            "train_scope": train_scope,
            "model": model,
            "reference_model": reference_model,
            "split": split,
            "participant_count": len(people),
            "participants": ",".join(people),
        }
        for field in METRIC_FIELDS:
            metric_values = [
                participant_metrics[(person, train_scope, model, split)][field]
                for person in people
            ]
            delta_values = [
                participant_deltas[
                    (person, train_scope, model, reference_model, split)
                ][field]
                for person in people
            ]
            row[f"mean_{field}"] = mean(metric_values)
            row[f"std_{field}"] = sample_std(metric_values)
            row[f"mean_delta_{field}"] = mean(delta_values)
            row[f"std_delta_{field}"] = sample_std(delta_values)
        aggregate.append(row)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize direct-head M1-M3 and pair each result with the matching legacy "
            "M0 and legacy delta model"
        )
    )
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-id", default="001484412812")
    parser.add_argument("--participants", nargs="+", default=["A", "D", "J", "M"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
    parser.add_argument(
        "--train-scopes",
        nargs="+",
        choices=["normal_only", "all_runs"],
        default=["normal_only", "all_runs"],
    )
    parser.add_argument("--require-complete-grid", action="store_true")
    args = parser.parse_args()

    direct_rows, delta_rows = collect_rows(
        outputs_root=Path(args.outputs_root),
        camera_id=args.camera_id,
        participants=args.participants,
        seeds=args.seeds,
        train_scopes=args.train_scopes,
        require_complete_grid=args.require_complete_grid,
    )
    if not direct_rows:
        raise FileNotFoundError(f"No complete direct-head comparisons found under {args.outputs_root}")
    aggregate = build_aggregate(direct_rows, delta_rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "direct_head_metrics.csv", direct_rows)
    write_csv(output_dir / "direct_head_paired_deltas.csv", delta_rows)
    write_csv(output_dir / "direct_head_aggregate.csv", aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_family": "direct_head_fusion",
                "participants": args.participants,
                "seeds": args.seeds,
                "train_scopes": args.train_scopes,
                "direct_metric_rows": len(direct_rows),
                "paired_delta_rows": len(delta_rows),
                "aggregate_rows": len(aggregate),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(
        f"direct_rows={len(direct_rows)} paired_rows={len(delta_rows)} "
        f"aggregate_rows={len(aggregate)}"
    )
    print(f"Saved direct-head summaries to {output_dir}")


if __name__ == "__main__":
    main()
