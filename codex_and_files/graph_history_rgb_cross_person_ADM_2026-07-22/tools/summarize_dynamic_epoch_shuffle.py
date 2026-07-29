from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DYNAMIC_MODELS = (
    "m3_dynamic_frozen_m0_delta",
    "m3_dynamic_joint_head_delta",
    "m3_dynamic_direct_fusion",
)
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
        "retrained_normal_only"
        if train_scope == "normal_only"
        else "retrained_all_runs"
    )
    return seed_root / "history_models" / representation / train_scope


def reference_paths(
    seed_root: Path,
    train_scope: str,
    dynamic_root: Path,
    model: str,
    split: str,
) -> list[tuple[str, Path]]:
    legacy_root = legacy_model_root(seed_root, train_scope)
    static_direct_root = (
        seed_root / "history_models" / "direct_head_fusion" / train_scope
    )
    suffix = Path("test_results") / f"{split}_metrics.json"
    references: dict[str, list[tuple[str, Path]]] = {
        "m3_dynamic_frozen_m0_delta": [
            ("m0", legacy_root / "m0" / suffix),
            ("m3", legacy_root / "m3" / suffix),
        ],
        "m3_dynamic_joint_head_delta": [
            ("m0", legacy_root / "m0" / suffix),
            ("m3", legacy_root / "m3" / suffix),
            (
                "m3_dynamic_frozen_m0_delta",
                dynamic_root / "m3_dynamic_frozen_m0_delta" / suffix,
            ),
        ],
        "m3_dynamic_direct_fusion": [
            ("m0", legacy_root / "m0" / suffix),
            ("m3_direct", static_direct_root / "m3_direct" / suffix),
            (
                "m3_dynamic_frozen_m0_delta",
                dynamic_root / "m3_dynamic_frozen_m0_delta" / suffix,
            ),
            (
                "m3_dynamic_joint_head_delta",
                dynamic_root / "m3_dynamic_joint_head_delta" / suffix,
            ),
        ],
    }
    return references[model]


def collect_rows(
    outputs_root: Path,
    camera_id: str,
    participants: list[str],
    seeds: list[int],
    train_scopes: list[str],
    require_complete_grid: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    missing: set[str] = set()
    for participant in participants:
        for seed in seeds:
            seed_root = (
                outputs_root
                / f"{participant}_as_test"
                / f"cam_{camera_id}"
                / f"seed_{seed}"
            )
            for train_scope in train_scopes:
                dynamic_root = (
                    seed_root
                    / "history_models"
                    / "dynamic_epoch_shuffle"
                    / train_scope
                )
                for model in DYNAMIC_MODELS:
                    for split in SPLITS:
                        metric_path = (
                            dynamic_root
                            / model
                            / "test_results"
                            / f"{split}_metrics.json"
                        )
                        if not metric_path.is_file():
                            missing.add(str(metric_path))
                            continue
                        metric = read_metric(metric_path)
                        metric_rows.append(
                            {
                                "participant": participant,
                                "seed": seed,
                                "train_scope": train_scope,
                                "model": model,
                                "split": split,
                                **metric,
                                "metrics_path": str(metric_path),
                            }
                        )
                        for reference_name, reference_path in reference_paths(
                            seed_root,
                            train_scope,
                            dynamic_root,
                            model,
                            split,
                        ):
                            if not reference_path.is_file():
                                missing.add(str(reference_path))
                                continue
                            reference_metric = read_metric(reference_path)
                            row: dict[str, Any] = {
                                "participant": participant,
                                "seed": seed,
                                "train_scope": train_scope,
                                "model": model,
                                "reference_model": reference_name,
                                "split": split,
                                "dynamic_metrics_path": str(metric_path),
                                "reference_metrics_path": str(reference_path),
                            }
                            for field in METRIC_FIELDS:
                                row[f"delta_{field}"] = (
                                    float(metric[field])
                                    - float(reference_metric[field])
                                )
                            delta_rows.append(row)

    if missing and require_complete_grid:
        preview = "\n".join(sorted(missing)[:20])
        raise FileNotFoundError(
            f"Dynamic epoch-shuffle grid is incomplete; missing {len(missing)} files. "
            f"First missing paths:\n{preview}"
        )
    return metric_rows, delta_rows


def build_aggregate(
    metric_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_metrics: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in metric_rows:
        grouped_metrics[
            (row["participant"], row["train_scope"], row["model"], row["split"])
        ].append(row)
    participant_metrics = {
        key: {
            field: mean([float(row[field]) for row in rows])
            for field in METRIC_FIELDS
        }
        for key, rows in grouped_metrics.items()
    }

    grouped_deltas: dict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
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
        participants = sorted(
            participant
            for participant, scope, current_model, current_split in participant_metrics
            if (scope, current_model, current_split)
            == (train_scope, model, split)
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
            "participant_count": len(participants),
            "participants": ",".join(participants),
        }
        for field in METRIC_FIELDS:
            metric_values = [
                participant_metrics[
                    (participant, train_scope, model, split)
                ][field]
                for participant in participants
            ]
            delta_values = [
                participant_deltas[
                    (
                        participant,
                        train_scope,
                        model,
                        reference_model,
                        split,
                    )
                ][field]
                for participant in participants
            ]
            row[f"mean_{field}"] = mean(metric_values)
            row[f"std_{field}"] = sample_std(metric_values)
            row[f"mean_delta_{field}"] = mean(delta_values)
            row[f"std_delta_{field}"] = sample_std(delta_values)
        aggregate.append(row)
    return aggregate


def ensure_new_summary_dir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Summary directory is not empty: {path}. "
            "Use --overwrite only for this dedicated dynamic summary."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the three dynamic graph-valid epoch-shuffle models and "
            "strictly pair them with static and dynamic references."
        )
    )
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-id", default="001484412812")
    parser.add_argument(
        "--participants",
        nargs="+",
        default=["A", "D", "J", "M"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
    parser.add_argument(
        "--train-scopes",
        nargs="+",
        choices=["normal_only", "all_runs"],
        default=["normal_only", "all_runs"],
    )
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metric_rows, delta_rows = collect_rows(
        outputs_root=Path(args.outputs_root),
        camera_id=args.camera_id,
        participants=args.participants,
        seeds=args.seeds,
        train_scopes=args.train_scopes,
        require_complete_grid=args.require_complete_grid,
    )
    if not metric_rows:
        raise FileNotFoundError(
            f"No complete dynamic epoch-shuffle results found under {args.outputs_root}"
        )
    aggregate_rows = build_aggregate(metric_rows, delta_rows)
    output_dir = ensure_new_summary_dir(
        Path(args.output_dir),
        overwrite=args.overwrite,
    )
    write_csv(output_dir / "dynamic_epoch_shuffle_metrics.csv", metric_rows)
    write_csv(
        output_dir / "dynamic_epoch_shuffle_paired_deltas.csv",
        delta_rows,
    )
    write_csv(
        output_dir / "dynamic_epoch_shuffle_aggregate.csv",
        aggregate_rows,
    )
    with (output_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_family": "dynamic_epoch_graph_valid_shuffle",
                "participants": args.participants,
                "seeds": args.seeds,
                "train_scopes": args.train_scopes,
                "models": list(DYNAMIC_MODELS),
                "metric_rows": len(metric_rows),
                "paired_delta_rows": len(delta_rows),
                "aggregate_rows": len(aggregate_rows),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(
        f"metric_rows={len(metric_rows)} paired_rows={len(delta_rows)} "
        f"aggregate_rows={len(aggregate_rows)}"
    )
    print(f"Saved dynamic epoch-shuffle summaries to {output_dir}")


if __name__ == "__main__":
    main()
