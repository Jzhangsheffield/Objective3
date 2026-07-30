from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ATOMIC_TAIL_MODELS = (
    "m3_atomic_tail_frozen_m0_delta",
    "m3_atomic_tail_joint_head_delta",
    "m3_atomic_tail_direct_fusion",
)
REFRESH_POLICIES = (
    "refresh_every_1",
    "refresh_every_10",
    "refresh_once",
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
        "tier3_balanced_accuracy": float(
            payload["tier3"]["balanced_accuracy"]
        ),
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
    atomic_root: Path,
    refresh_policy: str,
    model: str,
    split: str,
) -> list[tuple[str, Path, bool]]:
    legacy_root = legacy_model_root(seed_root, train_scope)
    direct_root = (
        seed_root / "history_models" / "direct_head_fusion" / train_scope
    )
    dynamic_root = (
        seed_root / "history_models" / "dynamic_epoch_shuffle" / train_scope
    )
    suffix = Path("test_results") / f"{split}_metrics.json"
    references: dict[str, list[tuple[str, Path, bool]]] = {
        "m3_atomic_tail_frozen_m0_delta": [
            ("m0", legacy_root / "m0" / suffix, True),
            ("m3", legacy_root / "m3" / suffix, True),
            (
                "m3_dynamic_frozen_m0_delta",
                dynamic_root / "m3_dynamic_frozen_m0_delta" / suffix,
                False,
            ),
        ],
        "m3_atomic_tail_joint_head_delta": [
            ("m0", legacy_root / "m0" / suffix, True),
            (
                "atomic_frozen_same_policy",
                atomic_root
                / refresh_policy
                / "m3_atomic_tail_frozen_m0_delta"
                / suffix,
                True,
            ),
            (
                "m3_dynamic_joint_head_delta",
                dynamic_root / "m3_dynamic_joint_head_delta" / suffix,
                False,
            ),
        ],
        "m3_atomic_tail_direct_fusion": [
            ("m0", legacy_root / "m0" / suffix, True),
            ("m3_direct", direct_root / "m3_direct" / suffix, True),
            (
                "atomic_joint_same_policy",
                atomic_root
                / refresh_policy
                / "m3_atomic_tail_joint_head_delta"
                / suffix,
                True,
            ),
            (
                "m3_dynamic_direct_fusion",
                dynamic_root / "m3_dynamic_direct_fusion" / suffix,
                False,
            ),
        ],
    }
    result = references[model]
    if refresh_policy != "refresh_every_1":
        result = [
            *result,
            (
                "same_model_refresh_every_1",
                atomic_root / "refresh_every_1" / model / suffix,
                True,
            ),
        ]
    return result


def collect_rows(
    outputs_root: Path,
    camera_id: str,
    participants: list[str],
    seeds: list[int],
    train_scopes: list[str],
    refresh_policies: list[str],
    require_complete_grid: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    missing_required: set[str] = set()
    missing_optional: set[str] = set()
    for participant in participants:
        for seed in seeds:
            seed_root = (
                outputs_root
                / f"{participant}_as_test"
                / f"cam_{camera_id}"
                / f"seed_{seed}"
            )
            for train_scope in train_scopes:
                atomic_root = (
                    seed_root
                    / "history_models"
                    / "atomic_tail_graph_valid"
                    / train_scope
                )
                for refresh_policy in refresh_policies:
                    for model in ATOMIC_TAIL_MODELS:
                        for split in SPLITS:
                            metric_path = (
                                atomic_root
                                / refresh_policy
                                / model
                                / "test_results"
                                / f"{split}_metrics.json"
                            )
                            if not metric_path.is_file():
                                missing_required.add(str(metric_path))
                                continue
                            metric = read_metric(metric_path)
                            metric_rows.append(
                                {
                                    "participant": participant,
                                    "seed": seed,
                                    "train_scope": train_scope,
                                    "refresh_policy": refresh_policy,
                                    "model": model,
                                    "split": split,
                                    **metric,
                                    "metrics_path": str(metric_path),
                                }
                            )
                            for (
                                reference_name,
                                reference_path,
                                required,
                            ) in reference_paths(
                                seed_root,
                                train_scope,
                                atomic_root,
                                refresh_policy,
                                model,
                                split,
                            ):
                                if not reference_path.is_file():
                                    target = (
                                        missing_required
                                        if required
                                        else missing_optional
                                    )
                                    target.add(str(reference_path))
                                    continue
                                reference_metric = read_metric(reference_path)
                                row: dict[str, Any] = {
                                    "participant": participant,
                                    "seed": seed,
                                    "train_scope": train_scope,
                                    "refresh_policy": refresh_policy,
                                    "model": model,
                                    "reference_model": reference_name,
                                    "reference_required": required,
                                    "split": split,
                                    "atomic_metrics_path": str(metric_path),
                                    "reference_metrics_path": str(
                                        reference_path
                                    ),
                                }
                                for field in METRIC_FIELDS:
                                    row[f"delta_{field}"] = (
                                        float(metric[field])
                                        - float(reference_metric[field])
                                    )
                                delta_rows.append(row)

    if missing_required and require_complete_grid:
        preview = "\n".join(sorted(missing_required)[:20])
        raise FileNotFoundError(
            "Atomic-tail graph-valid grid is incomplete; missing "
            f"{len(missing_required)} required files. First missing paths:\n"
            f"{preview}"
        )
    return metric_rows, delta_rows, sorted(missing_optional)


def build_aggregate(
    metric_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_metrics: dict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in metric_rows:
        grouped_metrics[
            (
                row["participant"],
                row["train_scope"],
                row["refresh_policy"],
                row["model"],
                row["split"],
            )
        ].append(row)
    participant_metrics = {
        key: {
            field: mean([float(row[field]) for row in rows])
            for field in METRIC_FIELDS
        }
        for key, rows in grouped_metrics.items()
    }

    grouped_deltas: dict[
        tuple[str, str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in delta_rows:
        grouped_deltas[
            (
                row["participant"],
                row["train_scope"],
                row["refresh_policy"],
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
            (
                train_scope,
                refresh_policy,
                model,
                reference_model,
                split,
            )
            for (
                _,
                train_scope,
                refresh_policy,
                model,
                reference_model,
                split,
            ) in participant_deltas
        }
    )
    aggregate: list[dict[str, Any]] = []
    for (
        train_scope,
        refresh_policy,
        model,
        reference_model,
        split,
    ) in experiment_keys:
        participants = sorted(
            participant
            for (
                participant,
                scope,
                policy,
                current_model,
                current_split,
            ) in participant_metrics
            if (
                scope,
                policy,
                current_model,
                current_split,
            )
            == (train_scope, refresh_policy, model, split)
            and (
                participant,
                train_scope,
                refresh_policy,
                model,
                reference_model,
                split,
            )
            in participant_deltas
        )
        row: dict[str, Any] = {
            "train_scope": train_scope,
            "refresh_policy": refresh_policy,
            "model": model,
            "reference_model": reference_model,
            "split": split,
            "participant_count": len(participants),
            "participants": ",".join(participants),
        }
        for field in METRIC_FIELDS:
            metric_values = [
                participant_metrics[
                    (
                        participant,
                        train_scope,
                        refresh_policy,
                        model,
                        split,
                    )
                ][field]
                for participant in participants
            ]
            delta_values = [
                participant_deltas[
                    (
                        participant,
                        train_scope,
                        refresh_policy,
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
            "Use --overwrite only for this dedicated atomic-tail summary."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the atomic-tail graph-valid experiment grid."
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
    parser.add_argument(
        "--refresh-policies",
        nargs="+",
        choices=list(REFRESH_POLICIES),
        default=list(REFRESH_POLICIES),
    )
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metric_rows, delta_rows, missing_optional = collect_rows(
        outputs_root=Path(args.outputs_root),
        camera_id=args.camera_id,
        participants=args.participants,
        seeds=args.seeds,
        train_scopes=args.train_scopes,
        refresh_policies=args.refresh_policies,
        require_complete_grid=args.require_complete_grid,
    )
    if not metric_rows:
        raise FileNotFoundError(
            f"No atomic-tail graph-valid results found under {args.outputs_root}"
        )
    aggregate_rows = build_aggregate(metric_rows, delta_rows)
    output_dir = ensure_new_summary_dir(
        Path(args.output_dir),
        overwrite=args.overwrite,
    )
    write_csv(output_dir / "atomic_tail_metrics.csv", metric_rows)
    write_csv(output_dir / "atomic_tail_paired_deltas.csv", delta_rows)
    write_csv(output_dir / "atomic_tail_aggregate.csv", aggregate_rows)
    with (output_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_family": "atomic_tail_graph_valid",
                "participants": args.participants,
                "seeds": args.seeds,
                "train_scopes": args.train_scopes,
                "refresh_policies": args.refresh_policies,
                "models": list(ATOMIC_TAIL_MODELS),
                "metric_rows": len(metric_rows),
                "paired_delta_rows": len(delta_rows),
                "aggregate_rows": len(aggregate_rows),
                "optional_reference_files_missing": missing_optional,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(
        f"metric_rows={len(metric_rows)} paired_rows={len(delta_rows)} "
        f"aggregate_rows={len(aggregate_rows)} "
        f"optional_missing={len(missing_optional)}"
    )
    print(f"Saved atomic-tail summaries to {output_dir}")


if __name__ == "__main__":
    main()
