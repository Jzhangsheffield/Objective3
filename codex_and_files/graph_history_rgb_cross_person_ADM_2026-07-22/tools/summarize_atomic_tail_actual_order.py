from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("test_normal", "test_fault", "test_all")
POLICIES = ("refresh_every_1", "refresh_every_10", "refresh_once")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize actual-order evaluations of existing Atomic-tail Direct "
            "Fusion checkpoints and pair them with same-seed M2 Direct results."
        )
    )
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--atomic-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-id", default="001484412812")
    parser.add_argument(
        "--participants", nargs="+", default=["A", "D", "J", "M"]
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
        choices=list(POLICIES),
        default=list(POLICIES),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    atomic_root = Path(args.atomic_root or outputs_root / "at_ad").resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Summary directory is not empty: {output_dir}. "
            "Use --overwrite only for this dedicated summary."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for participant in args.participants:
        for seed in args.seeds:
            seed_root = (
                outputs_root
                / f"{participant}_as_test"
                / f"cam_{args.camera_id}"
                / f"seed_{seed}"
            )
            for scope in args.train_scopes:
                m2_root = (
                    seed_root
                    / "history_models"
                    / "direct_head_fusion"
                    / scope
                    / "m2_direct"
                    / "test_results"
                )
                for policy in args.refresh_policies:
                    actual_root = (
                        atomic_root
                        / f"{participant}_s{seed}"
                        / scope
                        / policy
                        / "m3_atomic_tail_direct_fusion"
                        / "test_results_actual_order"
                    )
                    for split in SPLITS:
                        atomic_path = actual_root / f"{split}_metrics.json"
                        m2_path = m2_root / f"{split}_metrics.json"
                        if not atomic_path.is_file():
                            missing.append(str(atomic_path))
                            continue
                        if not m2_path.is_file():
                            missing.append(str(m2_path))
                            continue
                        atomic = read_json(atomic_path)
                        m2 = read_json(m2_path)
                        node = float(atomic["node"]["accuracy"])
                        tier3 = float(atomic["tier3"]["accuracy"])
                        m2_node = float(m2["node"]["accuracy"])
                        m2_tier3 = float(m2["tier3"]["accuracy"])
                        rows.append(
                            {
                                "participant": participant,
                                "seed": seed,
                                "train_scope": scope,
                                "refresh_policy": policy,
                                "split": split,
                                "atomic_node_accuracy": node,
                                "m2_direct_node_accuracy": m2_node,
                                "delta_node_accuracy": node - m2_node,
                                "atomic_tier3_accuracy": tier3,
                                "m2_direct_tier3_accuracy": m2_tier3,
                                "delta_tier3_accuracy": tier3 - m2_tier3,
                                "atomic_metrics_path": str(atomic_path),
                                "m2_metrics_path": str(m2_path),
                            }
                        )

    if missing:
        raise FileNotFoundError(
            f"Incomplete requested actual-order grid ({len(missing)} missing files):\n"
            + "\n".join(missing[:30])
        )
    if not rows:
        raise FileNotFoundError("No actual-order Atomic-tail metrics were found")

    seed_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    participant_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_groups[
            (
                row["train_scope"],
                row["refresh_policy"],
                row["split"],
                row["seed"],
            )
        ].append(row)
        participant_groups[
            (
                row["train_scope"],
                row["refresh_policy"],
                row["split"],
                row["participant"],
            )
        ].append(row)

    metric_fields = (
        "atomic_node_accuracy",
        "m2_direct_node_accuracy",
        "delta_node_accuracy",
        "atomic_tier3_accuracy",
        "m2_direct_tier3_accuracy",
        "delta_tier3_accuracy",
    )
    seed_rows: list[dict[str, Any]] = []
    for (scope, policy, split, seed), group in sorted(seed_groups.items()):
        row: dict[str, Any] = {
            "train_scope": scope,
            "refresh_policy": policy,
            "split": split,
            "seed": seed,
            "participants": len(group),
        }
        for field in metric_fields:
            row[field] = mean([float(item[field]) for item in group])
        seed_rows.append(row)

    participant_rows: list[dict[str, Any]] = []
    for (scope, policy, split, participant), group in sorted(
        participant_groups.items()
    ):
        row = {
            "train_scope": scope,
            "refresh_policy": policy,
            "split": split,
            "participant": participant,
            "seeds": len(group),
        }
        for field in metric_fields:
            row[field] = mean([float(item[field]) for item in group])
        participant_rows.append(row)

    overall_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        overall_groups[
            (row["train_scope"], row["refresh_policy"], row["split"])
        ].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (scope, policy, split), group in sorted(overall_groups.items()):
        row = {
            "train_scope": scope,
            "refresh_policy": policy,
            "split": split,
            "participants": len(group),
            "seeds_per_participant": len(args.seeds),
        }
        for field in metric_fields:
            values = [float(item[field]) for item in group]
            row[f"mean_{field}"] = mean(values)
            row[f"participant_std_{field}"] = sample_std(values)
        aggregate_rows.append(row)

    write_csv(output_dir / "atomic_actual_order_metrics.csv", rows)
    write_csv(output_dir / "atomic_actual_order_seed_aggregate.csv", seed_rows)
    write_csv(
        output_dir / "atomic_actual_order_participant_aggregate.csv",
        participant_rows,
    )
    write_csv(output_dir / "atomic_actual_order_aggregate.csv", aggregate_rows)
    (output_dir / "completed.json").write_text(
        json.dumps(
            {
                "experiment_family": "atomic_tail_actual_order_evaluation",
                "participants": args.participants,
                "seeds": args.seeds,
                "train_scopes": args.train_scopes,
                "refresh_policies": args.refresh_policies,
                "splits": list(SPLITS),
                "metric_rows": len(rows),
                "seed_aggregate_rows": len(seed_rows),
                "participant_aggregate_rows": len(participant_rows),
                "aggregate_rows": len(aggregate_rows),
                "evaluation_history_order": "actual_chronological",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"metrics={len(rows)} seed_rows={len(seed_rows)} "
        f"participant_rows={len(participant_rows)} aggregate_rows={len(aggregate_rows)}"
    )
    print(f"Saved actual-order summary: {output_dir}")


if __name__ == "__main__":
    main()
