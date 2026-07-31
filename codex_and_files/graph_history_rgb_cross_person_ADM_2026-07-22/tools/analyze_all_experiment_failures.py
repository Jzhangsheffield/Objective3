from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARTICIPANTS = ("A", "D", "J", "M")
ATOMIC_PARTICIPANTS = ("A", "D")
SEEDS = (1, 2, 42)
REPEATED_NODE_PAIRS = ((14, 21), (15, 22), (16, 19), (17, 20))

LEGACY_MODELS = tuple(f"m{index}" for index in range(7))
DIRECT_MODELS = ("m1_direct", "m2_direct", "m3_direct")
DYNAMIC_MODELS = (
    "m3_dynamic_frozen_m0_delta",
    "m3_dynamic_joint_head_delta",
    "m3_dynamic_direct_fusion",
)
E2E_NODE_MODELS = ("e2e_node_scratch", "e2e_node_from_tier3")
ATOMIC_POLICIES = ("refresh_every_1", "refresh_every_10", "refresh_once")
ATOMIC_MODEL_NAMES = tuple(f"atomic_direct_{policy}" for policy in ATOMIC_POLICIES)

MODEL_DISPLAY_NAMES = {
    **{model: model.upper() for model in LEGACY_MODELS},
    "m1_direct": "M1 Direct",
    "m2_direct": "M2 Direct",
    "m3_direct": "M3 Direct",
    "m3_dynamic_frozen_m0_delta": "Dynamic Frozen-M0 Delta",
    "m3_dynamic_joint_head_delta": "Dynamic Joint-Head Delta",
    "m3_dynamic_direct_fusion": "Dynamic Direct",
    "e2e_node_scratch": "E2E Node Scratch",
    "e2e_node_from_tier3": "E2E Node From Tier3",
    "atomic_direct_refresh_every_1": "Atomic Direct / every 1",
    "atomic_direct_refresh_every_10": "Atomic Direct / every 10",
    "atomic_direct_refresh_once": "Atomic Direct / once",
}


def io_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def read_graph(
    task_graph_path: Path,
) -> tuple[dict[int, str], dict[int, int], dict[int, str], set[int]]:
    payload = json.loads(task_graph_path.read_text(encoding="utf-8"))
    node_labels: dict[int, str] = {}
    node_to_tier3: dict[int, int] = {}
    tier3_labels: dict[int, str] = {}
    immediate_targets: set[int] = set()
    for node in payload["nodes"]:
        node_idx = int(node["node_idx"])
        if not 1 <= node_idx <= 35:
            continue
        tier3_id = int(node["action_id_tier3"])
        label = str(node["action_label_tier3"])
        node_labels[node_idx] = label
        node_to_tier3[node_idx] = tier3_id
        tier3_labels[tier3_id] = label
        immediate = node["execution_constraints"].get(
            "must_immediately_previous_node"
        )
        if immediate is not None:
            immediate_targets.add(node_idx)
    return node_labels, node_to_tier3, tier3_labels, immediate_targets


def standard_prediction_path(
    outputs_root: Path,
    participant: str,
    seed: int,
    model: str,
) -> Path:
    seed_root = (
        outputs_root
        / f"{participant}_as_test"
        / "cam_001484412812"
        / f"seed_{seed}"
    )
    if model in LEGACY_MODELS:
        return (
            seed_root
            / "history_models"
            / "retrained_all_runs"
            / "all_runs"
            / model
            / "test_results"
            / "test_all_predictions.csv"
        )
    if model in DIRECT_MODELS:
        return (
            seed_root
            / "history_models"
            / "direct_head_fusion"
            / "all_runs"
            / model
            / "test_results"
            / "test_all_predictions.csv"
        )
    if model in DYNAMIC_MODELS:
        return (
            seed_root
            / "history_models"
            / "dynamic_epoch_shuffle"
            / "all_runs"
            / model
            / "test_results"
            / "test_all_predictions.csv"
        )
    if model in E2E_NODE_MODELS:
        return (
            seed_root
            / "e2e_baselines"
            / "all_runs"
            / model
            / "test_results"
            / "test_all_predictions.csv"
        )
    raise KeyError(model)


def atomic_prediction_path(
    outputs_root: Path,
    participant: str,
    seed: int,
    policy: str,
) -> Path:
    return (
        outputs_root
        / "at_ad"
        / f"{participant}_s{seed}"
        / "all_runs"
        / policy
        / "m3_atomic_tail_direct_fusion"
        / "test_results"
        / "test_all_predictions.csv"
    )


def load_predictions(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    missing: list[str] = []
    standard_models = (
        *LEGACY_MODELS,
        *DIRECT_MODELS,
        *DYNAMIC_MODELS,
        *E2E_NODE_MODELS,
    )
    for model in standard_models:
        for participant in PARTICIPANTS:
            for seed in SEEDS:
                path = standard_prediction_path(
                    outputs_root, participant, seed, model
                )
                readable = io_path(path)
                if not readable.is_file():
                    missing.append(str(path))
                    continue
                frame = pd.read_csv(readable)
                frame["model"] = model
                frame["model_display"] = MODEL_DISPLAY_NAMES[model]
                frame["held_out_participant"] = participant
                frame["seed"] = seed
                frame["coverage"] = "ADJM"
                frames.append(frame)
                inventory.append(
                    {
                        "model": model,
                        "participant": participant,
                        "seed": seed,
                        "prediction_path": str(path),
                        "rows": len(frame),
                    }
                )
    for policy in ATOMIC_POLICIES:
        model = f"atomic_direct_{policy}"
        for participant in ATOMIC_PARTICIPANTS:
            for seed in SEEDS:
                path = atomic_prediction_path(
                    outputs_root, participant, seed, policy
                )
                readable = io_path(path)
                if not readable.is_file():
                    missing.append(str(path))
                    continue
                frame = pd.read_csv(readable)
                frame["model"] = model
                frame["model_display"] = MODEL_DISPLAY_NAMES[model]
                frame["held_out_participant"] = participant
                frame["seed"] = seed
                frame["coverage"] = "AD"
                frames.append(frame)
                inventory.append(
                    {
                        "model": model,
                        "participant": participant,
                        "seed": seed,
                        "prediction_path": str(path),
                        "rows": len(frame),
                    }
                )
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction files:\n"
            + "\n".join(missing[:20])
        )
    predictions = pd.concat(frames, ignore_index=True)
    predictions["node_correct"] = (
        predictions["true_node_idx"] == predictions["pred_node_idx"]
    ).astype(float)
    predictions["tier3_correct"] = (
        predictions["true_tier3_id"] == predictions["pred_tier3_id"]
    ).astype(float)
    predictions["sample_key"] = (
        predictions["held_out_participant"].astype(str)
        + ":"
        + predictions["sample_name"].astype(str)
    )
    return predictions, pd.DataFrame(inventory)


def seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "held_out_participant",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
            samples=("sample_name", "size"),
        )
    )


def participant_metrics(seed_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_frame.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "held_out_participant",
            ],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
            samples=("samples", "max"),
        )
    )


def overall_metrics(participant_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in participant_frame.groupby(
        ["model", "model_display", "coverage"]
    ):
        node_values = group["node_accuracy"].astype(float).tolist()
        tier_values = group["tier3_accuracy"].astype(float).tolist()
        rows.append(
            {
                "model": key[0],
                "model_display": key[1],
                "coverage": key[2],
                "participant_count": len(group),
                "participants": ",".join(
                    sorted(group["held_out_participant"].astype(str))
                ),
                "node_accuracy": float(np.mean(node_values)),
                "node_std": sample_std(node_values),
                "tier3_accuracy": float(np.mean(tier_values)),
                "tier3_std": sample_std(tier_values),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["coverage", "node_accuracy"], ascending=[True, False]
    )


def node_recall_tables(
    predictions: pd.DataFrame,
    node_labels: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_node = (
        predictions.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "held_out_participant",
                "seed",
                "true_node_idx",
            ],
            as_index=False,
        )
        .agg(recall=("node_correct", "mean"))
    )
    participant_node = (
        seed_node.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "held_out_participant",
                "true_node_idx",
            ],
            as_index=False,
        )
        .agg(recall=("recall", "mean"))
    )
    supports = (
        predictions[
            [
                "model",
                "held_out_participant",
                "true_node_idx",
                "sample_key",
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["model", "held_out_participant", "true_node_idx"],
            as_index=False,
        )
        .agg(unique_support=("sample_key", "size"))
    )
    participant_node = participant_node.merge(
        supports,
        on=["model", "held_out_participant", "true_node_idx"],
        how="left",
    )
    participant_node["node_label"] = participant_node["true_node_idx"].map(
        node_labels
    )

    overall = (
        participant_node.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "true_node_idx",
                "node_label",
            ],
            as_index=False,
        )
        .agg(
            recall=("recall", "mean"),
            participant_std=("recall", "std"),
            unique_support=("unique_support", "sum"),
            participant_count=("held_out_participant", "nunique"),
        )
    )
    return participant_node, overall


def confusion_tables(
    predictions: pd.DataFrame,
    node_labels: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = predictions[predictions["node_correct"] == 0].copy()
    overall = (
        errors.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "true_node_idx",
                "pred_node_idx",
            ],
            as_index=False,
        )
        .agg(errors=("sample_name", "size"))
    )
    participant = (
        errors.groupby(
            [
                "model",
                "model_display",
                "coverage",
                "held_out_participant",
                "true_node_idx",
                "pred_node_idx",
            ],
            as_index=False,
        )
        .agg(errors=("sample_name", "size"))
    )
    for frame in (overall, participant):
        frame["true_label"] = frame["true_node_idx"].map(node_labels)
        frame["pred_label"] = frame["pred_node_idx"].map(node_labels)
        frame["same_tier3"] = False
    tier_lookup = (
        predictions[
            ["true_node_idx", "true_tier3_id"]
        ]
        .drop_duplicates()
        .set_index("true_node_idx")["true_tier3_id"]
        .to_dict()
    )
    for frame in (overall, participant):
        frame["same_tier3"] = frame.apply(
            lambda row: tier_lookup.get(int(row["true_node_idx"]))
            == tier_lookup.get(int(row["pred_node_idx"])),
            axis=1,
        )
    return overall, participant


def hardest_node_destinations(
    participant_node: pd.DataFrame,
    overall_node: pd.DataFrame,
    participant_confusions: pd.DataFrame,
    overall_confusions: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def select_destinations(
        recall_frame: pd.DataFrame,
        confusion_frame: pd.DataFrame,
        group_keys: list[str],
    ) -> pd.DataFrame:
        hard = (
            recall_frame.sort_values(
                [*group_keys, "recall", "unique_support"],
                ascending=[*[True] * len(group_keys), True, False],
            )
            .groupby(group_keys, group_keys=False)
            .head(top_n)
            .copy()
        )
        destination = (
            confusion_frame.sort_values(
                [*group_keys, "true_node_idx", "errors"],
                ascending=[*[True] * (len(group_keys) + 1), False],
            )
            .groupby([*group_keys, "true_node_idx"], group_keys=False)
            .head(1)
            .rename(
                columns={
                    "pred_node_idx": "top_pred_node_idx",
                    "pred_label": "top_pred_label",
                    "errors": "top_pred_errors",
                    "same_tier3": "top_pred_same_tier3",
                }
            )
        )
        columns = [
            *group_keys,
            "true_node_idx",
            "top_pred_node_idx",
            "top_pred_label",
            "top_pred_errors",
            "top_pred_same_tier3",
        ]
        return hard.merge(
            destination[columns],
            on=[*group_keys, "true_node_idx"],
            how="left",
        )

    participant_hard = select_destinations(
        participant_node,
        participant_confusions,
        ["model", "model_display", "coverage", "held_out_participant"],
    )
    overall_hard = select_destinations(
        overall_node,
        overall_confusions,
        ["model", "model_display", "coverage"],
    )
    return participant_hard, overall_hard


def error_characteristics(
    predictions: pd.DataFrame,
    immediate_targets: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(
        ["model", "model_display", "coverage"]
    ):
        errors = group[group["node_correct"] == 0]
        same_tier = errors["tier3_correct"].sum()
        high_confidence = (
            errors["node_confidence"].astype(float) >= 0.9
        ).sum()
        immediate = group[
            group["true_node_idx"].astype(int).isin(immediate_targets)
        ]
        rows.append(
            {
                "model": key[0],
                "model_display": key[1],
                "coverage": key[2],
                "prediction_rows": len(group),
                "node_errors": len(errors),
                "same_tier3_node_errors": int(same_tier),
                "same_tier3_error_fraction": (
                    float(same_tier / len(errors)) if len(errors) else 0.0
                ),
                "cross_tier3_node_errors": int(len(errors) - same_tier),
                "cross_tier3_error_fraction": (
                    float(1.0 - same_tier / len(errors))
                    if len(errors)
                    else 0.0
                ),
                "mean_wrong_confidence": (
                    float(errors["node_confidence"].astype(float).mean())
                    if len(errors)
                    else 0.0
                ),
                "high_confidence_errors": int(high_confidence),
                "high_confidence_error_fraction": (
                    float(high_confidence / len(errors))
                    if len(errors)
                    else 0.0
                ),
                "immediate_node_accuracy": float(
                    immediate["node_correct"].mean()
                ),
                "immediate_tier3_accuracy": float(
                    immediate["tier3_correct"].mean()
                ),
            }
        )
        participant_stage = (
            group.groupby(
                ["held_out_participant", "seed", "stage_id"],
                as_index=False,
            )
            .agg(
                node_accuracy=("node_correct", "mean"),
                tier3_accuracy=("tier3_correct", "mean"),
            )
            .groupby(["held_out_participant", "stage_id"], as_index=False)
            .agg(
                node_accuracy=("node_accuracy", "mean"),
                tier3_accuracy=("tier3_accuracy", "mean"),
            )
        )
        for stage_id, stage_group in participant_stage.groupby("stage_id"):
            stage_rows.append(
                {
                    "model": key[0],
                    "model_display": key[1],
                    "coverage": key[2],
                    "stage_id": int(stage_id),
                    "node_accuracy": float(
                        stage_group["node_accuracy"].mean()
                    ),
                    "tier3_accuracy": float(
                        stage_group["tier3_accuracy"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(stage_rows)


def repeated_pair_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(
        ["model", "model_display", "coverage"]
    ):
        for left, right in REPEATED_NODE_PAIRS:
            count = (
                (
                    (group["true_node_idx"] == left)
                    & (group["pred_node_idx"] == right)
                )
                | (
                    (group["true_node_idx"] == right)
                    & (group["pred_node_idx"] == left)
                )
            ).sum()
            rows.append(
                {
                    "model": key[0],
                    "model_display": key[1],
                    "coverage": key[2],
                    "node_pair": f"{left}<->{right}",
                    "bidirectional_errors": int(count),
                }
            )
    return pd.DataFrame(rows)


def atomic_pairwise(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomic_models = list(ATOMIC_MODEL_NAMES)
    references = [
        "m2_direct",
        "m3_direct",
        "m3_dynamic_direct_fusion",
        *atomic_models,
    ]
    subset = predictions[
        predictions["held_out_participant"].isin(ATOMIC_PARTICIPANTS)
        & predictions["model"].isin(set(atomic_models + references))
    ].copy()
    seed_frame = seed_metrics(subset)
    keyed = seed_frame.set_index(
        ["model", "held_out_participant", "seed"]
    )
    rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    sample_keys = [
        "held_out_participant",
        "seed",
        "sample_name",
        "run",
        "annotation_row_index",
    ]
    sample_frame = subset.set_index(["model", *sample_keys])
    for model in atomic_models:
        comparison_refs = [
            "m2_direct",
            "m3_direct",
            "m3_dynamic_direct_fusion",
            *[other for other in atomic_models if other != model],
        ]
        for reference in comparison_refs:
            deltas: list[dict[str, Any]] = []
            for participant in ATOMIC_PARTICIPANTS:
                for seed in SEEDS:
                    current = keyed.loc[(model, participant, seed)]
                    baseline = keyed.loc[(reference, participant, seed)]
                    deltas.append(
                        {
                            "participant": participant,
                            "seed": seed,
                            "node_delta": float(
                                current["node_accuracy"]
                                - baseline["node_accuracy"]
                            ),
                            "tier3_delta": float(
                                current["tier3_accuracy"]
                                - baseline["tier3_accuracy"]
                            ),
                        }
                    )
            delta_frame = pd.DataFrame(deltas)
            rows.append(
                {
                    "model": model,
                    "model_display": MODEL_DISPLAY_NAMES[model],
                    "reference_model": reference,
                    "reference_display": MODEL_DISPLAY_NAMES[reference],
                    "pairs": len(delta_frame),
                    "mean_node_delta": float(
                        delta_frame["node_delta"].mean()
                    ),
                    "node_positive_pairs": int(
                        (delta_frame["node_delta"] > 1e-12).sum()
                    ),
                    "node_tied_pairs": int(
                        (delta_frame["node_delta"].abs() <= 1e-12).sum()
                    ),
                    "mean_tier3_delta": float(
                        delta_frame["tier3_delta"].mean()
                    ),
                    "tier3_positive_pairs": int(
                        (delta_frame["tier3_delta"] > 1e-12).sum()
                    ),
                    "tier3_tied_pairs": int(
                        (delta_frame["tier3_delta"].abs() <= 1e-12).sum()
                    ),
                }
            )
            left = sample_frame.loc[model][
                ["node_correct", "tier3_correct"]
            ].rename(
                columns={
                    "node_correct": "current_node",
                    "tier3_correct": "current_tier3",
                }
            )
            right = sample_frame.loc[reference][
                ["node_correct", "tier3_correct"]
            ].rename(
                columns={
                    "node_correct": "reference_node",
                    "tier3_correct": "reference_tier3",
                }
            )
            joined = left.join(right, how="inner", validate="one_to_one")
            flip_rows.append(
                {
                    "model": model,
                    "reference_model": reference,
                    "samples": len(joined),
                    "node_corrected": int(
                        (
                            (joined["current_node"] == 1)
                            & (joined["reference_node"] == 0)
                        ).sum()
                    ),
                    "node_regressed": int(
                        (
                            (joined["current_node"] == 0)
                            & (joined["reference_node"] == 1)
                        ).sum()
                    ),
                    "tier3_corrected": int(
                        (
                            (joined["current_tier3"] == 1)
                            & (joined["reference_tier3"] == 0)
                        ).sum()
                    ),
                    "tier3_regressed": int(
                        (
                            (joined["current_tier3"] == 0)
                            & (joined["reference_tier3"] == 1)
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(flip_rows)


def atomic_consensus_errors(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    atomic = predictions[predictions["model"].isin(ATOMIC_MODEL_NAMES)]
    rows: list[pd.DataFrame] = []
    for model, group in atomic.groupby("model"):
        consensus = (
            group.groupby(
                [
                    "held_out_participant",
                    "sample_name",
                    "run",
                    "stage_id",
                    "true_node_idx",
                ],
                as_index=False,
            )
            .agg(
                correct_seeds=("node_correct", "sum"),
                mean_confidence=("node_confidence", "mean"),
                predicted_nodes=(
                    "pred_node_idx",
                    lambda values: ",".join(map(str, values)),
                ),
            )
        )
        consensus = consensus[consensus["correct_seeds"] == 0].copy()
        consensus["model"] = model
        rows.append(consensus)
    return pd.concat(rows, ignore_index=True)


def atomic_audits(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for participant in ATOMIC_PARTICIPANTS:
        for seed in SEEDS:
            for policy in ATOMIC_POLICIES:
                path = (
                    outputs_root
                    / "at_ad"
                    / f"{participant}_s{seed}"
                    / "all_runs"
                    / policy
                    / "m3_atomic_tail_direct_fusion"
                    / "shuffle_audit.json"
                )
                audit = json.loads(
                    io_path(path).read_text(encoding="utf-8")
                )
                rows.append(
                    {
                        "participant": participant,
                        "seed": seed,
                        "refresh_policy": policy,
                        "epochs_audited": audit["epochs_audited"],
                        "unique_refresh_rounds": len(
                            audit["unique_refresh_rounds"]
                        ),
                        "total_examples": audit["total_examples"],
                        "atomic_tail_applied": audit[
                            "atomic_tail_applied"
                        ],
                        "atomic_tail_applied_fraction": audit[
                            "atomic_tail_applied_fraction"
                        ],
                        "atomic_tail_violations": audit[
                            "atomic_tail_violations"
                        ],
                        "samples_with_multiple_orders_fraction": audit[
                            "samples_with_multiple_orders_fraction"
                        ],
                        "samples_ever_different_from_actual_fraction": audit[
                            "samples_ever_different_from_actual_fraction"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze all comparable all-runs node models and the reduced "
            "A/D atomic-tail Direct Fusion experiment."
        )
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--task-graph",
        default="assets/integrated_task_graph_latest.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Non-empty analysis directory exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        node_labels,
        _node_to_tier3,
        _tier3_labels,
        immediate_targets,
    ) = read_graph(Path(args.task_graph))
    predictions, inventory = load_predictions(Path(args.outputs_root))
    seeds = seed_metrics(predictions)
    participants = participant_metrics(seeds)
    overall = overall_metrics(participants)
    participant_nodes, overall_nodes = node_recall_tables(
        predictions, node_labels
    )
    overall_confusions, participant_confusions = confusion_tables(
        predictions, node_labels
    )
    participant_hard, overall_hard = hardest_node_destinations(
        participant_nodes,
        overall_nodes,
        participant_confusions,
        overall_confusions,
    )
    characteristics, stages = error_characteristics(
        predictions, immediate_targets
    )
    repeated = repeated_pair_errors(predictions)
    atomic_deltas, atomic_flips = atomic_pairwise(predictions)
    atomic_consensus = atomic_consensus_errors(predictions)
    audits = atomic_audits(Path(args.outputs_root))
    atomic_comparable_models = {
        "m2_direct",
        "m3_direct",
        "m3_dynamic_direct_fusion",
        *ATOMIC_MODEL_NAMES,
    }
    atomic_comparable = predictions[
        predictions["held_out_participant"].isin(ATOMIC_PARTICIPANTS)
        & predictions["model"].isin(atomic_comparable_models)
    ].copy()
    atomic_comparable["coverage"] = "AD-comparable"
    atomic_comparable_seeds = seed_metrics(atomic_comparable)
    atomic_comparable_participants = participant_metrics(
        atomic_comparable_seeds
    )
    atomic_comparable_overall = overall_metrics(
        atomic_comparable_participants
    )
    (
        atomic_comparable_participant_nodes,
        atomic_comparable_nodes,
    ) = node_recall_tables(atomic_comparable, node_labels)
    (
        atomic_comparable_confusions,
        atomic_comparable_participant_confusions,
    ) = confusion_tables(atomic_comparable, node_labels)
    (
        atomic_comparable_participant_hard,
        atomic_comparable_hard,
    ) = hardest_node_destinations(
        atomic_comparable_participant_nodes,
        atomic_comparable_nodes,
        atomic_comparable_participant_confusions,
        atomic_comparable_confusions,
    )
    (
        atomic_comparable_characteristics,
        atomic_comparable_stages,
    ) = error_characteristics(atomic_comparable, immediate_targets)
    atomic_comparable_repeated = repeated_pair_errors(atomic_comparable)

    write_csv(inventory, output_dir / "prediction_inventory.csv")
    write_csv(seeds, output_dir / "seed_metrics.csv")
    write_csv(participants, output_dir / "participant_metrics.csv")
    write_csv(overall, output_dir / "model_overall.csv")
    write_csv(participant_nodes, output_dir / "participant_node_recall.csv")
    write_csv(overall_nodes, output_dir / "model_node_recall.csv")
    write_csv(
        participant_confusions,
        output_dir / "participant_model_confusions.csv",
    )
    write_csv(
        overall_confusions, output_dir / "model_confusions.csv"
    )
    write_csv(
        participant_hard,
        output_dir / "participant_model_hardest_nodes.csv",
    )
    write_csv(overall_hard, output_dir / "model_hardest_nodes.csv")
    write_csv(characteristics, output_dir / "error_characteristics.csv")
    write_csv(stages, output_dir / "stage_metrics.csv")
    write_csv(repeated, output_dir / "repeated_pair_errors.csv")
    write_csv(atomic_deltas, output_dir / "atomic_pairwise_deltas.csv")
    write_csv(atomic_flips, output_dir / "atomic_sample_flips.csv")
    write_csv(
        atomic_consensus, output_dir / "atomic_consensus_errors.csv"
    )
    write_csv(audits, output_dir / "atomic_shuffle_audits.csv")
    write_csv(
        atomic_comparable_overall,
        output_dir / "atomic_comparable_overall.csv",
    )
    write_csv(
        atomic_comparable_participants,
        output_dir / "atomic_comparable_participants.csv",
    )
    write_csv(
        atomic_comparable_nodes,
        output_dir / "atomic_comparable_node_recall.csv",
    )
    write_csv(
        atomic_comparable_hard,
        output_dir / "atomic_comparable_hardest_nodes.csv",
    )
    write_csv(
        atomic_comparable_confusions,
        output_dir / "atomic_comparable_confusions.csv",
    )
    write_csv(
        atomic_comparable_characteristics,
        output_dir / "atomic_comparable_error_characteristics.csv",
    )
    write_csv(
        atomic_comparable_stages,
        output_dir / "atomic_comparable_stage_metrics.csv",
    )
    write_csv(
        atomic_comparable_repeated,
        output_dir / "atomic_comparable_repeated_pair_errors.csv",
    )

    summary = {
        "analysis_scope": "all_runs/test_all",
        "prediction_files": len(inventory),
        "prediction_rows": len(predictions),
        "standard_models": [
            *LEGACY_MODELS,
            *DIRECT_MODELS,
            *DYNAMIC_MODELS,
            *E2E_NODE_MODELS,
        ],
        "atomic_models": list(ATOMIC_MODEL_NAMES),
        "standard_participants": list(PARTICIPANTS),
        "atomic_participants": list(ATOMIC_PARTICIPANTS),
        "seeds": list(SEEDS),
        "atomic_training_units": 18,
        "atomic_metric_files_all_splits": 54,
        "atomic_prediction_files_all_splits": 54,
        "atomic_tail_violations": int(
            audits["atomic_tail_violations"].sum()
        ),
        "output_files": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"prediction_files={len(inventory)} "
        f"prediction_rows={len(predictions)} "
        f"atomic_violations={summary['atomic_tail_violations']}"
    )
    print(f"Saved analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
