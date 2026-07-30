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
SEEDS = (1, 2, 42)
SCOPES = ("normal_only", "all_runs")
SPLITS = ("test_normal", "test_fault", "test_all")
DYNAMIC_MODELS = (
    "m3_dynamic_frozen_m0_delta",
    "m3_dynamic_joint_head_delta",
    "m3_dynamic_direct_fusion",
)
REFERENCE_MODELS = ("m0", "m3", "m3_direct")
REPEATED_NODE_PAIRS = ((14, 21), (15, 22), (16, 19), (17, 20))


def io_path(path: Path) -> Path:
    """Use the Win32 extended-length prefix for experiment paths near MAX_PATH."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def node_labels(task_graph_path: Path) -> tuple[dict[int, str], dict[int, str], set[int]]:
    payload = json.loads(task_graph_path.read_text(encoding="utf-8"))
    node_label: dict[int, str] = {}
    tier3_label: dict[int, str] = {}
    immediate_targets: set[int] = set()
    for node in payload["nodes"]:
        node_idx = int(node["node_idx"])
        if not 1 <= node_idx <= 35:
            continue
        label = str(node["action_label_tier3"])
        node_label[node_idx] = label
        tier3_label[int(node["action_id_tier3"])] = label
        immediate = node["execution_constraints"].get(
            "must_immediately_previous_node"
        )
        if immediate is not None:
            immediate_targets.add(node_idx)
    return node_label, tier3_label, immediate_targets


def dynamic_prediction_path(
    outputs_root: Path,
    participant: str,
    seed: int,
    scope: str,
    model: str,
    split: str,
) -> Path:
    return (
        outputs_root
        / f"{participant}_as_test"
        / "cam_001484412812"
        / f"seed_{seed}"
        / "history_models"
        / "dynamic_epoch_shuffle"
        / scope
        / model
        / "test_results"
        / f"{split}_predictions.csv"
    )


def reference_prediction_path(
    outputs_root: Path,
    participant: str,
    seed: int,
    scope: str,
    model: str,
    split: str,
) -> Path:
    seed_root = (
        outputs_root
        / f"{participant}_as_test"
        / "cam_001484412812"
        / f"seed_{seed}"
    )
    if model == "m3_direct":
        return (
            seed_root
            / "history_models"
            / "direct_head_fusion"
            / scope
            / model
            / "test_results"
            / f"{split}_predictions.csv"
        )
    representation = (
        "retrained_normal_only" if scope == "normal_only" else "retrained_all_runs"
    )
    return (
        seed_root
        / "history_models"
        / representation
        / scope
        / model
        / "test_results"
        / f"{split}_predictions.csv"
    )


def load_dynamic_predictions(outputs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for participant in PARTICIPANTS:
        for seed in SEEDS:
            for scope in SCOPES:
                for model in DYNAMIC_MODELS:
                    for split in SPLITS:
                        path = dynamic_prediction_path(
                            outputs_root,
                            participant,
                            seed,
                            scope,
                            model,
                            split,
                        )
                        readable_path = io_path(path)
                        if not readable_path.is_file():
                            missing.append(str(path))
                            continue
                        frame = pd.read_csv(readable_path)
                        frame["held_out_participant"] = participant
                        frame["seed"] = seed
                        frame["train_scope"] = scope
                        frame["model"] = model
                        frame["split"] = split
                        frames.append(frame)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} dynamic prediction files: {missing[:10]}"
        )
    return pd.concat(frames, ignore_index=True)


def load_comparison_predictions(outputs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    models = (*REFERENCE_MODELS, *DYNAMIC_MODELS)
    for participant in PARTICIPANTS:
        for seed in SEEDS:
            for model in models:
                path = (
                    dynamic_prediction_path(
                        outputs_root,
                        participant,
                        seed,
                        "all_runs",
                        model,
                        "test_all",
                    )
                    if model in DYNAMIC_MODELS
                    else reference_prediction_path(
                        outputs_root,
                        participant,
                        seed,
                        "all_runs",
                        model,
                        "test_all",
                    )
                )
                frame = pd.read_csv(io_path(path))
                frame["held_out_participant"] = participant
                frame["seed"] = seed
                frame["train_scope"] = "all_runs"
                frame["model"] = model
                frame["split"] = "test_all"
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_correctness(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["node_correct"] = (
        result["true_node_idx"] == result["pred_node_idx"]
    ).astype(float)
    result["tier3_correct"] = (
        result["true_tier3_id"] == result["pred_tier3_id"]
    ).astype(float)
    return result


def participant_metrics(dynamic: pd.DataFrame) -> pd.DataFrame:
    seed_level = (
        dynamic.groupby(
            [
                "model",
                "train_scope",
                "split",
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
    return (
        seed_level.groupby(
            ["model", "train_scope", "split", "held_out_participant"],
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
    keys = ["model", "train_scope", "split"]
    for key, group in participant_frame.groupby(keys):
        node_values = group["node_accuracy"].astype(float).tolist()
        tier_values = group["tier3_accuracy"].astype(float).tolist()
        rows.append(
            {
                **dict(zip(keys, key)),
                "node_accuracy": float(np.mean(node_values)),
                "node_std": sample_std(node_values),
                "tier3_accuracy": float(np.mean(tier_values)),
                "tier3_std": sample_std(tier_values),
            }
        )
    return pd.DataFrame(rows)


def training_scope_deltas(dynamic: pd.DataFrame) -> pd.DataFrame:
    seed_level = (
        dynamic.groupby(
            ["model", "split", "held_out_participant", "seed", "train_scope"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
        )
    )
    pivot = seed_level.pivot(
        index=["model", "split", "held_out_participant", "seed"],
        columns="train_scope",
        values=["node_accuracy", "tier3_accuracy"],
    )
    rows: list[dict[str, Any]] = []
    for (model, split), group in pivot.groupby(level=[0, 1]):
        node_delta = (
            group[("node_accuracy", "all_runs")]
            - group[("node_accuracy", "normal_only")]
        )
        tier_delta = (
            group[("tier3_accuracy", "all_runs")]
            - group[("tier3_accuracy", "normal_only")]
        )
        rows.append(
            {
                "model": model,
                "split": split,
                "mean_node_delta": float(node_delta.mean()),
                "node_positive_pairs": int((node_delta > 0).sum()),
                "node_total_pairs": int(node_delta.size),
                "mean_tier3_delta": float(tier_delta.mean()),
                "tier3_positive_pairs": int((tier_delta > 0).sum()),
                "tier3_total_pairs": int(tier_delta.size),
            }
        )
    return pd.DataFrame(rows)


def seed_stability(dynamic: pd.DataFrame) -> pd.DataFrame:
    seed_level = (
        dynamic.groupby(
            ["model", "train_scope", "split", "held_out_participant", "seed"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
        )
    )
    return (
        seed_level.groupby(["model", "train_scope", "split", "seed"], as_index=False)
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
        )
    )


def stage_metrics(dynamic: pd.DataFrame) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
    ]
    seed_level = (
        focus.groupby(
            ["model", "stage_id", "held_out_participant", "seed"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
            samples=("sample_name", "size"),
        )
    )
    participant_level = (
        seed_level.groupby(
            ["model", "stage_id", "held_out_participant"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
            samples=("samples", "max"),
        )
    )
    return (
        participant_level.groupby(["model", "stage_id"], as_index=False)
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
            participant_count=("held_out_participant", "nunique"),
            unique_samples=("samples", "sum"),
        )
    )


def node_recall(
    dynamic: pd.DataFrame,
    node_label: dict[int, str],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
    ]
    seed_level = (
        focus.groupby(
            ["model", "true_node_idx", "held_out_participant", "seed"],
            as_index=False,
        )
        .agg(recall=("node_correct", "mean"))
    )
    participant_level = (
        seed_level.groupby(
            ["model", "true_node_idx", "held_out_participant"],
            as_index=False,
        )
        .agg(recall=("recall", "mean"))
    )
    result = (
        participant_level.groupby(["model", "true_node_idx"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            participant_std=("recall", "std"),
            participant_count=("held_out_participant", "nunique"),
        )
    )
    unique_support = (
        focus[focus["seed"] == 1]
        .groupby(["model", "true_node_idx"], as_index=False)
        .agg(unique_support=("sample_name", "size"))
    )
    result = result.merge(unique_support, on=["model", "true_node_idx"])
    result["node_label"] = result["true_node_idx"].map(node_label)
    return result


def participant_hardest_nodes(
    dynamic: pd.DataFrame,
    node_label: dict[int, str],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
        & (dynamic["model"] == "m3_dynamic_direct_fusion")
    ]
    result = (
        focus.groupby(
            ["held_out_participant", "true_node_idx"],
            as_index=False,
        )
        .agg(
            recall=("node_correct", "mean"),
            repeated_support=("sample_name", "size"),
        )
    )
    result["unique_support"] = result["repeated_support"] // len(SEEDS)
    result["node_label"] = result["true_node_idx"].map(node_label)
    return (
        result.sort_values(
            ["held_out_participant", "recall", "true_node_idx"]
        )
        .groupby("held_out_participant", as_index=False)
        .head(5)
    )


def tier3_recall(
    dynamic: pd.DataFrame,
    tier3_label: dict[int, str],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
    ]
    seed_level = (
        focus.groupby(
            ["model", "true_tier3_id", "held_out_participant", "seed"],
            as_index=False,
        )
        .agg(recall=("tier3_correct", "mean"))
    )
    participant_level = (
        seed_level.groupby(
            ["model", "true_tier3_id", "held_out_participant"],
            as_index=False,
        )
        .agg(recall=("recall", "mean"))
    )
    result = (
        participant_level.groupby(["model", "true_tier3_id"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            participant_std=("recall", "std"),
            participant_count=("held_out_participant", "nunique"),
        )
    )
    unique_support = (
        focus[focus["seed"] == 1]
        .groupby(["model", "true_tier3_id"], as_index=False)
        .agg(unique_support=("sample_name", "size"))
    )
    result = result.merge(unique_support, on=["model", "true_tier3_id"])
    result["tier3_label"] = result["true_tier3_id"].map(tier3_label)
    return result


def top_tier3_confusions(
    dynamic: pd.DataFrame,
    tier3_label: dict[int, str],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
        & (dynamic["tier3_correct"] == 0)
    ]
    grouped = (
        focus.groupby(
            ["model", "true_tier3_id", "pred_tier3_id"],
            as_index=False,
        )
        .agg(errors=("sample_name", "size"))
    )
    grouped["true_label"] = grouped["true_tier3_id"].map(tier3_label)
    grouped["pred_label"] = grouped["pred_tier3_id"].map(tier3_label)
    return (
        grouped.sort_values(["model", "errors"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(10)
    )


def top_confusions(
    dynamic: pd.DataFrame,
    node_label: dict[int, str],
    tier3_by_node: dict[int, int],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
        & (dynamic["node_correct"] == 0)
    ].copy()
    grouped = (
        focus.groupby(
            ["model", "true_node_idx", "pred_node_idx"],
            as_index=False,
        )
        .agg(errors=("sample_name", "size"))
    )
    grouped["true_label"] = grouped["true_node_idx"].map(node_label)
    grouped["pred_label"] = grouped["pred_node_idx"].map(node_label)
    grouped["same_tier3"] = grouped.apply(
        lambda row: (
            tier3_by_node[int(row["true_node_idx"])]
            == tier3_by_node[int(row["pred_node_idx"])]
        ),
        axis=1,
    )
    return (
        grouped.sort_values(["model", "errors"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(12)
    )


def repeated_pair_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_frame in comparison.groupby("model"):
        for left, right in REPEATED_NODE_PAIRS:
            errors = model_frame[
                (
                    (model_frame["true_node_idx"] == left)
                    & (model_frame["pred_node_idx"] == right)
                )
                | (
                    (model_frame["true_node_idx"] == right)
                    & (model_frame["pred_node_idx"] == left)
                )
            ]
            rows.append(
                {
                    "model": model,
                    "node_pair": f"{left}<->{right}",
                    "bidirectional_errors": int(len(errors)),
                }
            )
    return pd.DataFrame(rows)


def immediate_group_metrics(
    comparison: pd.DataFrame,
    immediate_targets: set[int],
) -> pd.DataFrame:
    focus = comparison.copy()
    focus["target_group"] = np.where(
        focus["true_node_idx"].isin(immediate_targets),
        "immediate_target",
        "other_target",
    )
    seed_level = (
        focus.groupby(
            ["model", "target_group", "held_out_participant", "seed"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
        )
    )
    participant_level = (
        seed_level.groupby(
            ["model", "target_group", "held_out_participant"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
        )
    )
    return (
        participant_level.groupby(["model", "target_group"], as_index=False)
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
        )
    )


def selected_target_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    target_sets = {
        "node_11": {11},
        "stage2_nodes_13_25": set(range(13, 26)),
        "four_repeated_pairs": {
            node_idx for pair in REPEATED_NODE_PAIRS for node_idx in pair
        },
    }
    rows: list[dict[str, Any]] = []
    for group_name, target_nodes in target_sets.items():
        focus = comparison[comparison["true_node_idx"].isin(target_nodes)]
        seed_level = (
            focus.groupby(
                ["model", "held_out_participant", "seed"],
                as_index=False,
            )
            .agg(
                node_accuracy=("node_correct", "mean"),
                tier3_accuracy=("tier3_correct", "mean"),
            )
        )
        participant_level = (
            seed_level.groupby(
                ["model", "held_out_participant"],
                as_index=False,
            )
            .agg(
                node_accuracy=("node_accuracy", "mean"),
                tier3_accuracy=("tier3_accuracy", "mean"),
            )
        )
        for model, model_frame in participant_level.groupby("model"):
            rows.append(
                {
                    "target_group": group_name,
                    "model": model,
                    "node_accuracy": float(
                        model_frame["node_accuracy"].mean()
                    ),
                    "tier3_accuracy": float(
                        model_frame["tier3_accuracy"].mean()
                    ),
                    "participant_count": int(
                        model_frame["held_out_participant"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_metrics(dynamic: pd.DataFrame) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
    ]
    seed_level = (
        focus.groupby(
            ["model", "held_out_participant", "run", "seed"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
            samples=("sample_name", "size"),
        )
    )
    return (
        seed_level.groupby(
            ["model", "held_out_participant", "run"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
            samples=("samples", "max"),
        )
    )


def run_level_comparisons(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    pairs = (
        (
            "m3_dynamic_frozen_m0_delta",
            "m3",
            "dynamic_frozen_vs_static_m3",
        ),
        (
            "m3_dynamic_direct_fusion",
            "m3_direct",
            "dynamic_direct_vs_static_direct",
        ),
        (
            "m3_dynamic_joint_head_delta",
            "m3_dynamic_frozen_m0_delta",
            "dynamic_joint_vs_dynamic_frozen",
        ),
    )
    seed_level = (
        comparison.groupby(
            ["model", "held_out_participant", "run", "seed"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_correct", "mean"),
            tier3_accuracy=("tier3_correct", "mean"),
        )
    )
    run_level = (
        seed_level.groupby(
            ["model", "held_out_participant", "run"],
            as_index=False,
        )
        .agg(
            node_accuracy=("node_accuracy", "mean"),
            tier3_accuracy=("tier3_accuracy", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for candidate, reference, label in pairs:
        left = run_level[run_level["model"] == candidate]
        right = run_level[run_level["model"] == reference]
        merged = left.merge(
            right,
            on=["held_out_participant", "run"],
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        for metric in ("node_accuracy", "tier3_accuracy"):
            delta = (
                merged[f"{metric}_candidate"]
                - merged[f"{metric}_reference"]
            )
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "run_count": int(len(delta)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "positive_runs": int((delta > 0).sum()),
                    "tied_runs": int((delta == 0).sum()),
                    "negative_runs": int((delta < 0).sum()),
                }
            )
    return rows


def consensus_failures(
    dynamic: pd.DataFrame,
    node_label: dict[int, str],
) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
        & (dynamic["model"] == "m3_dynamic_direct_fusion")
    ].copy()
    focus["wrong_node_confidence"] = np.where(
        focus["node_correct"] == 0,
        focus["node_confidence"],
        np.nan,
    )
    grouped = (
        focus.groupby(
            [
                "held_out_participant",
                "sample_name",
                "run",
                "stage_id",
                "true_node_idx",
                "true_tier3_id",
            ],
            as_index=False,
        )
        .agg(
            node_correct_seeds=("node_correct", "sum"),
            tier3_correct_seeds=("tier3_correct", "sum"),
            mean_node_confidence=("node_confidence", "mean"),
            mean_wrong_node_confidence=("wrong_node_confidence", "mean"),
            max_wrong_node_confidence=("wrong_node_confidence", "max"),
            predicted_nodes=(
                "pred_node_idx",
                lambda values: ",".join(str(int(value)) for value in values),
            ),
        )
    )
    grouped["node_label"] = grouped["true_node_idx"].map(node_label)
    return grouped[grouped["node_correct_seeds"] == 0].sort_values(
        ["max_wrong_node_confidence", "held_out_participant"],
        ascending=[False, True],
    )


def paired_flips(comparison: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        (
            "m3_dynamic_frozen_m0_delta",
            "m3",
            "dynamic_frozen_vs_static_m3",
        ),
        (
            "m3_dynamic_direct_fusion",
            "m3_direct",
            "dynamic_direct_vs_static_direct",
        ),
        (
            "m3_dynamic_joint_head_delta",
            "m3_dynamic_frozen_m0_delta",
            "dynamic_joint_vs_dynamic_frozen",
        ),
    )
    rows: list[dict[str, Any]] = []
    keys = ["held_out_participant", "seed", "sample_name"]
    for candidate, reference, label in pairs:
        left = comparison[comparison["model"] == candidate][
            keys + ["node_correct", "tier3_correct", "true_node_idx"]
        ]
        right = comparison[comparison["model"] == reference][
            keys + ["node_correct", "tier3_correct"]
        ]
        merged = left.merge(
            right,
            on=keys,
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        rows.append(
            {
                "comparison": label,
                "samples": int(len(merged)),
                "node_corrected": int(
                    (
                        (merged["node_correct_reference"] == 0)
                        & (merged["node_correct_candidate"] == 1)
                    ).sum()
                ),
                "node_regressed": int(
                    (
                        (merged["node_correct_reference"] == 1)
                        & (merged["node_correct_candidate"] == 0)
                    ).sum()
                ),
                "tier3_corrected": int(
                    (
                        (merged["tier3_correct_reference"] == 0)
                        & (merged["tier3_correct_candidate"] == 1)
                    ).sum()
                ),
                "tier3_regressed": int(
                    (
                        (merged["tier3_correct_reference"] == 1)
                        & (merged["tier3_correct_candidate"] == 0)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def confidence_analysis(dynamic: pd.DataFrame) -> pd.DataFrame:
    focus = dynamic[
        (dynamic["train_scope"] == "all_runs")
        & (dynamic["split"] == "test_all")
    ]
    rows: list[dict[str, Any]] = []
    for model, group in focus.groupby("model"):
        confidence = group["node_confidence"].astype(float)
        correct = group["node_correct"].astype(float)
        bin_ids = np.minimum((confidence * 10).astype(int), 9)
        ece = 0.0
        for bin_id in range(10):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
        wrong = group[group["node_correct"] == 0]
        rows.append(
            {
                "model": model,
                "mean_confidence_correct": float(
                    group.loc[group["node_correct"] == 1, "node_confidence"].mean()
                ),
                "mean_confidence_wrong": float(wrong["node_confidence"].mean()),
                "high_confidence_errors_ge_0_9": int(
                    (wrong["node_confidence"] >= 0.9).sum()
                ),
                "total_errors": int(len(wrong)),
                "ece_10bin": ece,
            }
        )
    return pd.DataFrame(rows)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed failure analysis for dynamic epoch-shuffle models"
    )
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = (
        output_dir / "dynamic_failure_participant_metrics.csv",
        output_dir / "dynamic_failure_stage_metrics.csv",
        output_dir / "dynamic_failure_node_recall.csv",
        output_dir / "dynamic_failure_participant_hardest_nodes.csv",
        output_dir / "dynamic_failure_top_confusions.csv",
        output_dir / "dynamic_failure_repeated_pairs.csv",
        output_dir / "dynamic_failure_run_metrics.csv",
        output_dir / "dynamic_failure_consensus_errors.csv",
        output_dir / "dynamic_failure_paired_flips.csv",
        output_dir / "dynamic_failure_analysis.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Detailed failure-analysis outputs already exist. "
            f"Use --overwrite to replace only these files: {existing[:5]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    node_label, tier3_label, immediate_targets = node_labels(
        Path(args.task_graph)
    )
    task_graph = json.loads(Path(args.task_graph).read_text(encoding="utf-8"))
    tier3_by_node = {
        int(node["node_idx"]): int(node["action_id_tier3"])
        for node in task_graph["nodes"]
        if 1 <= int(node["node_idx"]) <= 35
    }
    dynamic = add_correctness(load_dynamic_predictions(Path(args.outputs_root)))
    comparison = add_correctness(
        load_comparison_predictions(Path(args.outputs_root))
    )

    participant = participant_metrics(dynamic)
    overall = overall_metrics(participant)
    scope_delta = training_scope_deltas(dynamic)
    seeds = seed_stability(dynamic)
    stages = stage_metrics(dynamic)
    nodes = node_recall(dynamic, node_label)
    participant_nodes = participant_hardest_nodes(dynamic, node_label)
    tier3_nodes = tier3_recall(dynamic, tier3_label)
    tier3_confusions = top_tier3_confusions(dynamic, tier3_label)
    confusions = top_confusions(dynamic, node_label, tier3_by_node)
    repeated = repeated_pair_errors(comparison)
    immediate = immediate_group_metrics(comparison, immediate_targets)
    selected_targets = selected_target_metrics(comparison)
    runs = run_metrics(dynamic)
    run_comparisons = run_level_comparisons(comparison)
    consensus = consensus_failures(dynamic, node_label)
    flips = paired_flips(comparison)
    confidence = confidence_analysis(dynamic)

    participant.to_csv(targets[0], index=False)
    stages.to_csv(targets[1], index=False)
    nodes.to_csv(targets[2], index=False)
    participant_nodes.to_csv(targets[3], index=False)
    confusions.to_csv(targets[4], index=False)
    repeated.to_csv(targets[5], index=False)
    runs.to_csv(targets[6], index=False)
    consensus.to_csv(targets[7], index=False)
    flips.to_csv(targets[8], index=False)

    report = {
        "dynamic_prediction_rows": int(len(dynamic)),
        "comparison_prediction_rows": int(len(comparison)),
        "node_labels": {str(key): value for key, value in node_label.items()},
        "tier3_labels": {str(key): value for key, value in tier3_label.items()},
        "immediate_target_nodes": sorted(immediate_targets),
        "overall_metrics": to_records(overall),
        "participant_metrics": to_records(participant),
        "training_scope_deltas": to_records(scope_delta),
        "seed_stability": to_records(seeds),
        "stage_metrics": to_records(stages),
        "node_recall": to_records(nodes),
        "participant_hardest_nodes": to_records(participant_nodes),
        "tier3_recall": to_records(tier3_nodes),
        "top_tier3_confusions": to_records(tier3_confusions),
        "top_confusions": to_records(confusions),
        "repeated_pair_errors": to_records(repeated),
        "immediate_group_metrics": to_records(immediate),
        "selected_target_metrics": to_records(selected_targets),
        "run_metrics": to_records(runs),
        "run_level_comparisons": run_comparisons,
        "consensus_errors": to_records(consensus),
        "paired_flips": to_records(flips),
        "confidence_analysis": to_records(confidence),
    }
    targets[9].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"dynamic_rows={len(dynamic)} comparison_rows={len(comparison)} "
        f"consensus_errors={len(consensus)}"
    )
    print(f"Saved detailed failure analysis to {output_dir}")


if __name__ == "__main__":
    main()
