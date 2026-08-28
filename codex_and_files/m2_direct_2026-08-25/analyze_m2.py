from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXPERIMENT_ROOT = WORKSPACE / "codex_and_files" / "graph_history_rgb_cross_person_ADM_2026-07-22"
OUTPUTS_ROOT = EXPERIMENT_ROOT / "outputs"
GRAPH_PATH = EXPERIMENT_ROOT / "assets" / "integrated_task_graph_latest.json"

PARTICIPANTS = ("A", "D", "J", "M")
SEEDS = (1, 2, 42)
SPLITS = ("all", "normal", "fault")
NUM_NODES = 35
NUM_TIER3 = 31


def prediction_path(participant: str, seed: int, split: str) -> Path:
    return (
        OUTPUTS_ROOT / f"{participant}_as_test" / "cam_001484412812" / f"seed_{seed}"
        / "history_models" / "direct_head_fusion" / "all_runs" / "m2_direct"
        / "test_results" / f"test_{split}_predictions.csv"
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, np.ndarray | float]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    total = int(matrix.sum())
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def load_graph() -> tuple[dict[int, dict], dict[int, str], dict[int, int]]:
    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes: dict[int, dict] = {}
    tier3_labels: dict[int, str] = {}
    node_to_tier3: dict[int, int] = {}
    for node in payload["nodes"]:
        node_idx = int(node["node_idx"])
        if not 1 <= node_idx <= NUM_NODES:
            continue
        tier3_id = int(node["action_id_tier3"])
        nodes[node_idx] = {"label": str(node["action_label_tier3"]), "tier3_id": tier3_id, "stage": int(node["stage_id"])}
        tier3_labels[tier3_id] = str(node["action_label_tier3"])
        node_to_tier3[node_idx] = tier3_id
    assert set(nodes) == set(range(1, NUM_NODES + 1))
    assert set(tier3_labels) == set(range(NUM_TIER3))
    return nodes, tier3_labels, node_to_tier3


def load_predictions() -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for split in SPLITS:
        for participant in PARTICIPANTS:
            seed_truth: dict[int, pd.DataFrame] = {}
            for seed in SEEDS:
                path = prediction_path(participant, seed, split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                frame["held_out_participant"] = participant
                frame["seed"] = seed
                frame["split"] = split
                frames.append(frame)
                seed_truth[seed] = frame[["sample_name", "true_node_idx", "true_tier3_id"]].sort_values("sample_name").reset_index(drop=True)
                inventory.append({"participant": participant, "seed": seed, "split": split, "rows": int(len(frame)), "path": str(path)})
            reference = seed_truth[SEEDS[0]]
            for seed in SEEDS[1:]:
                if not reference.equals(seed_truth[seed]):
                    raise ValueError(f"Truth rows differ across seeds: participant={participant}, split={split}")
    predictions = pd.concat(frames, ignore_index=True)
    predictions["node_correct"] = (predictions["true_node_idx"] == predictions["pred_node_idx"]).astype(int)
    predictions["tier3_correct"] = (predictions["true_tier3_id"] == predictions["pred_tier3_id"]).astype(int)
    return predictions, inventory


def class_table(predictions: pd.DataFrame, kind: str, labels: dict[int, str], nodes: dict[int, dict]) -> pd.DataFrame:
    if kind == "node":
        true_col, pred_col, class_ids, offset = "true_node_idx", "pred_node_idx", list(range(1, 36)), 1
    else:
        true_col, pred_col, class_ids, offset = "true_tier3_id", "pred_tier3_id", list(range(31)), 0

    seed_metrics: dict[str, list[np.ndarray]] = defaultdict(list)
    split_support: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_frame = predictions[predictions["split"] == split]
        for seed in SEEDS:
            group = split_frame[split_frame["seed"] == seed]
            matrix = confusion(group[true_col].to_numpy() - offset, group[pred_col].to_numpy() - offset, len(class_ids))
            metrics = metrics_from_confusion(matrix)
            for metric in ("precision", "recall", "f1"):
                seed_metrics[f"{split}_{metric}"].append(metrics[metric])
            if split not in split_support:
                split_support[split] = metrics["support"]
            elif not np.array_equal(split_support[split], metrics["support"]):
                raise ValueError(f"Support differs across seeds: {kind}, {split}")

    participant_recall: dict[str, np.ndarray] = {}
    all_frame = predictions[predictions["split"] == "all"]
    for participant in PARTICIPANTS:
        recalls = []
        for seed in SEEDS:
            group = all_frame[(all_frame["held_out_participant"] == participant) & (all_frame["seed"] == seed)]
            matrix = confusion(group[true_col].to_numpy() - offset, group[pred_col].to_numpy() - offset, len(class_ids))
            recalls.append(metrics_from_confusion(matrix)["recall"])
        participant_recall[participant] = np.mean(recalls, axis=0)

    rows = []
    for index, class_id in enumerate(class_ids):
        recalls_by_participant = np.array([participant_recall[p][index] for p in PARTICIPANTS], dtype=float)
        worst_index = int(recalls_by_participant.argmin())
        rows.append({
            f"{kind}_id": class_id,
            "label": labels[class_id],
            "stage": nodes[class_id]["stage"] if kind == "node" else ",".join(str(stage) for stage in sorted({node["stage"] for node in nodes.values() if node["tier3_id"] == class_id})),
            "support_all": int(split_support["all"][index]),
            "precision_all_mean": float(np.mean(seed_metrics["all_precision"], axis=0)[index]),
            "precision_all_sd": float(np.std(seed_metrics["all_precision"], axis=0, ddof=1)[index]),
            "recall_all_mean": float(np.mean(seed_metrics["all_recall"], axis=0)[index]),
            "recall_all_sd": float(np.std(seed_metrics["all_recall"], axis=0, ddof=1)[index]),
            "f1_all_mean": float(np.mean(seed_metrics["all_f1"], axis=0)[index]),
            "f1_all_sd": float(np.std(seed_metrics["all_f1"], axis=0, ddof=1)[index]),
            "support_normal": int(split_support["normal"][index]),
            "recall_normal_mean": float(np.mean(seed_metrics["normal_recall"], axis=0)[index]),
            "support_fault": int(split_support["fault"][index]),
            "recall_fault_mean": float(np.mean(seed_metrics["fault_recall"], axis=0)[index]),
            "worst_participant": PARTICIPANTS[worst_index],
            "worst_participant_recall": float(recalls_by_participant[worst_index]),
            "participant_recall_range": float(recalls_by_participant.max() - recalls_by_participant.min()),
        })
    return pd.DataFrame(rows).sort_values(f"{kind}_id").reset_index(drop=True)


def overall_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_seed_rows = []
    stage_rows = []
    for split in SPLITS:
        for participant in PARTICIPANTS:
            for seed in SEEDS:
                group = predictions[(predictions["split"] == split) & (predictions["held_out_participant"] == participant) & (predictions["seed"] == seed)]
                row = {"split": split, "participant": participant, "seed": seed, "samples": int(len(group))}
                for kind, true_col, pred_col, n_classes, offset in (
                    ("node", "true_node_idx", "pred_node_idx", NUM_NODES, 1),
                    ("tier3", "true_tier3_id", "pred_tier3_id", NUM_TIER3, 0),
                ):
                    matrix = confusion(group[true_col].to_numpy() - offset, group[pred_col].to_numpy() - offset, n_classes)
                    metrics = metrics_from_confusion(matrix)
                    row[f"{kind}_accuracy"] = metrics["accuracy"]
                    row[f"{kind}_macro_f1"] = metrics["macro_f1"]
                    row[f"{kind}_balanced_accuracy"] = metrics["balanced_accuracy"]
                fold_seed_rows.append(row)
                if split == "all":
                    for stage in (1, 2, 3):
                        stage_group = group[group["stage_id"] == stage]
                        stage_rows.append({"participant": participant, "seed": seed, "stage": stage, "samples": int(len(stage_group)), "node_accuracy": float(stage_group["node_correct"].mean()), "tier3_accuracy": float(stage_group["tier3_correct"].mean())})

    fold_seed = pd.DataFrame(fold_seed_rows)
    participant = fold_seed.groupby(["split", "participant"], as_index=False).agg(
        samples=("samples", "first"), node_accuracy=("node_accuracy", "mean"),
        node_macro_f1=("node_macro_f1", "mean"), node_balanced_accuracy=("node_balanced_accuracy", "mean"),
        tier3_accuracy=("tier3_accuracy", "mean"), tier3_macro_f1=("tier3_macro_f1", "mean"),
        tier3_balanced_accuracy=("tier3_balanced_accuracy", "mean"),
    )
    aggregate_rows = []
    for split, group in participant.groupby("split"):
        row = {"split": split, "participants": len(group)}
        for metric in ("node_accuracy", "node_macro_f1", "node_balanced_accuracy", "tier3_accuracy", "tier3_macro_f1", "tier3_balanced_accuracy"):
            row[metric] = float(group[metric].mean())
            row[f"{metric}_participant_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)

    stages = pd.DataFrame(stage_rows).groupby(["participant", "stage"], as_index=False).agg(
        samples=("samples", "first"), node_accuracy=("node_accuracy", "mean"), tier3_accuracy=("tier3_accuracy", "mean")
    ).groupby("stage", as_index=False).agg(
        samples_per_seed=("samples", "sum"), node_accuracy=("node_accuracy", "mean"),
        node_accuracy_participant_sd=("node_accuracy", "std"), tier3_accuracy=("tier3_accuracy", "mean"),
        tier3_accuracy_participant_sd=("tier3_accuracy", "std"),
    )
    return fold_seed, participant, pd.DataFrame(aggregate_rows), stages


def top_confusions(predictions: pd.DataFrame, kind: str, labels: dict[int, str], limit: int = 20) -> pd.DataFrame:
    frame = predictions[predictions["split"] == "all"]
    if kind == "node":
        true_col, pred_col, offset, n_classes = "true_node_idx", "pred_node_idx", 1, NUM_NODES
    else:
        true_col, pred_col, offset, n_classes = "true_tier3_id", "pred_tier3_id", 0, NUM_TIER3
    matrices = []
    for seed in SEEDS:
        group = frame[frame["seed"] == seed]
        matrices.append(confusion(group[true_col].to_numpy() - offset, group[pred_col].to_numpy() - offset, n_classes))
    matrix = np.mean(matrices, axis=0)
    support = matrix.sum(axis=1)
    rows = []
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx or matrix[true_idx, pred_idx] <= 0:
                continue
            true_id, pred_id = true_idx + offset, pred_idx + offset
            rows.append({f"true_{kind}_id": true_id, "true_label": labels[true_id], f"pred_{kind}_id": pred_id, "pred_label": labels[pred_id], "mean_errors_per_seed": float(matrix[true_idx, pred_idx]), "fraction_of_true_class": float(matrix[true_idx, pred_idx] / support[true_idx])})
    return pd.DataFrame(rows).sort_values(["mean_errors_per_seed", "fraction_of_true_class"], ascending=False).head(limit).reset_index(drop=True)


def ambiguity_analysis(predictions: pd.DataFrame, nodes: dict[int, dict], node_to_tier3: dict[int, int]) -> tuple[dict, pd.DataFrame]:
    frame = predictions[predictions["split"] == "all"].copy()
    node_errors = frame[frame["node_correct"] == 0].copy()
    node_errors["same_tier3_node_error"] = [node_to_tier3[int(t)] == node_to_tier3[int(p)] for t, p in zip(node_errors["true_node_idx"], node_errors["pred_node_idx"])]
    repeated_sets = ({14, 21}, {15, 22}, {16, 19}, {17, 20})
    node_errors["repeated_pair_error"] = [any({int(t), int(p)} == pair for pair in repeated_sets) for t, p in zip(node_errors["true_node_idx"], node_errors["pred_node_idx"])]
    summary = {
        "rows_across_3_seeds": int(len(frame)),
        "unique_clips_per_seed": int(len(frame) / len(SEEDS)),
        "node_errors_across_3_seeds": int(len(node_errors)),
        "same_tier3_node_errors_across_3_seeds": int(node_errors["same_tier3_node_error"].sum()),
        "same_tier3_fraction_of_node_errors": float(node_errors["same_tier3_node_error"].mean()),
        "repeated_pair_errors_across_3_seeds": int(node_errors["repeated_pair_error"].sum()),
        "repeated_pair_fraction_of_node_errors": float(node_errors["repeated_pair_error"].mean()),
        "tier3_correct_but_node_wrong_fraction_of_all_predictions": float(((frame["tier3_correct"] == 1) & (frame["node_correct"] == 0)).mean()),
    }
    rows = []
    for true_node, pred_node in ((14, 21), (21, 14), (15, 22), (22, 15), (16, 19), (19, 16), (17, 20), (20, 17)):
        subset = frame[frame["true_node_idx"] == true_node]
        count = int((subset["pred_node_idx"] == pred_node).sum())
        rows.append({"true_node": true_node, "true_label": nodes[true_node]["label"], "pred_node": pred_node, "pred_label": nodes[pred_node]["label"], "errors_across_3_seeds": count, "mean_errors_per_seed": count / len(SEEDS), "support_per_seed": int(len(subset) / len(SEEDS)), "error_rate": float(count / len(subset)) if len(subset) else 0.0})
    return summary, pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8f")


def main() -> None:
    nodes, tier3_labels, node_to_tier3 = load_graph()
    node_labels = {node_id: data["label"] for node_id, data in nodes.items()}
    predictions, inventory = load_predictions()
    fold_seed, participant, aggregate, stages = overall_metrics(predictions)
    tier3 = class_table(predictions, "tier3", tier3_labels, nodes)
    node = class_table(predictions, "node", node_labels, nodes)
    tier3_confusions = top_confusions(predictions, "tier3", tier3_labels)
    node_confusions = top_confusions(predictions, "node", node_labels)
    ambiguity, repeated_pairs = ambiguity_analysis(predictions, nodes, node_to_tier3)

    for name, frame in {
        "m2_fold_seed_metrics.csv": fold_seed, "m2_participant_metrics.csv": participant,
        "m2_overall_metrics.csv": aggregate, "m2_stage_metrics.csv": stages,
        "m2_tier3_per_class_metrics.csv": tier3, "m2_node_per_class_metrics.csv": node,
        "m2_tier3_top_confusions.csv": tier3_confusions, "m2_node_top_confusions.csv": node_confusions,
        "m2_repeated_node_pair_errors.csv": repeated_pairs,
    }.items():
        write_csv(frame, HERE / name)
    with (HERE / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "experiment": {"camera": "001484412812", "model": "m2_direct", "train_scope": "all_runs", "participants": list(PARTICIPANTS), "seeds": list(SEEDS), "prediction_files": len(inventory), "test_all_unique_clips_per_seed": int(len(predictions[predictions["split"] == "all"]) / len(SEEDS))},
            "ambiguity": ambiguity, "inventory": inventory,
        }, handle, ensure_ascii=False, indent=2)

    print("aggregate")
    print(aggregate.to_string(index=False))
    print("\nstages")
    print(stages.to_string(index=False))
    print("\nweakest tier3 by recall")
    print(tier3.sort_values("recall_all_mean").head(12)[["tier3_id", "label", "support_all", "recall_all_mean", "recall_fault_mean"]].to_string(index=False))
    print("\nweakest nodes by recall")
    print(node.sort_values("recall_all_mean").head(15)[["node_id", "label", "support_all", "recall_all_mean", "recall_fault_mean"]].to_string(index=False))
    print("\nambiguity")
    print(json.dumps(ambiguity, ensure_ascii=False, indent=2))
    print("\ntop node confusions")
    print(node_confusions.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
