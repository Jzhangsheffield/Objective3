from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import safe_load
from .io import write_csv, write_json
from .metrics import aggregate_node_probabilities, classification_metrics, top_confusions


def align_probability_files(paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    loaded = [safe_load(path) for path in paths]
    lookups = [{str(row["sample_name"]): index for index, row in enumerate(value["rows"])} for value in loaded]
    names = [str(row["sample_name"]) for row in loaded[0]["rows"]]
    if any(set(lookup) != set(names) for lookup in lookups):
        raise ValueError(f"Probability sample mismatch: {paths}")
    rows = [loaded[0]["rows"][lookups[0][name]] for name in names]
    probabilities = [
        torch.stack([value["node_probabilities"][lookup[name]] for name in names])
        for value, lookup in zip(loaded, lookups)
    ]
    return rows, probabilities


def probability_rows_from_batch(rows: list[dict], probabilities: torch.Tensor) -> list[dict]:
    result = []
    predicted = probabilities.argmax(dim=-1)
    for index, row in enumerate(rows):
        result.append({
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
            "stage_id": int(row["stage_id"]),
            "true_node_idx": int(row.get("true_node_idx", row.get("node_idx"))),
            "pred_node_idx": int(predicted[index]) + 1,
            "true_tier3_id": int(row.get("true_tier3_id", row.get("tier3_id"))),
            "node_confidence": float(probabilities[index, predicted[index]]),
        })
    return result


def write_probability_evaluation(
    rows: list[dict],
    node_probabilities: torch.Tensor,
    node_to_tier3: list[int],
    output_dir: str | Path,
    split: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tier3_probabilities = aggregate_node_probabilities(node_probabilities, node_to_tier3)
    output_rows = probability_rows_from_batch(rows, node_probabilities)
    node_true = [int(row["true_node_idx"]) - 1 for row in output_rows]
    node_pred = node_probabilities.argmax(dim=-1).tolist()
    tier3_true = [int(row["true_tier3_id"]) for row in output_rows]
    tier3_pred = tier3_probabilities.argmax(dim=-1).tolist()
    for index, row in enumerate(output_rows):
        row["pred_tier3_id"] = tier3_pred[index]
        row["tier3_confidence"] = float(tier3_probabilities[index, tier3_pred[index]])
    metrics = {
        "split": split,
        "samples": len(rows),
        "node": classification_metrics(node_true, node_pred, 35),
        "tier3": classification_metrics(tier3_true, tier3_pred, 31),
        "top_12_node_confusions": top_confusions(node_true, node_pred, 12),
        "top_12_tier3_confusions": top_confusions(tier3_true, tier3_pred, 12),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        selected = [index for index, row in enumerate(output_rows) if int(row["stage_id"]) == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(selected),
            "node": classification_metrics([node_true[i] for i in selected], [node_pred[i] for i in selected], 35),
            "tier3": classification_metrics([tier3_true[i] for i in selected], [tier3_pred[i] for i in selected], 31),
        }
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_csv(output_dir / f"{split}_predictions.csv", output_rows)
    torch.save({"node_probabilities": node_probabilities.cpu(), "rows": output_rows},
               output_dir / f"{split}_probabilities.pt")
    return metrics
