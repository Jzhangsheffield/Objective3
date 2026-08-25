from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .cache import safe_load
from .io import write_csv, write_json
from .metrics import aggregate_node_probabilities, classification_metrics, top_confusions


def align_probability_files(paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    loaded = [safe_load(path) for path in paths]
    lookups = [{str(row["sample_name"]): index for index, row in enumerate(item["rows"])} for item in loaded]
    names = [str(row["sample_name"]) for row in loaded[0]["rows"]]
    expected = set(names)
    for path, lookup in zip(paths, lookups):
        if set(lookup) != expected:
            raise ValueError(f"Probability sample mismatch: {path}")
    rows = [loaded[0]["rows"][lookups[0][name]] for name in names]
    probabilities = [torch.stack([item["node_probabilities"][lookup[name]] for name in names])
                     for item, lookup in zip(loaded, lookups)]
    return rows, probabilities


def write_probability_evaluation(
    rows: list[dict[str, Any]], node_probabilities: torch.Tensor,
    node_to_tier3: list[int], output_dir: str | Path, split: str,
) -> dict[str, Any]:
    tier3_probabilities = aggregate_node_probabilities(node_probabilities, node_to_tier3)
    predicted_nodes = node_probabilities.argmax(-1).tolist()
    predicted_tier3 = tier3_probabilities.argmax(-1).tolist()
    node_true = [int(row["true_node_idx"]) - 1 for row in rows]
    tier3_true = [int(row["true_tier3_id"]) for row in rows]
    stages = [int(row["stage_id"]) for row in rows]
    output_rows = []
    for index, row in enumerate(rows):
        node, tier3 = predicted_nodes[index], predicted_tier3[index]
        output_rows.append({
            "sample_name": row["sample_name"], "participant": row["participant"],
            "run": row["run"], "annotation_row_index": row["annotation_row_index"],
            "stage_id": stages[index], "true_node_idx": node_true[index] + 1,
            "pred_node_idx": node + 1, "true_tier3_id": tier3_true[index],
            "pred_tier3_id": tier3,
            "node_confidence": float(node_probabilities[index, node]),
            "tier3_confidence": float(tier3_probabilities[index, tier3]),
        })
    metrics = {
        "split": split, "samples": len(rows),
        "node": classification_metrics(node_true, predicted_nodes, 35),
        "tier3": classification_metrics(tier3_true, predicted_tier3, 31),
        "top_12_node_confusions": top_confusions(node_true, predicted_nodes, 12),
        "top_12_tier3_confusions": top_confusions(tier3_true, predicted_tier3, 12),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        selected = [i for i, value in enumerate(stages) if value == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(selected),
            "node": classification_metrics([node_true[i] for i in selected], [predicted_nodes[i] for i in selected], 35),
            "tier3": classification_metrics([tier3_true[i] for i in selected], [predicted_tier3[i] for i in selected], 31),
        }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_csv(output_dir / f"{split}_predictions.csv", output_rows)
    torch.save({"node_probabilities": node_probabilities, "rows": output_rows},
               output_dir / f"{split}_probabilities.pt")
    return metrics
