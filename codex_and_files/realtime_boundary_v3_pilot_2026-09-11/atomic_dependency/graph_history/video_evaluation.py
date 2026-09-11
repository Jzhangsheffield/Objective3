from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .constants import NUM_GRAPH_NODES, NUM_TIER3_CLASSES
from .metrics import aggregate_node_probabilities, classification_metrics
from .utils import ensure_dir, write_json


def _stage_metrics(
    truth: list[int], prediction: list[int], stages: list[int], num_classes: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stage in (1, 2, 3):
        indices = [index for index, value in enumerate(stages) if value == stage]
        output[str(stage)] = {
            "samples": len(indices),
            "metrics": classification_metrics(
                [truth[index] for index in indices],
                [prediction[index] for index in indices],
                num_classes,
            ),
        }
    return output


def _write_predictions(output_dir: Path, split_name: str, rows: list[dict[str, Any]]) -> None:
    with (output_dir / f"{split_name}_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def evaluate_tier3_video_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: str | Path,
    split_name: str,
    model_name: str,
    checkpoint: str | Path,
    amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    stages: list[int] = []
    rows: list[dict[str, Any]] = []
    probabilities_all: list[torch.Tensor] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["tier3_target"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(video)
        probabilities = F.softmax(logits.float(), dim=-1)
        predicted = probabilities.argmax(dim=-1)
        probabilities_all.append(probabilities.cpu())
        truth.extend(target.cpu().tolist())
        prediction.extend(predicted.cpu().tolist())
        batch_stages = batch["stage_id"].cpu().tolist()
        stages.extend(batch_stages)
        for index in range(target.shape[0]):
            pred = int(predicted[index])
            rows.append(
                {
                    "sample_name": batch["sample_name"][index],
                    "participant": batch["participant"][index],
                    "run": batch["run"][index],
                    "annotation_row_index": int(batch["annotation_row_index"][index]),
                    "stage_id": int(batch_stages[index]),
                    "true_tier3_id": int(target[index]),
                    "pred_tier3_id": pred,
                    "tier3_confidence": float(probabilities[index, pred]),
                }
            )
    metrics = {
        "model": model_name,
        "target_space": "tier3",
        "checkpoint": str(checkpoint),
        "split": split_name,
        "samples": len(rows),
        "node": None,
        "tier3": classification_metrics(truth, prediction, NUM_TIER3_CLASSES),
        "per_stage": _stage_metrics(truth, prediction, stages, NUM_TIER3_CLASSES),
    }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    _write_predictions(output_dir, split_name, rows)
    torch.save(
        {
            "tier3_probabilities": torch.cat(probabilities_all, dim=0)
            if probabilities_all
            else torch.empty((0, NUM_TIER3_CLASSES)),
            "rows": rows,
        },
        output_dir / f"{split_name}_probabilities.pt",
    )
    return metrics


@torch.no_grad()
def evaluate_node_video_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    output_dir: str | Path,
    split_name: str,
    model_name: str,
    checkpoint: str | Path,
    amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    node_to_tier3 = node_to_tier3.to(device)
    node_truth: list[int] = []
    node_prediction: list[int] = []
    tier3_truth: list[int] = []
    tier3_prediction: list[int] = []
    stages: list[int] = []
    rows: list[dict[str, Any]] = []
    node_probabilities_all: list[torch.Tensor] = []
    tier3_probabilities_all: list[torch.Tensor] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        node_target = batch["node_target"].to(device, non_blocking=True)
        tier3_target = batch["tier3_target"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(video)
        node_probabilities = F.softmax(logits.float(), dim=-1)
        tier3_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, NUM_TIER3_CLASSES
        )
        predicted_nodes = node_probabilities.argmax(dim=-1)
        predicted_tier3 = tier3_probabilities.argmax(dim=-1)
        node_probabilities_all.append(node_probabilities.cpu())
        tier3_probabilities_all.append(tier3_probabilities.cpu())
        node_truth.extend(node_target.cpu().tolist())
        node_prediction.extend(predicted_nodes.cpu().tolist())
        tier3_truth.extend(tier3_target.cpu().tolist())
        tier3_prediction.extend(predicted_tier3.cpu().tolist())
        batch_stages = batch["stage_id"].cpu().tolist()
        stages.extend(batch_stages)
        for index in range(node_target.shape[0]):
            pred_node = int(predicted_nodes[index])
            pred_tier3 = int(predicted_tier3[index])
            rows.append(
                {
                    "sample_name": batch["sample_name"][index],
                    "participant": batch["participant"][index],
                    "run": batch["run"][index],
                    "annotation_row_index": int(batch["annotation_row_index"][index]),
                    "stage_id": int(batch_stages[index]),
                    "true_node_idx": int(node_target[index]) + 1,
                    "pred_node_idx": pred_node + 1,
                    "true_tier3_id": int(tier3_target[index]),
                    "pred_tier3_id": pred_tier3,
                    "node_confidence": float(node_probabilities[index, pred_node]),
                    "tier3_confidence": float(tier3_probabilities[index, pred_tier3]),
                }
            )
    metrics = {
        "model": model_name,
        "target_space": "node",
        "checkpoint": str(checkpoint),
        "split": split_name,
        "samples": len(rows),
        "node": classification_metrics(node_truth, node_prediction, NUM_GRAPH_NODES),
        "tier3": classification_metrics(tier3_truth, tier3_prediction, NUM_TIER3_CLASSES),
        "per_stage": {},
    }
    node_stage = _stage_metrics(node_truth, node_prediction, stages, NUM_GRAPH_NODES)
    tier3_stage = _stage_metrics(tier3_truth, tier3_prediction, stages, NUM_TIER3_CLASSES)
    for stage in ("1", "2", "3"):
        metrics["per_stage"][stage] = {
            "samples": node_stage[stage]["samples"],
            "node": node_stage[stage]["metrics"],
            "tier3": tier3_stage[stage]["metrics"],
        }
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    _write_predictions(output_dir, split_name, rows)
    torch.save(
        {
            "node_probabilities": torch.cat(node_probabilities_all, dim=0)
            if node_probabilities_all
            else torch.empty((0, NUM_GRAPH_NODES)),
            "tier3_probabilities": torch.cat(tier3_probabilities_all, dim=0)
            if tier3_probabilities_all
            else torch.empty((0, NUM_TIER3_CLASSES)),
            "rows": rows,
        },
        output_dir / f"{split_name}_probabilities.pt",
    )
    return metrics
