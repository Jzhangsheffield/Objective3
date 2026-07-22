from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .constants import NUM_GRAPH_NODES, NUM_TIER3_CLASSES
from .metrics import aggregate_node_probabilities, classification_metrics
from .models import FeatureNodeClassifier
from .utils import ensure_dir, write_json


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_node_model(model: torch.nn.Module, batch: dict[str, Any]):
    if isinstance(model, FeatureNodeClassifier):
        logits = model(batch["current_feature"])
        return logits, {}
    return model(
        current_feature=batch["current_feature"],
        history_features=batch["history_features"],
        history_position_ids=batch["history_position_ids"],
        history_node_classes=batch["history_node_classes"],
        history_padding_mask=batch["history_padding_mask"],
    )


def compute_loss(
    logits: torch.Tensor,
    node_target: torch.Tensor,
    tier3_target: torch.Tensor,
    node_to_tier3: torch.Tensor,
    action_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    node_loss = F.cross_entropy(logits, node_target)
    loss = node_loss
    action_loss = logits.new_tensor(0.0)
    if action_loss_weight > 0:
        node_probabilities = F.softmax(logits, dim=-1)
        action_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, NUM_TIER3_CLASSES
        )
        selected = action_probabilities.gather(1, tier3_target[:, None]).squeeze(1).clamp_min(1e-12)
        action_loss = -selected.log().mean()
        loss = loss + float(action_loss_weight) * action_loss
    return loss, {
        "node_loss": float(node_loss.detach()),
        "action_loss": float(action_loss.detach()),
    }


def train_feature_model(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    epochs: int,
    action_loss_weight: float = 0.0,
    amp: bool = False,
) -> list[dict[str, float]]:
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    node_to_tier3 = node_to_tier3.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        correct = 0
        total = 0
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits, _ = forward_node_model(model, batch)
                loss, _ = compute_loss(
                    logits,
                    batch["node_target"],
                    batch["tier3_target"],
                    node_to_tier3,
                    action_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["node_target"].shape[0])
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=-1) == batch["node_target"]).sum())
            total += batch_size
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(1, total),
            "train_node_accuracy": correct / max(1, total),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{epochs:03d} "
            f"loss={row['train_loss']:.6f} node_acc={row['train_node_accuracy']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return history


@torch.no_grad()
def evaluate_feature_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    output_dir: str | Path,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    node_to_tier3 = node_to_tier3.to(device)
    node_true: list[int] = []
    node_pred: list[int] = []
    tier3_true: list[int] = []
    tier3_pred: list[int] = []
    stages: list[int] = []
    rows: list[dict[str, Any]] = []
    all_node_probabilities: list[torch.Tensor] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        logits, _ = forward_node_model(model, batch)
        node_probabilities = F.softmax(logits, dim=-1)
        tier3_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, NUM_TIER3_CLASSES
        )
        predicted_nodes = node_probabilities.argmax(dim=-1)
        predicted_actions = tier3_probabilities.argmax(dim=-1)
        all_node_probabilities.append(node_probabilities.cpu())

        for index in range(logits.shape[0]):
            truth_node = int(batch["node_target"][index])
            pred_node = int(predicted_nodes[index])
            truth_action = int(batch["tier3_target"][index])
            pred_action = int(predicted_actions[index])
            stage = int(batch["stage_id"][index])
            node_true.append(truth_node)
            node_pred.append(pred_node)
            tier3_true.append(truth_action)
            tier3_pred.append(pred_action)
            stages.append(stage)
            rows.append(
                {
                    "sample_name": raw_batch["sample_name"][index],
                    "participant": raw_batch["participant"][index],
                    "run": raw_batch["run"][index],
                    "annotation_row_index": raw_batch["annotation_row_index"][index],
                    "stage_id": stage,
                    "true_node_idx": truth_node + 1,
                    "pred_node_idx": pred_node + 1,
                    "true_tier3_id": truth_action,
                    "pred_tier3_id": pred_action,
                    "node_confidence": float(node_probabilities[index, pred_node]),
                    "tier3_confidence": float(tier3_probabilities[index, pred_action]),
                }
            )

    metrics: dict[str, Any] = {
        "split": split_name,
        "samples": len(rows),
        "node": classification_metrics(node_true, node_pred, NUM_GRAPH_NODES),
        "tier3": classification_metrics(tier3_true, tier3_pred, NUM_TIER3_CLASSES),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        indices = [idx for idx, value in enumerate(stages) if value == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(indices),
            "node": classification_metrics(
                [node_true[idx] for idx in indices],
                [node_pred[idx] for idx in indices],
                NUM_GRAPH_NODES,
            ),
            "tier3": classification_metrics(
                [tier3_true[idx] for idx in indices],
                [tier3_pred[idx] for idx in indices],
                NUM_TIER3_CLASSES,
            ),
        }

    output_dir = ensure_dir(output_dir)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    with (output_dir / f"{split_name}_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    torch.save(
        {
            "node_probabilities": torch.cat(all_node_probabilities, dim=0)
            if all_node_probabilities else torch.empty((0, NUM_GRAPH_NODES)),
            "rows": rows,
        },
        output_dir / f"{split_name}_probabilities.pt",
    )
    return metrics

