from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .io import write_csv, write_json
from .metrics import aggregate_node_probabilities, classification_metrics, top_confusions


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def train_direct(
    model,
    loader,
    device: torch.device,
    target_key: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    accumulation_steps: int,
) -> tuple[list[dict[str, float]], torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum = correct = count = 0
        for step, raw_batch in enumerate(loader, 1):
            batch = move_batch(raw_batch, device)
            logits = model(batch["signal"])
            loss = F.cross_entropy(logits, batch[target_key])
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = int(logits.shape[0])
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(-1) == batch[target_key]).sum())
            count += batch_size
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(1, count),
            "target_accuracy": correct / max(1, count),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        log.append(row)
        print(
            f"epoch={epoch:03d}/{epochs:03d} loss={row['loss']:.6f} target_acc={row['target_accuracy']:.4f}",
            flush=True,
        )
    return log, optimizer


@torch.no_grad()
def evaluate_direct(
    model,
    loader,
    device: torch.device,
    task: str,
    node_to_tier3: list[int],
    output_dir: str | Path,
    split: str,
) -> dict[str, Any]:
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_true: list[int] = []
    node_pred: list[int] = []
    tier3_true: list[int] = []
    tier3_pred: list[int] = []
    stages: list[int] = []
    rows: list[dict[str, Any]] = []
    probabilities = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch["signal"])
        probability = F.softmax(logits, dim=-1)
        probabilities.append(probability.cpu())
        if task == "direct_node":
            node_probability = probability
            tier3_probability = aggregate_node_probabilities(node_probability, node_to_tier3)
            predicted_node = node_probability.argmax(-1)
            predicted_tier3 = tier3_probability.argmax(-1)
        elif task == "direct_tier3":
            node_probability = None
            tier3_probability = probability
            predicted_node = None
            predicted_tier3 = tier3_probability.argmax(-1)
        else:
            raise ValueError(task)
        for index in range(logits.shape[0]):
            true_node = int(batch["node_target"][index])
            true_tier3 = int(batch["tier3_target"][index])
            pred_tier3 = int(predicted_tier3[index])
            stage = int(batch["stage_id"][index])
            tier3_true.append(true_tier3)
            tier3_pred.append(pred_tier3)
            stages.append(stage)
            if predicted_node is not None:
                pred_node = int(predicted_node[index])
                node_true.append(true_node)
                node_pred.append(pred_node)
                node_confidence: float | str = float(node_probability[index, pred_node])
                pred_node_output: int | str = pred_node + 1
            else:
                node_confidence = ""
                pred_node_output = ""
            rows.append({
                "sample_name": raw_batch["sample_name"][index],
                "participant": raw_batch["participant"][index],
                "run": raw_batch["run"][index],
                "annotation_row_index": raw_batch["annotation_row_index"][index],
                "stage_id": stage,
                "true_node_idx": true_node + 1,
                "pred_node_idx": pred_node_output,
                "true_tier3_id": true_tier3,
                "pred_tier3_id": pred_tier3,
                "node_confidence": node_confidence,
                "tier3_confidence": float(tier3_probability[index, pred_tier3]),
            })
    node_metrics = classification_metrics(node_true, node_pred, 35) if task == "direct_node" else None
    metrics: dict[str, Any] = {
        "split": split,
        "task": task,
        "samples": len(rows),
        "node": node_metrics,
        "tier3": classification_metrics(tier3_true, tier3_pred, 31),
        "top_12_node_confusions": top_confusions(node_true, node_pred, 12) if node_metrics else [],
        "top_12_tier3_confusions": top_confusions(tier3_true, tier3_pred, 12),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        selected = [index for index, value in enumerate(stages) if value == stage]
        stage_node = None
        if task == "direct_node":
            stage_node = classification_metrics(
                [node_true[index] for index in selected], [node_pred[index] for index in selected], 35
            )
        metrics["per_stage"][str(stage)] = {
            "samples": len(selected),
            "node": stage_node,
            "tier3": classification_metrics(
                [tier3_true[index] for index in selected], [tier3_pred[index] for index in selected], 31
            ),
        }
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_csv(output_dir / f"{split}_predictions.csv", rows)
    key = "node_probabilities" if task == "direct_node" else "tier3_probabilities"
    classes = 35 if task == "direct_node" else 31
    torch.save({key: torch.cat(probabilities) if probabilities else torch.empty((0, classes)), "rows": rows},
               output_dir / f"{split}_probabilities.pt")
    return metrics
