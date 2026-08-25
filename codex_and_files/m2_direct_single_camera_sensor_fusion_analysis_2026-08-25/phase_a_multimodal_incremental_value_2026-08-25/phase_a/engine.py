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


def train_model(
    model, loader, device: torch.device, epochs: int, learning_rate: float,
    weight_decay: float, action_loss_weight: float, node_to_tier3: list[int],
    accumulation_steps: int = 1,
) -> tuple[list[dict[str, float]], torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    log = []
    mapping = torch.tensor(node_to_tier3, dtype=torch.long, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        started, loss_sum, correct, count = time.perf_counter(), 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(loader, 1):
            batch = move_batch(raw_batch, device)
            logits, _ = model(batch)
            loss = F.cross_entropy(logits, batch["node_target"])
            if action_loss_weight > 0:
                node_prob = F.softmax(logits, dim=-1)
                tier3_prob = aggregate_node_probabilities(node_prob, node_to_tier3)
                loss = loss + action_loss_weight * F.nll_loss(
                    tier3_prob.clamp_min(1e-8).log(), batch["tier3_target"]
                )
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_count = int(logits.shape[0])
            loss_sum += float(loss.detach()) * batch_count
            correct += int((logits.argmax(-1) == batch["node_target"]).sum())
            count += batch_count
        row = {
            "epoch": epoch, "loss": loss_sum / max(1, count),
            "node_accuracy": correct / max(1, count),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        log.append(row)
        print(f"epoch={epoch:03d}/{epochs:03d} loss={row['loss']:.6f} node_acc={row['node_accuracy']:.4f}", flush=True)
    return log, optimizer


@torch.no_grad()
def evaluate(model, loader, device: torch.device, node_to_tier3: list[int], output_dir: str | Path, split: str):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_true, node_pred, tier3_true, tier3_pred, stages = [], [], [], [], []
    rows, probabilities = [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits, _ = model(batch)
        node_prob = F.softmax(logits, dim=-1)
        tier3_prob = aggregate_node_probabilities(node_prob, node_to_tier3)
        predicted_node, predicted_tier3 = node_prob.argmax(-1), tier3_prob.argmax(-1)
        probabilities.append(node_prob.cpu())
        for index in range(logits.shape[0]):
            nt, np = int(batch["node_target"][index]), int(predicted_node[index])
            tt, tp = int(batch["tier3_target"][index]), int(predicted_tier3[index])
            stage = int(batch["stage_id"][index])
            node_true.append(nt); node_pred.append(np); tier3_true.append(tt); tier3_pred.append(tp); stages.append(stage)
            rows.append({
                "sample_name": raw_batch["sample_name"][index],
                "participant": raw_batch["participant"][index], "run": raw_batch["run"][index],
                "annotation_row_index": raw_batch["annotation_row_index"][index], "stage_id": stage,
                "true_node_idx": nt + 1, "pred_node_idx": np + 1,
                "true_tier3_id": tt, "pred_tier3_id": tp,
                "node_confidence": float(node_prob[index, np]),
                "tier3_confidence": float(tier3_prob[index, tp]),
            })
    metrics = {
        "split": split, "samples": len(rows),
        "node": classification_metrics(node_true, node_pred, 35),
        "tier3": classification_metrics(tier3_true, tier3_pred, 31),
        "top_12_node_confusions": top_confusions(node_true, node_pred, 12),
        "top_12_tier3_confusions": top_confusions(tier3_true, tier3_pred, 12),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        selected = [index for index, value in enumerate(stages) if value == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(selected),
            "node": classification_metrics([node_true[i] for i in selected], [node_pred[i] for i in selected], 35),
            "tier3": classification_metrics([tier3_true[i] for i in selected], [tier3_pred[i] for i in selected], 31),
        }
    write_json(output_dir / f"{split}_metrics.json", metrics)
    write_csv(output_dir / f"{split}_predictions.csv", rows)
    torch.save({
        "node_probabilities": torch.cat(probabilities) if probabilities else torch.empty((0, 35)),
        "rows": rows,
    }, output_dir / f"{split}_probabilities.pt")
    return metrics
