from __future__ import annotations

import time

import torch

from .engine import compute_loss, forward_node_model, move_batch_to_device


def train_epoch_shuffled_feature_model(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    epochs: int,
    action_loss_weight: float = 0.0,
    amp: bool = False,
) -> list[dict[str, float]]:
    """Train with one deterministic graph-valid resample per sample and epoch."""
    if getattr(loader, "persistent_workers", False):
        raise ValueError(
            "Epoch-shuffled training requires persistent_workers=False so worker "
            "datasets receive the updated epoch."
        )
    if not hasattr(loader.dataset, "set_epoch"):
        raise TypeError("Epoch-shuffled loader dataset must implement set_epoch(epoch)")

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    node_to_tier3 = node_to_tier3.to(device)
    for epoch in range(1, int(epochs) + 1):
        loader.dataset.set_epoch(epoch)
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
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
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
            "history_shuffle_epoch": float(epoch),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{epochs:03d} "
            f"loss={row['train_loss']:.6f} "
            f"node_acc={row['train_node_accuracy']:.4f} "
            f"graph_shuffle_epoch={epoch:03d} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return history
