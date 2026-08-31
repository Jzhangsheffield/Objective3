from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluation import write_probability_evaluation


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.no_grad()
def evaluate_logits_model(
    model,
    loader,
    device: torch.device,
    node_to_tier3: list[int],
    output_dir: str | Path,
    split: str,
    forward_key: str = "batch",
) -> dict:
    model.eval()
    probabilities, rows = [], []
    for batch in loader:
        moved = move_batch(batch, device)
        if forward_key == "imu":
            logits = model(moved["imu"])
        else:
            value = model(moved)
            logits = value[0] if isinstance(value, tuple) else value
        probabilities.append(F.softmax(logits, dim=-1).cpu())
        if "rows" in batch:
            rows.extend(batch["rows"])
        elif "row" in batch:
            row_value = batch["row"]
            if isinstance(row_value, list):
                rows.extend(row_value)
    if not probabilities:
        raise ValueError(f"Empty loader for {split}")
    return write_probability_evaluation(rows, torch.cat(probabilities), node_to_tier3, output_dir, split)
