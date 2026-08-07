from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BoundaryChunkDataset, collate_chunks
from .features import load_feature_cache
from .metrics import evaluate_run
from .models import CausalBoundaryTCN, compute_loss
from .online import run_state_machine
from .utils import safe_torch_load, write_json


def build_model(cfg: dict[str, Any]) -> CausalBoundaryTCN:
    return CausalBoundaryTCN(
        feature_dim=int(cfg["feature_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    )


def make_loader(cache_root: Path, runs: list[str], cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = BoundaryChunkDataset(
        cache_root, runs, int(cfg["chunk_length_steps"]), int(cfg["chunk_overlap_steps"]),
    )
    return DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=shuffle,
        num_workers=int(cfg["num_workers"]), collate_fn=collate_chunks,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        with torch.set_grad_enabled(training):
            outputs = model(tensors["features"])
            loss, values = compute_loss(
                outputs, tensors, loss_cfg["weights"], loss_cfg["positive_weights"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(loss_cfg["gradient_clip_norm"]))
                optimizer.step()
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_boundary_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_boundary_checkpoint(path: str | Path, device: torch.device):
    checkpoint = safe_torch_load(path, device)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def infer_run(model: torch.nn.Module, cache: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    features = cache["features"].float().unsqueeze(0).to(device)
    outputs = model(features)
    return {
        "state_probability": torch.softmax(outputs["state_logits"], dim=-1)[0, :, 1].cpu().numpy(),
        "start_probability": torch.sigmoid(outputs["start_logits"])[0].cpu().numpy(),
        "end_probability": torch.sigmoid(outputs["end_logits"])[0].cpu().numpy(),
    }


def evaluate_caches(
    model: torch.nn.Module,
    cache_root: str | Path,
    runs: list[str],
    device: torch.device,
    online_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    per_run: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for run in runs:
        cache = load_feature_cache(Path(cache_root) / f"{run}.pt")
        probabilities = infer_run(model, cache, device)
        segments = run_state_machine(**probabilities, settings=online_cfg)
        pred_state = np.zeros(len(cache["state"]), dtype=np.int64)
        for segment in segments:
            pred_state[segment["start_index"] : segment["end_index"] + 1] = 1
        original = cache["original_frame_idx"].numpy()
        gt_state = cache["state"].numpy()
        gt_segments = []
        start = None
        for index, value in enumerate(list(gt_state) + [0]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                gt_segments.append((start, index - 1)); start = None
        gt_starts = [int(original[index]) for index in np.flatnonzero(cache["exact_start"].numpy() > 0)]
        gt_ends = [int(original[index]) for index in np.flatnonzero(cache["exact_end"].numpy() > 0)]
        pred_segments_anchor = [(int(x["start_index"]), int(x["end_index"])) for x in segments]
        pred_segments_frames = [(int(original[x["start_index"]]), int(original[x["end_index"]])) for x in segments]
        per_run[run] = evaluate_run(
            gt_state, pred_state, gt_starts, gt_ends, pred_segments_anchor,
            [int(x) for x in evaluation_cfg["boundary_tolerance_frames"]],
            [segment[0] for segment in pred_segments_frames],
            [segment[1] for segment in pred_segments_frames],
        )
        emission_delays = [
            int(original[min(x["emitted_at_index"], len(original) - 1)]) - int(original[x["end_index"]])
            for x in segments
        ]
        per_run[run]["emission_delay_frames"] = float(np.mean(emission_delays)) if emission_delays else float("nan")
        for segment, frame_segment in zip(segments, pred_segments_frames):
            prediction_rows.append({"sample_name": run, **segment, "start_original_frame_idx": frame_segment[0], "end_original_frame_idx": frame_segment[1]})
    metrics = _macro_average(per_run)
    result = {"runs": len(runs), "macro": metrics, "per_run": per_run}
    write_json(output_root / "metrics.json", result)
    with (output_root / "predicted_segments.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result


def _macro_average(per_run: dict[str, Any]) -> dict[str, Any]:
    if not per_run:
        return {}
    state_keys = ("precision", "recall", "f1", "accuracy")
    result: dict[str, Any] = {
        "frame_state": {key: float(np.mean([row["frame_state"][key] for row in per_run.values()])) for key in state_keys},
        "edit_score": float(np.mean([row["edit_score"] for row in per_run.values()])),
        "emission_delay_frames": float(np.nanmean([row["emission_delay_frames"] for row in per_run.values()])),
        "segmental_f1": {},
        "boundary": {},
    }
    for threshold in next(iter(per_run.values()))["segmental_f1"]:
        result["segmental_f1"][threshold] = {
            key: float(np.mean([row["segmental_f1"][threshold][key] for row in per_run.values()]))
            for key in ("precision", "recall", "f1")
        }
    for tolerance in next(iter(per_run.values()))["boundary"]:
        result["boundary"][tolerance] = {}
        for kind in ("start", "end"):
            result["boundary"][tolerance][kind] = {
                key: float(np.nanmean([row["boundary"][tolerance][kind][key] for row in per_run.values()]))
                for key in ("precision", "recall", "f1", "mean_signed_error_frames", "mean_absolute_error_frames")
            }
    return result
