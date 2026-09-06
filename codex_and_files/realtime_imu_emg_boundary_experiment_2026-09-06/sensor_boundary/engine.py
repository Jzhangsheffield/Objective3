from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SensorChunkDataset, collate_chunks
from .metrics import evaluate_run
from .models import CausalSensorBoundaryModel, compute_loss
from .online import run_state_machine
from .utils import read_json, safe_torch_load, write_json


def compute_normalization(cache_root: str | Path, runs: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    total = total_sq = None
    count = 0
    channel_names: list[str] | None = None
    validity: list[int] = []
    for run in runs:
        meta = read_json(Path(cache_root) / f"{run}.json")
        names = [str(x) for x in meta["channel_names"]]
        if channel_names is None:
            channel_names = names
            validity = [int(x) for x in meta.get("validity_channel_indices", [])]
            total = np.zeros(len(names), dtype=np.float64)
            total_sq = np.zeros(len(names), dtype=np.float64)
        if names != channel_names:
            raise ValueError(f"Channel mismatch in {run}")
        total += np.asarray(meta["stat_sum"], dtype=np.float64)
        total_sq += np.asarray(meta["stat_sumsq"], dtype=np.float64)
        count += int(meta["stat_count"])
    if not count or total is None or total_sq is None or channel_names is None:
        raise ValueError("No training statistics available")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    for index in validity:
        mean[index], std[index] = 0.0, 1.0
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32)), channel_names


def build_model(cfg: dict[str, Any], mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> CausalSensorBoundaryModel:
    return CausalSensorBoundaryModel(cfg, mean, std)


def make_loader(cache_root: Path, runs: list[str], cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = SensorChunkDataset(
        cache_root,
        runs,
        int(cfg["chunk_length_steps"]),
        int(cfg["chunk_overlap_steps"]),
        int(cfg.get("max_cached_runs_per_worker", 2)),
    )
    workers = int(cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_chunks,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if workers > 0 else None,
    )


def run_epoch(model, loader, device, loss_cfg, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
        with torch.set_grad_enabled(training):
            outputs = model(tensors["windows"])
            loss, values = compute_loss(outputs, tensors, loss_cfg)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(loss_cfg["gradient_clip_norm"]))
                optimizer.step()
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, metrics: dict, channel_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "channel_names": channel_names,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = safe_torch_load(path, device)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def infer_run(model, cache: dict[str, Any], device: torch.device, chunk_steps: int) -> dict[str, np.ndarray]:
    windows = cache["windows"].float()
    state_parts, start_parts, end_parts = [], [], []
    context = model.receptive_field_steps - 1
    for begin in range(0, len(windows), chunk_steps):
        context_begin = max(0, begin - context)
        end = min(len(windows), begin + chunk_steps)
        outputs = model(windows[context_begin:end].unsqueeze(0).to(device))
        keep = begin - context_begin
        state_parts.append(torch.softmax(outputs["state_logits"], dim=-1)[0, keep:, 1].cpu())
        start_parts.append(torch.sigmoid(outputs["start_logits"])[0, keep:].cpu())
        end_parts.append(torch.sigmoid(outputs["end_logits"])[0, keep:].cpu())
    return {
        "state_probability": torch.cat(state_parts).numpy(),
        "start_probability": torch.cat(start_parts).numpy(),
        "end_probability": torch.cat(end_parts).numpy(),
    }


def evaluate_caches(model, cache_root, runs, device, online_cfg, evaluation_cfg, output_root) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    per_run: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    total_compute = 0.0
    total_duration = 0.0
    for run in runs:
        cache = safe_torch_load(Path(cache_root) / f"{run}.pt")
        started = time.perf_counter()
        probabilities = infer_run(model, cache, device, int(evaluation_cfg["inference_chunk_steps"]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds = time.perf_counter() - started
        segments = run_state_machine(**probabilities, settings=online_cfg)
        times = cache["times"].numpy()
        pred_state = np.zeros(len(times), dtype=np.int64)
        for segment in segments:
            pred_state[segment["start_index"] : segment["end_index"] + 1] = 1
        gt_start_times = times[np.flatnonzero(cache["exact_start"].numpy() > 0)].tolist()
        gt_end_times = times[np.flatnonzero(cache["exact_end"].numpy() > 0)].tolist()
        pred_pairs = [(int(x["start_index"]), int(x["end_index"])) for x in segments]
        pred_start_times = [float(times[x[0]]) for x in pred_pairs]
        pred_end_times = [float(times[x[1]]) for x in pred_pairs]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        metrics = evaluate_run(
            cache["state"].numpy(), pred_state, cache["segment_id"].numpy(),
            gt_start_times, gt_end_times, pred_pairs, pred_start_times, pred_end_times,
            [int(x) for x in evaluation_cfg["boundary_tolerance_ms"]], duration,
        )
        delays = [1000.0 * (float(times[min(x["emitted_at_index"], len(times) - 1)]) - float(times[x["end_index"]])) for x in segments]
        metrics["emission_delay_ms"] = float(np.mean(delays)) if delays else None
        metrics["compute_seconds"] = compute_seconds
        metrics["real_time_factor"] = compute_seconds / max(duration, 1e-9)
        per_run[run] = metrics
        total_compute += compute_seconds
        total_duration += duration
        for segment, start_time, end_time in zip(segments, pred_start_times, pred_end_times):
            prediction_rows.append({"sample_name": run, **segment, "start_time": start_time, "end_time": end_time})
    result = {
        "runs": len(runs),
        "macro": macro_average(per_run),
        "total_compute_seconds": total_compute,
        "total_duration_seconds": total_duration,
        "overall_real_time_factor": total_compute / max(total_duration, 1e-9),
        "per_run": per_run,
    }
    write_json(output_root / "metrics.json", result)
    with (output_root / "predicted_segments.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(x) for x in values if x is not None]
    return float(np.mean(present)) if present else None


def macro_average(per_run: dict[str, Any]) -> dict[str, Any]:
    if not per_run:
        return {}
    rows = list(per_run.values())
    result: dict[str, Any] = {
        "frame_state": {key: float(np.mean([row["frame_state"][key] for row in rows])) for key in ("precision", "recall", "f1", "accuracy")},
        "edit_score": float(np.mean([row["edit_score"] for row in rows])),
        "emission_delay_ms": _mean_present([row["emission_delay_ms"] for row in rows]),
        "predicted_segments_per_minute": float(np.mean([row["predicted_segments_per_minute"] for row in rows])),
        "short_fragment_fraction_lt_200ms": float(np.mean([row["short_fragment_fraction_lt_200ms"] for row in rows])),
        "segmental_f1": {},
        "boundary": {},
    }
    for threshold in rows[0]["segmental_f1"]:
        result["segmental_f1"][threshold] = {
            key: float(np.mean([row["segmental_f1"][threshold][key] for row in rows]))
            for key in ("precision", "recall", "f1")
        }
    for tolerance in rows[0]["boundary"]:
        result["boundary"][tolerance] = {}
        for kind in ("start", "end"):
            result["boundary"][tolerance][kind] = {
                key: _mean_present([row["boundary"][tolerance][kind][key] for row in rows])
                for key in ("precision", "recall", "f1", "median_signed_error_ms", "median_absolute_error_ms", "p90_absolute_error_ms")
            }
    return result
