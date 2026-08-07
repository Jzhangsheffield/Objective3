from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from .annotations import RunInfo, dilate_binary_targets, load_frame_table
from .utils import safe_torch_load, sha256_file


def _import_atomic_modules(project_root: str | Path):
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from graph_history.backbone import generate_model
    from graph_history.utils import load_compatible_state
    return generate_model, load_compatible_state


def build_frozen_backbone(project_root: str | Path, checkpoint: str | Path, device: torch.device):
    generate_model, load_compatible_state = _import_atomic_modules(project_root)
    model = generate_model(18, num_classes=31)
    report = load_compatible_state(model, checkpoint)
    if report["missing_keys"] or report["unexpected_keys"] or report["loaded_keys"] != report["model_keys"]:
        raise RuntimeError(f"Backbone checkpoint is not an exact architecture match: {report}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, report


def _load_rgb(path: Path, size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    image = read_image(str(path)).float().div_(255.0)
    image = resize(image, [size, size], antialias=True)
    mean_t = torch.tensor(mean, dtype=image.dtype)[:, None, None]
    std_t = torch.tensor(std, dtype=image.dtype)[:, None, None]
    return (image - mean_t) / std_t


def causal_clip_indices(anchor: int, clip_frames: int) -> list[int]:
    return [max(0, anchor - clip_frames + 1 + offset) for offset in range(clip_frames)]


def _events_on_anchor_grid(exact_target, anchors: list[int]) -> torch.Tensor:
    result = torch.zeros(len(anchors), dtype=torch.float32)
    anchor_tensor = torch.tensor(anchors)
    for event in torch.from_numpy(exact_target).nonzero().flatten():
        candidates = (anchor_tensor >= event).nonzero().flatten()
        result[int(candidates[0] if len(candidates) else len(anchors) - 1)] = 1.0
    return result


@torch.inference_mode()
def extract_closed_segment_feature(
    frame_paths: list[Path], start_row: int, end_row: int, model: torch.nn.Module,
    device: torch.device, feature_cfg: dict[str, Any],
) -> torch.Tensor:
    """Extract the M3-compatible 16-frame feature only after a segment has ended."""
    if end_row < start_row:
        raise ValueError("end_row must be >= start_row")
    count = int(feature_cfg["clip_frames"])
    positions = torch.linspace(start_row, end_row, count).round().long().tolist()
    frames = [
        _load_rgb(frame_paths[index], int(feature_cfg["rgb_size"]), list(feature_cfg["mean"]), list(feature_cfg["std"]))
        for index in positions
    ]
    clip = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    return model.forward_features(clip)[0].cpu()


@torch.inference_mode()
def extract_run_features(
    info: RunInfo,
    model: torch.nn.Module,
    output_path: str | Path,
    device: torch.device,
    feature_cfg: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    table = load_frame_table(info)
    stride = int(feature_cfg["stride_frames"])
    anchors = list(range(0, len(table["frame_paths"]), stride))
    batch_size = int(feature_cfg["batch_size"])
    clip_frames = int(feature_cfg["clip_frames"])
    features: list[torch.Tensor] = []
    for batch_start in range(0, len(anchors), batch_size):
        clips = []
        for anchor in anchors[batch_start : batch_start + batch_size]:
            frames = [
                _load_rgb(
                    table["frame_paths"][index],
                    int(feature_cfg["rgb_size"]),
                    list(feature_cfg["mean"]),
                    list(feature_cfg["std"]),
                )
                for index in causal_clip_indices(anchor, clip_frames)
            ]
            clips.append(torch.stack(frames, dim=1))
        batch = torch.stack(clips).to(device, non_blocking=True)
        features.append(model.forward_features(batch).cpu())
    anchor_tensor = torch.tensor(anchors, dtype=torch.long)
    radius_frames = int(feature_cfg.get("boundary_label_radius_frames", 0))
    radius_anchors = (radius_frames + stride - 1) // stride
    anchor_start = _events_on_anchor_grid(table["start"], anchors)
    anchor_end = _events_on_anchor_grid(table["end"], anchors)
    payload = {
        "sample_name": info.sample_name,
        "participant": info.participant,
        "source_run": info.source_run,
        "features": torch.cat(features, dim=0),
        "anchor_row_index": anchor_tensor,
        "frame_idx": torch.from_numpy(table["frame_idx"][anchors]),
        "original_frame_idx": torch.from_numpy(table["original_frame_idx"][anchors]),
        "timestamps": [table["timestamps"][index] for index in anchors],
        "state": torch.from_numpy(table["state"][anchors]).long(),
        "start": torch.from_numpy(dilate_binary_targets(anchor_start.numpy(), radius_anchors)),
        "end": torch.from_numpy(dilate_binary_targets(anchor_end.numpy(), radius_anchors)),
        "exact_start": anchor_start,
        "exact_end": anchor_end,
        "action": [table["action"][index] for index in anchors],
        "object": [table["object"][index] for index in anchors],
        "segment_no": torch.from_numpy(table["segment_no"][anchors]).long(),
        "metadata": {
            "causal": True,
            "clip_frames": clip_frames,
            "stride_frames": stride,
            "rgb_size": int(feature_cfg["rgb_size"]),
            "feature_dim": int(torch.cat(features, dim=0).shape[1]),
            "backbone_checkpoint": str(Path(checkpoint_path).resolve()),
            "backbone_checkpoint_sha256": sha256_file(checkpoint_path),
            "annotation_file": str(info.frame_annotation.resolve()),
            "annotation_sha256": sha256_file(info.frame_annotation),
            "available_frame_count": len(table["rows"]),
            "anchor_count": len(anchors),
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload["metadata"]


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    payload = safe_torch_load(path)
    required = {"features", "state", "start", "end", "original_frame_idx", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Invalid feature cache {path}; missing={sorted(missing)}")
    return payload
