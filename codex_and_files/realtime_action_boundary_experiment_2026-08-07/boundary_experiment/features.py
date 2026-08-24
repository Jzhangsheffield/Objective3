from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from .annotations import RunInfo, dilate_binary_targets, load_frame_table
from .utils import safe_torch_load, sha256_file


class RGBFrameDataset(Dataset):
    """Decode and normalize each source frame exactly once, in chronological order."""

    def __init__(self, frame_paths: list[Path], size: int, mean: list[float], std: list[float]):
        self.frame_paths = frame_paths
        self.size = int(size)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = read_image(str(self.frame_paths[index])).float().div_(255.0)
        image = resize(image, [self.size, self.size], antialias=True)
        return (image - self.mean) / self.std


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
    num_workers = int(feature_cfg.get("num_workers", 0))
    frame_loader_batch_size = int(feature_cfg.get("frame_loader_batch_size", max(batch_size, 1)))
    pin_memory = bool(feature_cfg.get("pin_memory", True)) and device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": frame_loader_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(feature_cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(feature_cfg.get("persistent_workers", True))
    frame_loader = DataLoader(
        RGBFrameDataset(
            table["frame_paths"], int(feature_cfg["rgb_size"]),
            list(feature_cfg["mean"]), list(feature_cfg["std"]),
        ),
        **loader_kwargs,
    )

    features: list[torch.Tensor] = []
    rolling: deque[torch.Tensor] = deque(maxlen=clip_frames)
    pending_clips: list[torch.Tensor] = []

    def flush_pending() -> None:
        if not pending_clips:
            return
        batch = torch.stack(pending_clips)
        if pin_memory and not batch.is_pinned():
            batch = batch.pin_memory()
        features.append(model.forward_features(batch.to(device, non_blocking=pin_memory)).cpu())
        pending_clips.clear()

    row_index = 0
    for frame_batch in frame_loader:
        for frame in frame_batch:
            if not rolling:
                rolling.extend([frame] * clip_frames)
            else:
                rolling.append(frame)
            if row_index % stride == 0:
                pending_clips.append(torch.stack(list(rolling), dim=1))
                if len(pending_clips) >= batch_size:
                    flush_pending()
            row_index += 1
    flush_pending()
    if row_index != len(table["frame_paths"]):
        raise RuntimeError(f"Frame loader returned {row_index} frames, expected {len(table['frame_paths'])}")

    feature_tensor = torch.cat(features, dim=0)
    if feature_tensor.shape[0] != len(anchors):
        raise RuntimeError(f"Extracted {feature_tensor.shape[0]} anchors, expected {len(anchors)}")
    anchor_tensor = torch.tensor(anchors, dtype=torch.long)
    radius_frames = int(feature_cfg.get("boundary_label_radius_frames", 0))
    radius_anchors = (radius_frames + stride - 1) // stride
    anchor_start = _events_on_anchor_grid(table["start"], anchors)
    anchor_end = _events_on_anchor_grid(table["end"], anchors)
    payload = {
        "sample_name": info.sample_name,
        "participant": info.participant,
        "source_run": info.source_run,
        "features": feature_tensor,
        "anchor_row_index": anchor_tensor,
        "frame_idx": torch.from_numpy(table["frame_idx"][anchors]),
        "original_frame_idx": torch.from_numpy(table["original_frame_idx"][anchors]),
        # 这里的是按照anchors来从原本的索引中进行取值，也就得到每个anchor对应的frame_idx和original_frame_idx
        # 这里是因为原视频的开头和结尾被裁剪去了一部分，所以frame_idx是从1开始的，而original_frame_idx则是真实的frame_idx.
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
            "feature_dim": int(feature_tensor.shape[1]),
            "extract_batch_size": batch_size,
            "frame_loader_batch_size": frame_loader_batch_size,
            "frame_loader_num_workers": num_workers,
            "frame_loader_prefetch_factor": int(feature_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
            "frame_loader_pin_memory": pin_memory,
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
