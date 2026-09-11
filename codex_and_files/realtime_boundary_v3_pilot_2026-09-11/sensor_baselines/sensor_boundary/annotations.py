from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import parse_camera_timestamp, read_csv, read_jsonl


@dataclass(frozen=True)
class RunInfo:
    sample_name: str
    participant: str
    source_run: str
    left_csv: Path
    right_csv: Path
    frame_annotation: Path
    segment_annotation: Path


def load_run_index(dataset_root: str | Path, annotation_root: str | Path) -> dict[str, RunInfo]:
    dataset_root, annotation_root = Path(dataset_root), Path(annotation_root)
    result: dict[str, RunInfo] = {}
    for row in read_jsonl(dataset_root / "manifest.jsonl"):
        name = str(row["sample_name"])
        result[name] = RunInfo(
            sample_name=name,
            participant=str(row["participant"]),
            source_run=str(row["source_run"]),
            left_csv=dataset_root / str(row.get("mindrove_left_csv", f"raw/{name}/mindrove_left.csv")),
            right_csv=dataset_root / str(row.get("mindrove_right_csv", f"raw/{name}/mindrove_right.csv")),
            frame_annotation=annotation_root / f"{name}_frame_annotation.csv",
            segment_annotation=annotation_root / f"{name}_segmentation_annotation.csv",
        )
    return result


def load_annotation(info: RunInfo) -> dict[str, Any]:
    frames = read_csv(info.frame_annotation)
    segments = read_csv(info.segment_annotation)
    if not frames or not segments:
        raise ValueError(f"Empty annotation for {info.sample_name}")
    frame_times = np.asarray([parse_camera_timestamp(row["timestamp"]) for row in frames])
    if np.any(np.diff(frame_times) <= 0):
        raise ValueError(f"Non-increasing frame timestamps: {info.frame_annotation}")
    parsed_segments = []
    for row in segments:
        parsed_segments.append(
            {
                "segment_no": int(row["No"]),
                "action": row["action"].strip(),
                "object": row["object"].strip(),
                "start": parse_camera_timestamp(row["start"]),
                "end": parse_camera_timestamp(row["end"]),
            }
        )
    return {
        "frame_start": float(frame_times[0]),
        "frame_end": float(frame_times[-1]),
        "segments": parsed_segments,
    }


def targets_on_grid(annotation: dict[str, Any], grid_times: np.ndarray, radius_steps: int) -> dict[str, Any]:
    length = len(grid_times)
    state = np.zeros(length, dtype=np.int64)
    exact_start = np.zeros(length, dtype=np.float32)
    exact_end = np.zeros(length, dtype=np.float32)
    segment_id = np.zeros(length, dtype=np.int32)
    action_names: dict[int, str] = {}
    for segment in annotation["segments"]:
        if segment["action"].lower() == "background":
            continue
        start, end = float(segment["start"]), float(segment["end"])
        # A microsecond epsilon prevents decimal floating-point round-off from
        # dropping a decision exactly on an annotation boundary.
        inside = (grid_times >= start - 1e-6) & (grid_times <= end + 1e-6)
        state[inside] = 1
        segment_id[inside] = int(segment["segment_no"])
        action_names[int(segment["segment_no"])] = segment["action"]
        if grid_times[0] <= start <= grid_times[-1]:
            exact_start[int(np.argmin(np.abs(grid_times - start)))] = 1.0
        if grid_times[0] <= end <= grid_times[-1]:
            exact_end[int(np.argmin(np.abs(grid_times - end)))] = 1.0
    return {
        "state": state,
        "start": dilate(exact_start, radius_steps),
        "end": dilate(exact_end, radius_steps),
        "exact_start": exact_start,
        "exact_end": exact_end,
        "segment_id": segment_id,
        "action_names": action_names,
    }


def dilate(target: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return target.astype(np.float32, copy=True)
    result = np.zeros_like(target, dtype=np.float32)
    for index in np.flatnonzero(target > 0):
        result[max(0, index - radius) : min(len(result), index + radius + 1)] = 1.0
    return result
