from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_csv, read_jsonl


@dataclass(frozen=True)
class RunInfo:
    sample_name: str
    participant: str
    source_run: str
    raw_dir: Path
    camera_dir: Path
    frame_annotation: Path


def load_run_index(dataset_root: str | Path, annotation_root: str | Path, camera_id: str) -> dict[str, RunInfo]:
    dataset_root = Path(dataset_root)
    annotation_root = Path(annotation_root)
    result: dict[str, RunInfo] = {}
    for row in read_jsonl(dataset_root / "manifest.jsonl"):
        sample_name = str(row["sample_name"])
        camera_rel = row.get("camera_dirs", {}).get(camera_id, f"raw/{sample_name}/{camera_id}")
        result[sample_name] = RunInfo(
            sample_name=sample_name,
            participant=str(row["participant"]),
            source_run=str(row["source_run"]),
            raw_dir=dataset_root / str(row["raw_dir"]),
            camera_dir=dataset_root / str(camera_rel),
            frame_annotation=annotation_root / f"{sample_name}_frame_annotation.csv",
        )
    return result


def load_frame_table(info: RunInfo) -> dict[str, Any]:
    rows = read_csv(info.frame_annotation)
    if not rows:
        raise ValueError(f"Empty frame annotation: {info.frame_annotation}")
    frame_paths = [info.camera_dir / row["frame_name"] for row in rows]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated frames missing; examples={missing[:3]}")
    is_action = np.asarray([row["action"].strip().lower() != "background" for row in rows], dtype=np.int64)
    original_idx = np.asarray([int(row["original_frame_idx"]) for row in rows], dtype=np.int64)
    frame_idx = np.asarray([int(row["frame_idx"]) for row in rows], dtype=np.int64)
    if not np.all(np.diff(frame_idx) == 1):
        raise ValueError(f"frame_idx is not continuous in {info.frame_annotation}")
    if not np.all(np.diff(original_idx) > 0):
        raise ValueError(f"original_frame_idx is not strictly increasing in {info.frame_annotation}")
    action = [row["action"] for row in rows]
    obj = [row["object"] for row in rows]
    segment_no = np.asarray([int(row["segment_no"]) for row in rows], dtype=np.int64)
    starts = np.zeros(len(rows), dtype=np.float32)
    ends = np.zeros(len(rows), dtype=np.float32)
    for i in range(len(rows)):
        if is_action[i] and (i == 0 or not is_action[i - 1] or segment_no[i] != segment_no[i - 1]):
            starts[i] = 1.0
        if is_action[i] and (i == len(rows) - 1 or not is_action[i + 1] or segment_no[i] != segment_no[i + 1]):
            ends[i] = 1.0
    return {
        "rows": rows,
        "frame_paths": frame_paths,
        "frame_idx": frame_idx,
        "original_frame_idx": original_idx,
        "timestamps": [row["timestamp"] for row in rows],
        "state": is_action,
        "start": starts,
        "end": ends,
        "action": action,
        "object": obj,
        "segment_no": segment_no,
    }


def dilate_binary_targets(target: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return target.astype(np.float32, copy=True)
    result = np.zeros_like(target, dtype=np.float32)
    for index in np.flatnonzero(target > 0):
        result[max(0, index - radius) : min(len(result), index + radius + 1)] = 1.0
    return result
