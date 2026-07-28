from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else DEMO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_tier3(row: dict[str, str]) -> str:
    values = []
    for key in ("action", "object", "mark"):
        value = str(row[key]).strip()
        if value.lower() != "none":
            values.append(value)
    return " ".join(values)


def normalized(text: str) -> str:
    return " ".join(str(text).lower().split())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def prepare(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    paths = {key: resolve(value) for key, value in config["paths"].items()}
    demo = config["demo"]

    source_manifest_rows = read_jsonl(paths["source_dataset_manifest"])
    matching_runs = [
        row
        for row in source_manifest_rows
        if row["participant"] == demo["participant"]
        and row["source_run"] == demo["source_run"]
    ]
    if len(matching_runs) != 1:
        raise ValueError(f"Expected one target run, found {len(matching_runs)}")
    source_run = matching_runs[0]

    frame_rows = read_csv(paths["source_frame_annotation"])
    segmentation_rows = read_csv(paths["source_segmentation_annotation"])
    protocol_rows = [
        row
        for row in read_jsonl(paths["existing_protocol_manifest"])
        if row["participant"] == demo["participant"] and row["run"] == demo["source_run"]
    ]
    protocol_rows.sort(key=lambda row: int(row["annotation_row_index"]))

    raw_dir = paths["raw_frames_dir"]
    raw_files = sorted(raw_dir.glob("*.jpg"))
    if not raw_files:
        raise FileNotFoundError(f"No JPEG frames found: {raw_dir}")

    frame_indices = [int(row["frame_idx"]) for row in frame_rows]
    original_indices = [int(row["original_frame_idx"]) for row in frame_rows]
    expected_frame_indices = list(range(1, len(frame_rows) + 1))
    if frame_indices != expected_frame_indices:
        raise ValueError("frame_idx is not contiguous from 1")
    expected_original = list(range(original_indices[0], original_indices[0] + len(frame_rows)))
    if original_indices != expected_original:
        raise ValueError("original_frame_idx is not contiguous")

    frame_names = [row["frame_name"] for row in frame_rows]
    raw_names = [path.name for path in raw_files]
    if frame_names != raw_names:
        raise ValueError("Frame annotation file names do not exactly match sorted raw JPEG names")

    frames_by_segment: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_segment[int(row["segment_no"])].append(row)

    graph_json = read_json(paths["task_graph"])
    graph_nodes = {
        int(node["node_idx"]): node
        for node in graph_json["nodes"]
        if 1 <= int(node["node_idx"]) <= 35
    }

    action_protocol_cursor = 0
    derived_segments: list[dict[str, Any]] = []
    label_matches = 0
    length_matches = 0
    rebase_matches = 0
    original_offset = original_indices[0] - 1

    for source_segment in segmentation_rows:
        segment_no = int(source_segment["No"])
        kept_frames = frames_by_segment.get(segment_no, [])
        if not kept_frames:
            continue
        current_start = int(kept_frames[0]["frame_idx"])
        current_end = int(kept_frames[-1]["frame_idx"])
        current_original_start = int(kept_frames[0]["original_frame_idx"])
        current_original_end = int(kept_frames[-1]["original_frame_idx"])
        expected_start = max(int(source_segment["start_idx"]), original_indices[0]) - original_offset
        expected_end = min(int(source_segment["end_idx"]), original_indices[-1]) - original_offset
        rebase_ok = current_start == expected_start and current_end == expected_end
        rebase_matches += int(rebase_ok)

        is_background = source_segment["action"].strip().lower() == "background"
        protocol: dict[str, Any] | None = None
        if not is_background:
            if action_protocol_cursor >= len(protocol_rows):
                raise ValueError("More action segments than protocol action rows")
            protocol = protocol_rows[action_protocol_cursor]
            action_protocol_cursor += 1
            label_ok = normalized(compose_tier3(source_segment)) == normalized(protocol["tier3"])
            length_ok = len(kept_frames) == int(protocol[f"{demo['camera_id']}_num_rgb_frames"])
            label_matches += int(label_ok)
            length_matches += int(length_ok)
            node = graph_nodes[int(protocol["node_idx"])]
        else:
            label_ok = True
            length_ok = True
            node = None

        derived_segments.append(
            {
                "segment_no": segment_no,
                "is_background": is_background,
                "participant": demo["participant"],
                "run": demo["source_run"],
                "run_sample_name": demo["run_sample_name"],
                "camera_id": demo["camera_id"],
                "current_start_idx": current_start,
                "current_end_idx": current_end,
                "current_frame_count": len(kept_frames),
                "current_start_frame_name": kept_frames[0]["frame_name"],
                "current_end_frame_name": kept_frames[-1]["frame_name"],
                "current_start_timestamp": kept_frames[0]["timestamp"],
                "current_end_timestamp": kept_frames[-1]["timestamp"],
                "current_original_start_idx": current_original_start,
                "current_original_end_idx": current_original_end,
                "source_start_idx": int(source_segment["start_idx"]),
                "source_end_idx": int(source_segment["end_idx"]),
                "source_start_timestamp": source_segment["start"],
                "source_end_timestamp": source_segment["end"],
                "action": source_segment["action"],
                "object": source_segment["object"],
                "mark": source_segment["mark"],
                "annotation_row_index": int(protocol["annotation_row_index"]) if protocol else None,
                "original_action_sample_name": protocol["sample_name"] if protocol else None,
                "node_idx": int(protocol["node_idx"]) if protocol else None,
                "node_id": node["node_id"] if node else None,
                "stage_id": int(protocol["stage_id"]) if protocol else None,
                "tier3_id": int(protocol["tier3_id"]) if protocol else None,
                "tier3_label": protocol["tier3"] if protocol else "background",
                "rebase_check": rebase_ok,
                "label_check": label_ok,
                "frame_count_check": length_ok,
            }
        )

    if action_protocol_cursor != len(protocol_rows):
        raise ValueError(
            f"Mapped {action_protocol_cursor} action segments but protocol has {len(protocol_rows)}"
        )

    duration_seconds = (
        _timestamp_seconds(frame_rows[-1]["timestamp"])
        - _timestamp_seconds(frame_rows[0]["timestamp"])
    )
    action_segments = [row for row in derived_segments if not row["is_background"]]
    background_frames = sum(
        row["current_frame_count"] for row in derived_segments if row["is_background"]
    )
    action_frames = len(frame_rows) - background_frames

    validation_report = {
        "schema_version": "task-graph-realtime-demo-validation-v1",
        "source_files": {
            str(path.relative_to(DEMO_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                paths["source_dataset_manifest"],
                paths["source_frame_annotation"],
                paths["source_segmentation_annotation"],
                paths["source_trim_tracking"],
            )
        },
        "target_run_records": len(matching_runs),
        "raw_jpeg_frames": len(raw_files),
        "frame_annotation_rows": len(frame_rows),
        "frame_names_match_raw_exactly": frame_names == raw_names,
        "frame_idx_contiguous": frame_indices == expected_frame_indices,
        "original_frame_idx_contiguous": original_indices == expected_original,
        "original_frame_idx_span": [original_indices[0], original_indices[-1]],
        "current_frame_idx_span": [frame_indices[0], frame_indices[-1]],
        "segmentation_rows_source": len(segmentation_rows),
        "segments_retained": len(derived_segments),
        "segment_rebase_matches": rebase_matches,
        "action_segments": len(action_segments),
        "background_segments": len(derived_segments) - len(action_segments),
        "protocol_action_rows": len(protocol_rows),
        "action_label_matches": label_matches,
        "action_frame_count_matches": length_matches,
        "background_frames": background_frames,
        "action_frames": action_frames,
        "background_fraction": background_frames / len(frame_rows),
        "duration_seconds": duration_seconds,
        "effective_fps": (len(frame_rows) - 1) / duration_seconds,
        "source_manifest_stale_after_trim": (
            int(source_run["reference_frame_count"]) != len(raw_files)
        ),
        "source_manifest_reference_frame_count": int(source_run["reference_frame_count"]),
        "current_reference_frame_count": len(raw_files),
        "all_checks_pass": (
            frame_names == raw_names
            and rebase_matches == len(derived_segments)
            and label_matches == len(action_segments)
            and length_matches == len(action_segments)
        ),
    }

    derived_manifest = {
        "schema_version": config["schema_version"],
        "created_from": {
            "source_run_manifest_record": source_run,
            "source_files_are_snapshots": True,
            "source_dataset_was_modified": False,
        },
        "demo": demo,
        "raw_frames_dir": str(raw_dir),
        "current_frame_count": len(raw_files),
        "current_start_frame": raw_files[0].name,
        "current_end_frame": raw_files[-1].name,
        "original_frame_offset": original_offset,
        "segments_file": str(paths["derived_segments"].relative_to(DEMO_ROOT)),
        "validation_report": str(paths["validation_report"].relative_to(DEMO_ROOT)),
        "preprocessing": config["preprocessing"],
    }

    write_jsonl(paths["derived_segments"], derived_segments)
    write_json(paths["derived_manifest"], derived_manifest)
    write_json(paths["validation_report"], validation_report)
    return validation_report


def _timestamp_seconds(value: str) -> float:
    from datetime import datetime

    timestamp = datetime.strptime(str(value), "%Y%m%d_%H%M%S_%f")
    return timestamp.timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable-source A/run_7 demo metadata")
    parser.add_argument("--config", type=Path, default=DEMO_ROOT / "config.json")
    args = parser.parse_args()
    report = prepare(args.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
