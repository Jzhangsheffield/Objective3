from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_run_index
from boundary_experiment.engine import infer_run, load_boundary_checkpoint
from boundary_experiment.features import load_feature_cache
from boundary_experiment.metrics import segment_iou
from boundary_experiment.online import run_state_machine
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import read_csv, resolve_device, write_json


TOLERANCES = (3, 5, 10)
CALIBRATED_SETTINGS = {
    "start_threshold": 0.85,
    "end_threshold": 0.85,
    "action_threshold": 0.65,
    "start_debounce": 2,
    "end_debounce": 2,
    "min_action_steps": 5,
    "merge_gap_steps": 0,
}
CALIBRATED_MERGE_GAP = 3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timestamp_seconds(start: str, end: str) -> float:
    fmt = "%Y%m%d_%H%M%S_%f"
    try:
        return max(0.0, (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds())
    except ValueError:
        return float("nan")


def _event_matches(gt: list[int], pred: list[int], tolerance: int) -> dict[int, tuple[int, int]]:
    candidates = sorted(
        (abs(p - g), p_index, g_index)
        for p_index, p in enumerate(pred)
        for g_index, g in enumerate(gt)
        if abs(p - g) <= tolerance
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    result: dict[int, tuple[int, int]] = {}
    for _, p_index, g_index in candidates:
        if p_index in used_pred or g_index in used_gt:
            continue
        used_pred.add(p_index)
        used_gt.add(g_index)
        result[g_index] = (p_index, pred[p_index] - gt[g_index])
    return result


def _segment_matches(gt: list[tuple[int, int]], pred: list[tuple[int, int]], threshold: float) -> dict[int, tuple[int, float]]:
    candidates = sorted(
        (-segment_iou(p, g), p_index, g_index)
        for p_index, p in enumerate(pred)
        for g_index, g in enumerate(gt)
        if segment_iou(p, g) >= threshold
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    result: dict[int, tuple[int, float]] = {}
    for negative_iou, p_index, g_index in candidates:
        if p_index in used_pred or g_index in used_gt:
            continue
        used_pred.add(p_index)
        used_gt.add(g_index)
        result[g_index] = (p_index, -negative_iou)
    return result


def _nearest(value: int, candidates: list[int]) -> tuple[int | None, int | None]:
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda item: (abs(item - value), item))
    return nearest, nearest - value


def _segments_from_exact(cache: dict[str, Any]) -> list[tuple[int, int]]:
    starts = np.flatnonzero(cache["exact_start"].numpy() > 0).tolist()
    ends = np.flatnonzero(cache["exact_end"].numpy() > 0).tolist()
    if len(starts) != len(ends):
        raise ValueError(f"Mismatched exact boundaries in {cache['sample_name']}: starts={len(starts)} ends={len(ends)}")
    segments = list(zip(starts, ends))
    if any(start > end for start, end in segments):
        raise ValueError(f"Invalid exact boundary order in {cache['sample_name']}")
    return segments


def _local_max(values: np.ndarray, index: int, radius: int) -> float:
    return float(np.max(values[max(0, index - radius) : min(len(values), index + radius + 1)]))


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def _mean_abs(rows: list[dict[str, Any]], key: str) -> float:
    values = [abs(float(row[key])) for row in rows if row[key] not in (None, "")]
    return float(np.mean(values)) if values else float("nan")


def _summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...], prefix: str = "") -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for values, group in grouped.items():
        record = {key: value for key, value in zip(group_keys, values)}
        record.update({
            "segments": len(group),
            "start_miss_rate_3": _rate(group, f"{prefix}start_miss_3"),
            "start_miss_rate_5": _rate(group, f"{prefix}start_miss_5"),
            "start_miss_rate_10": _rate(group, f"{prefix}start_miss_10"),
            "end_miss_rate_3": _rate(group, f"{prefix}end_miss_3"),
            "end_miss_rate_5": _rate(group, f"{prefix}end_miss_5"),
            "end_miss_rate_10": _rate(group, f"{prefix}end_miss_10"),
            "any_boundary_miss_rate_5": _rate(group, f"{prefix}any_boundary_miss_5"),
            "segment_miss_rate_iou50": _rate(group, f"{prefix}segment_miss_iou50"),
            "hard_error_rate": _rate(group, f"{prefix}hard_error"),
            "mean_abs_nearest_start_error_frames": _mean_abs(group, f"{prefix}nearest_start_error_frames"),
            "mean_abs_nearest_end_error_frames": _mean_abs(group, f"{prefix}nearest_end_error_frames"),
            "median_duration_seconds": float(np.nanmedian([row["duration_seconds"] for row in group])),
        })
        output.append(record)
    return output


def _merge_short_gap_segments(segments: list[dict[str, Any]], maximum_gap: int) -> list[dict[str, Any]]:
    """Match the validation-calibration diagnostic: decode first, then merge gaps <=3 steps."""
    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: (int(item["start_index"]), int(item["end_index"]))):
        current = dict(segment)
        current["merged_component_count"] = int(current.get("merged_component_count", 1))
        if merged and int(current["start_index"]) - int(merged[-1]["end_index"]) - 1 <= maximum_gap:
            previous = merged[-1]
            previous["end_index"] = max(int(previous["end_index"]), int(current["end_index"]))
            previous["end_score"] = current.get("end_score", previous.get("end_score"))
            previous["emitted_at_index"] = current.get("emitted_at_index", previous.get("emitted_at_index"))
            previous["merged_component_count"] += current["merged_component_count"]
        else:
            merged.append(current)
    return merged


def _error_type(record: dict[str, Any], prefix: str = "") -> str:
    if record[f"{prefix}start_miss_10"] and record[f"{prefix}end_miss_10"]:
        return "start_and_end_missed_10"
    if record[f"{prefix}start_miss_10"]:
        return "start_missed_10"
    if record[f"{prefix}end_miss_10"]:
        return "end_missed_10"
    if record[f"{prefix}segment_miss_iou50"]:
        return "boundary_events_hit_but_segment_iou_below_0.5"
    return "matched"


def _severity(record: dict[str, Any], prefix: str = "") -> int:
    return (
        3 * int(record[f"{prefix}segment_miss_iou50"])
        + 2 * int(record[f"{prefix}start_miss_10"])
        + 2 * int(record[f"{prefix}end_miss_10"])
        + int(record[f"{prefix}start_miss_5"])
        + int(record[f"{prefix}end_miss_5"])
    )


def _plot_before_after(original: list[dict[str, Any]], calibrated: list[dict[str, Any]], path: Path) -> None:
    order = ["train", "validation", "test_normal", "test_fault"]
    labels = ["Train", "Validation", "Test normal", "Test fault"]
    original_lookup = {row["split"]: row for row in original}
    calibrated_lookup = {row["split"]: row for row in calibrated}
    x = np.arange(len(order))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, title in [
        (axes[0], "segment_miss_rate_iou50", "GT segment miss at IoU 0.5"),
        (axes[1], "hard_error_rate", "Hard GT segment error"),
    ]:
        before = [100 * original_lookup[item][key] for item in order]
        after = [100 * calibrated_lookup[item][key] for item in order]
        ax.bar(x - width / 2, before, width, label="Original", color="#E45756")
        ax.bar(x + width / 2, after, width, label="Validation-calibrated", color="#59A14F")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Rate (%)")
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _duration_bin(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    if seconds < 0.5:
        return "<0.5 s"
    if seconds < 1.0:
        return "0.5-1 s"
    if seconds < 2.0:
        return "1-2 s"
    if seconds < 4.0:
        return "2-4 s"
    return ">=4 s"


def _gap_bin(frames: int | None) -> str:
    if frames is None:
        return "run edge"
    if frames == 0:
        return "0"
    if frames <= 7:
        return "1-7"
    if frames <= 15:
        return "8-15"
    if frames <= 30:
        return "16-30"
    return ">30"


def _plot_split(summary: list[dict[str, Any]], path: Path) -> None:
    order = ["train", "validation", "test_normal", "test_fault"]
    labels = ["Train", "Validation", "Test normal", "Test fault"]
    lookup = {row["split"]: row for row in summary}
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ("Start miss @±5", "start_miss_rate_5", "#4C78A8"),
        ("End miss @±5", "end_miss_rate_5", "#F58518"),
        ("Segment miss @IoU 0.5", "segment_miss_rate_iou50", "#E45756"),
    ]
    for offset, (label, key, color) in enumerate(series):
        values = [lookup[item][key] * 100 for item in order]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [f"{label}\n(n={lookup[key]['segments']})" for label, key in zip(labels, order)])
    ax.set_ylabel("Miss rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Boundary and segment misses by data split")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_test_classes(summary: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"], row["segments"]), reverse=True)
    labels = [f"{row['class_label']} (n={row['segments']})" for row in rows]
    y = np.arange(len(rows))
    fig_height = max(9, 0.36 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(y + 0.22, [row["segment_miss_rate_iou50"] * 100 for row in rows], 0.22, label="Segment miss @IoU 0.5", color="#E45756")
    ax.barh(y, [row["start_miss_rate_5"] * 100 for row in rows], 0.22, label="Start miss @±5", color="#4C78A8")
    ax.barh(y - 0.22, [row["end_miss_rate_5"] * 100 for row in rows], 0.22, label="End miss @±5", color="#F58518")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Miss rate (%)")
    ax.set_title("Held-out A test errors by action-object class")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_factors(duration_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]], path: Path) -> None:
    duration_order = ["<0.5 s", "0.5-1 s", "1-2 s", "2-4 s", ">=4 s"]
    gap_order = ["0", "1-7", "8-15", "16-30", ">30", "run edge"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, rows, order, title in [
        (axes[0], duration_rows, duration_order, "Test errors by GT duration"),
        (axes[1], gap_rows, gap_order, "Test errors by nearest background gap"),
    ]:
        group_key = "duration_bin" if rows and "duration_bin" in rows[0] else "gap_bin"
        lookup = {row[group_key]: row for row in rows}
        active_order = [item for item in order if item in lookup]
        x = np.arange(len(active_order))
        ax.plot(x, [lookup[item]["segment_miss_rate_iou50"] * 100 for item in active_order], marker="o", label="Segment miss @IoU 0.5", color="#E45756")
        ax.plot(x, [lookup[item]["any_boundary_miss_rate_5"] * 100 for item in active_order], marker="o", label="Any boundary miss @±5", color="#4C78A8")
        ax.set_xticks(x, [f"{item}\n(n={lookup[item]['segments']})" for item in active_order])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Miss rate (%)")
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_test_runs(summary: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"]), reverse=True)
    labels = [f"{row['sample_name']} (n={row['segments']})" for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, max(8, 0.34 * len(rows) + 2)))
    ax.barh(y + 0.17, [row["segment_miss_rate_iou50"] * 100 for row in rows], 0.34, label="Segment miss @IoU 0.5", color="#E45756")
    ax.barh(y - 0.17, [row["any_boundary_miss_rate_5"] * 100 for row in rows], 0.34, label="Any boundary miss @±5", color="#4C78A8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Miss rate (%)")
    ax.set_title("Held-out A test errors by run")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], percent_keys: set[str] | None = None) -> str:
    percent_keys = percent_keys or set()
    lines = ["| " + " | ".join(label for label, _ in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if key in percent_keys and value is not None:
                values.append(f"{100 * float(value):.2f}%")
            elif isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.2f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GT action segments that are difficult for the boundary detector")
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--camera-id", default="001484412812")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    condition_root = Path(args.condition_root)
    cache_root = Path(args.cache_root)
    protocol_root = Path(args.protocol_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((condition_root / "resolved_config.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model, checkpoint = load_boundary_checkpoint(condition_root / "best.pth", device)
    run_index = load_run_index(args.dataset_root, args.annotation_root, args.camera_id)

    split_runs = {
        "train": list(config["train_runs"]),
        "validation": list(config["validation_runs"]),
        "test_normal": load_protocol_runs(protocol_root / "test_normal.jsonl"),
        "test_fault": load_protocol_runs(protocol_root / "test_fault.jsonl"),
    }
    saved_test_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_prediction_file = condition_root / "evaluation" / "test_all" / "predicted_segments.jsonl"
    with original_prediction_file.open(encoding="utf-8") as handle:
        for line in handle:
            segment = json.loads(line)
            saved_test_predictions[str(segment["sample_name"])].append(segment)
    rows: list[dict[str, Any]] = []
    prediction_counts: dict[str, int] = {}
    redecoded_prediction_counts: dict[str, int] = {}
    calibrated_prediction_counts: dict[str, int] = {}
    for split, runs in split_runs.items():
        for run_position, run_name in enumerate(runs, 1):
            info = run_index[run_name]
            cache_path = cache_root / f"{run_name}.pt"
            cache = load_feature_cache(cache_path)
            probabilities = infer_run(model, cache, device)
            redecoded = run_state_machine(**probabilities, settings=config["online"])
            calibrated_raw = run_state_machine(**probabilities, settings=CALIBRATED_SETTINGS)
            calibrated = _merge_short_gap_segments(calibrated_raw, CALIBRATED_MERGE_GAP)
            redecoded_prediction_counts[run_name] = len(redecoded)
            calibrated_prediction_counts[run_name] = len(calibrated)
            predicted = (
                saved_test_predictions[run_name]
                if split.startswith("test_")
                else redecoded
            )
            prediction_counts[run_name] = len(predicted)
            original = cache["original_frame_idx"].numpy().astype(int)
            gt_anchor = _segments_from_exact(cache)
            gt_frame = [(int(original[start]), int(original[end])) for start, end in gt_anchor]
            pred_anchor = [(int(segment["start_index"]), int(segment["end_index"])) for segment in predicted]
            pred_frame = [(int(original[start]), int(original[end])) for start, end in pred_anchor]
            calibrated_pred_anchor = [(int(segment["start_index"]), int(segment["end_index"])) for segment in calibrated]
            calibrated_pred_frame = [(int(original[start]), int(original[end])) for start, end in calibrated_pred_anchor]
            gt_starts = [item[0] for item in gt_frame]
            gt_ends = [item[1] for item in gt_frame]
            pred_starts = [item[0] for item in pred_frame]
            pred_ends = [item[1] for item in pred_frame]
            calibrated_pred_starts = [item[0] for item in calibrated_pred_frame]
            calibrated_pred_ends = [item[1] for item in calibrated_pred_frame]
            start_matches = {tol: _event_matches(gt_starts, pred_starts, tol) for tol in TOLERANCES}
            end_matches = {tol: _event_matches(gt_ends, pred_ends, tol) for tol in TOLERANCES}
            segment_matches = _segment_matches(gt_frame, pred_frame, 0.5)
            calibrated_start_matches = {tol: _event_matches(gt_starts, calibrated_pred_starts, tol) for tol in TOLERANCES}
            calibrated_end_matches = {tol: _event_matches(gt_ends, calibrated_pred_ends, tol) for tol in TOLERANCES}
            calibrated_segment_matches = _segment_matches(gt_frame, calibrated_pred_frame, 0.5)
            annotation_rows = read_csv(info.frame_annotation)

            for gt_index, ((start, end), (start_frame, end_frame)) in enumerate(zip(gt_anchor, gt_frame)):
                start_row = int(cache["anchor_row_index"][start])
                end_row = int(cache["anchor_row_index"][end])
                start_annotation = annotation_rows[start_row]
                end_annotation = annotation_rows[end_row]
                start_timestamp = str(cache["timestamps"][start])
                end_timestamp = str(cache["timestamps"][end])
                previous_end = gt_anchor[gt_index - 1][1] if gt_index > 0 else None
                next_start = gt_anchor[gt_index + 1][0] if gt_index + 1 < len(gt_anchor) else None
                gap_before = start - previous_end - 1 if previous_end is not None else None
                gap_after = next_start - end - 1 if next_start is not None else None
                finite_gaps = [value for value in (gap_before, gap_after) if value is not None]
                min_gap = min(finite_gaps) if finite_gaps else None
                nearest_start, nearest_start_error = _nearest(start_frame, pred_starts)
                nearest_end, nearest_end_error = _nearest(end_frame, pred_ends)
                calibrated_nearest_start, calibrated_nearest_start_error = _nearest(start_frame, calibrated_pred_starts)
                calibrated_nearest_end, calibrated_nearest_end_error = _nearest(end_frame, calibrated_pred_ends)
                matched = segment_matches.get(gt_index)
                matched_pred_index = matched[0] if matched else None
                matched_pred = predicted[matched_pred_index] if matched_pred_index is not None else None
                matched_frame = pred_frame[matched_pred_index] if matched_pred_index is not None else None
                calibrated_matched = calibrated_segment_matches.get(gt_index)
                calibrated_matched_pred_index = calibrated_matched[0] if calibrated_matched else None
                calibrated_matched_pred = calibrated[calibrated_matched_pred_index] if calibrated_matched_pred_index is not None else None
                calibrated_matched_frame = calibrated_pred_frame[calibrated_matched_pred_index] if calibrated_matched_pred_index is not None else None
                record: dict[str, Any] = {
                    "split": split,
                    "phase": "training_data" if split in ("train", "validation") else "heldout_test",
                    "sample_name": run_name,
                    "participant": info.participant,
                    "source_run": info.source_run,
                    "gt_segment_ordinal": gt_index + 1,
                    "segment_no": int(cache["segment_no"][start]),
                    "action": cache["action"][start],
                    "object": cache["object"][start],
                    "class_label": f"{cache['action'][start]} · {cache['object'][start]}",
                    "gt_start_anchor_index": start,
                    "gt_end_anchor_index": end,
                    "gt_start_original_frame_idx": start_frame,
                    "gt_end_original_frame_idx": end_frame,
                    "gt_start_timestamp": start_timestamp,
                    "gt_end_timestamp": end_timestamp,
                    "gt_start_frame_name": start_annotation["frame_name"],
                    "gt_end_frame_name": end_annotation["frame_name"],
                    "gt_start_frame_path": str((info.camera_dir / start_annotation["frame_name"]).resolve()),
                    "gt_end_frame_path": str((info.camera_dir / end_annotation["frame_name"]).resolve()),
                    "duration_frames": end_frame - start_frame + 1,
                    "duration_seconds": _timestamp_seconds(start_timestamp, end_timestamp),
                    "background_gap_before_steps": gap_before,
                    "background_gap_after_steps": gap_after,
                    "nearest_background_gap_steps": min_gap,
                    "start_probability_at_gt": float(probabilities["start_probability"][start]),
                    "end_probability_at_gt": float(probabilities["end_probability"][end]),
                    "action_probability_at_gt_start": float(probabilities["state_probability"][start]),
                    "action_probability_at_gt_end": float(probabilities["state_probability"][end]),
                    "start_probability_local_max_radius2": _local_max(probabilities["start_probability"], start, 2),
                    "end_probability_local_max_radius2": _local_max(probabilities["end_probability"], end, 2),
                    "start_probability_local_max_radius5": _local_max(probabilities["start_probability"], start, 5),
                    "end_probability_local_max_radius5": _local_max(probabilities["end_probability"], end, 5),
                    "nearest_predicted_start_original_frame_idx": nearest_start,
                    "nearest_start_error_frames": nearest_start_error,
                    "nearest_predicted_end_original_frame_idx": nearest_end,
                    "nearest_end_error_frames": nearest_end_error,
                    "segment_matched_iou50": matched is not None,
                    "segment_iou": matched[1] if matched else None,
                    "matched_predicted_segment_ordinal": matched_pred_index + 1 if matched_pred_index is not None else None,
                    "matched_predicted_start_original_frame_idx": matched_frame[0] if matched_frame else None,
                    "matched_predicted_end_original_frame_idx": matched_frame[1] if matched_frame else None,
                    "matched_predicted_start_score": float(matched_pred["start_score"]) if matched_pred else None,
                    "matched_predicted_end_score": float(matched_pred["end_score"]) if matched_pred else None,
                    "calibrated_nearest_predicted_start_original_frame_idx": calibrated_nearest_start,
                    "calibrated_nearest_start_error_frames": calibrated_nearest_start_error,
                    "calibrated_nearest_predicted_end_original_frame_idx": calibrated_nearest_end,
                    "calibrated_nearest_end_error_frames": calibrated_nearest_end_error,
                    "calibrated_segment_matched_iou50": calibrated_matched is not None,
                    "calibrated_segment_iou": calibrated_matched[1] if calibrated_matched else None,
                    "calibrated_matched_predicted_segment_ordinal": calibrated_matched_pred_index + 1 if calibrated_matched_pred_index is not None else None,
                    "calibrated_matched_predicted_start_original_frame_idx": calibrated_matched_frame[0] if calibrated_matched_frame else None,
                    "calibrated_matched_predicted_end_original_frame_idx": calibrated_matched_frame[1] if calibrated_matched_frame else None,
                    "calibrated_matched_predicted_start_score": float(calibrated_matched_pred["start_score"]) if calibrated_matched_pred and calibrated_matched_pred.get("start_score") is not None else None,
                    "calibrated_matched_predicted_end_score": float(calibrated_matched_pred["end_score"]) if calibrated_matched_pred and calibrated_matched_pred.get("end_score") is not None else None,
                    "calibrated_matched_merged_components": int(calibrated_matched_pred.get("merged_component_count", 1)) if calibrated_matched_pred else None,
                    "annotation_file": str(info.frame_annotation.resolve()),
                    "feature_cache": str(cache_path.resolve()),
                }
                for tolerance in TOLERANCES:
                    start_match = start_matches[tolerance].get(gt_index)
                    end_match = end_matches[tolerance].get(gt_index)
                    record[f"start_hit_{tolerance}"] = start_match is not None
                    record[f"start_miss_{tolerance}"] = start_match is None
                    record[f"start_matched_error_frames_{tolerance}"] = start_match[1] if start_match else None
                    record[f"end_hit_{tolerance}"] = end_match is not None
                    record[f"end_miss_{tolerance}"] = end_match is None
                    record[f"end_matched_error_frames_{tolerance}"] = end_match[1] if end_match else None
                    calibrated_start_match = calibrated_start_matches[tolerance].get(gt_index)
                    calibrated_end_match = calibrated_end_matches[tolerance].get(gt_index)
                    record[f"calibrated_start_hit_{tolerance}"] = calibrated_start_match is not None
                    record[f"calibrated_start_miss_{tolerance}"] = calibrated_start_match is None
                    record[f"calibrated_start_matched_error_frames_{tolerance}"] = calibrated_start_match[1] if calibrated_start_match else None
                    record[f"calibrated_end_hit_{tolerance}"] = calibrated_end_match is not None
                    record[f"calibrated_end_miss_{tolerance}"] = calibrated_end_match is None
                    record[f"calibrated_end_matched_error_frames_{tolerance}"] = calibrated_end_match[1] if calibrated_end_match else None
                record["any_boundary_miss_5"] = bool(record["start_miss_5"] or record["end_miss_5"])
                record["segment_miss_iou50"] = not bool(record["segment_matched_iou50"])
                record["hard_error"] = bool(record["start_miss_10"] or record["end_miss_10"] or record["segment_miss_iou50"])
                record["error_type"] = _error_type(record)
                record["severity_score"] = _severity(record)
                record["calibrated_any_boundary_miss_5"] = bool(record["calibrated_start_miss_5"] or record["calibrated_end_miss_5"])
                record["calibrated_segment_miss_iou50"] = not bool(record["calibrated_segment_matched_iou50"])
                record["calibrated_hard_error"] = bool(record["calibrated_start_miss_10"] or record["calibrated_end_miss_10"] or record["calibrated_segment_miss_iou50"])
                record["calibrated_error_type"] = _error_type(record, "calibrated_")
                record["calibrated_severity_score"] = _severity(record, "calibrated_")
                record["calibrated_segment_fixed"] = bool(record["segment_miss_iou50"] and not record["calibrated_segment_miss_iou50"])
                record["calibrated_segment_became_miss"] = bool(not record["segment_miss_iou50"] and record["calibrated_segment_miss_iou50"])
                record["calibrated_hard_resolved"] = bool(record["hard_error"] and not record["calibrated_hard_error"])
                record["calibrated_hard_new"] = bool(not record["hard_error"] and record["calibrated_hard_error"])
                if record["calibrated_hard_resolved"]:
                    record["calibration_outcome"] = "resolved_hard_error"
                elif record["calibrated_hard_new"]:
                    record["calibration_outcome"] = "new_hard_error"
                elif record["hard_error"] and record["calibrated_hard_error"]:
                    record["calibration_outcome"] = "persistent_hard_error"
                else:
                    record["calibration_outcome"] = "non_hard_before_and_after"
                record["duration_bin"] = _duration_bin(record["duration_seconds"])
                record["gap_bin"] = _gap_bin(min_gap)
                rows.append(record)
            print(f"[{split} {run_position}/{len(runs)}] {run_name}: gt={len(gt_anchor)} original={len(predicted)} calibrated={len(calibrated)}")

    split_summary = sorted(_summarize(rows, ("split",)), key=lambda row: ["train", "validation", "test_normal", "test_fault"].index(row["split"]))
    split_order = {name: index for index, name in enumerate(("train", "validation", "test_normal", "test_fault"))}
    class_summary_all = sorted(
        _summarize(rows, ("split", "class_label", "action", "object")),
        key=lambda row: (split_order[row["split"]], -row["hard_error_rate"], -row["segments"]),
    )
    test_rows = [row for row in rows if row["phase"] == "heldout_test"]
    test_class_summary = sorted(_summarize(test_rows, ("class_label", "action", "object")), key=lambda row: (-row["hard_error_rate"], -row["segments"]))
    test_run_summary = sorted(_summarize(test_rows, ("sample_name", "split")), key=lambda row: (-row["hard_error_rate"], -row["segments"]))
    duration_summary = sorted(_summarize(test_rows, ("duration_bin",)), key=lambda row: ["<0.5 s", "0.5-1 s", "1-2 s", "2-4 s", ">=4 s", "unknown"].index(row["duration_bin"]))
    gap_summary = sorted(_summarize(test_rows, ("gap_bin",)), key=lambda row: ["0", "1-7", "8-15", "16-30", ">30", "run edge"].index(row["gap_bin"]))
    hard_rows = sorted([row for row in rows if row["hard_error"]], key=lambda row: (row["phase"] != "heldout_test", -row["severity_score"], row["sample_name"], row["gt_start_original_frame_idx"]))
    test_hard_rows = [row for row in hard_rows if row["phase"] == "heldout_test"]

    calibrated_split_summary = sorted(_summarize(rows, ("split",), "calibrated_"), key=lambda row: split_order[row["split"]])
    calibrated_class_summary_all = sorted(
        _summarize(rows, ("split", "class_label", "action", "object"), "calibrated_"),
        key=lambda row: (split_order[row["split"]], -row["hard_error_rate"], -row["segments"]),
    )
    calibrated_test_class_summary = sorted(
        _summarize(test_rows, ("class_label", "action", "object"), "calibrated_"),
        key=lambda row: (-row["hard_error_rate"], -row["segments"]),
    )
    calibrated_test_run_summary = sorted(
        _summarize(test_rows, ("sample_name", "split"), "calibrated_"),
        key=lambda row: (-row["hard_error_rate"], -row["segments"]),
    )
    calibrated_duration_summary = sorted(
        _summarize(test_rows, ("duration_bin",), "calibrated_"),
        key=lambda row: ["<0.5 s", "0.5-1 s", "1-2 s", "2-4 s", ">=4 s", "unknown"].index(row["duration_bin"]),
    )
    calibrated_gap_summary = sorted(
        _summarize(test_rows, ("gap_bin",), "calibrated_"),
        key=lambda row: ["0", "1-7", "8-15", "16-30", ">30", "run edge"].index(row["gap_bin"]),
    )
    calibrated_hard_rows = sorted(
        [row for row in rows if row["calibrated_hard_error"]],
        key=lambda row: (row["phase"] != "heldout_test", -row["calibrated_severity_score"], row["sample_name"], row["gt_start_original_frame_idx"]),
    )
    calibrated_test_hard_rows = [row for row in calibrated_hard_rows if row["phase"] == "heldout_test"]

    original_split_lookup = {row["split"]: row for row in split_summary}
    calibrated_split_lookup = {row["split"]: row for row in calibrated_split_summary}
    decoder_comparison = []
    for split in ("train", "validation", "test_normal", "test_fault"):
        split_gt_rows = [row for row in rows if row["split"] == split]
        run_names = split_runs[split]
        original_pred_count = sum(prediction_counts[name] for name in run_names)
        calibrated_pred_count = sum(calibrated_prediction_counts[name] for name in run_names)
        original_matches = sum(not row["segment_miss_iou50"] for row in split_gt_rows)
        calibrated_matches = sum(not row["calibrated_segment_miss_iou50"] for row in split_gt_rows)
        original_precision = original_matches / original_pred_count if original_pred_count else 0.0
        calibrated_precision = calibrated_matches / calibrated_pred_count if calibrated_pred_count else 0.0
        original_recall = original_matches / len(split_gt_rows) if split_gt_rows else 0.0
        calibrated_recall = calibrated_matches / len(split_gt_rows) if split_gt_rows else 0.0
        original_f1 = 2 * original_precision * original_recall / (original_precision + original_recall) if original_precision + original_recall else 0.0
        calibrated_f1 = 2 * calibrated_precision * calibrated_recall / (calibrated_precision + calibrated_recall) if calibrated_precision + calibrated_recall else 0.0
        before = original_split_lookup[split]
        after = calibrated_split_lookup[split]
        decoder_comparison.append({
            "split": split,
            "gt_segments": len(split_gt_rows),
            "original_predicted_segments": original_pred_count,
            "calibrated_predicted_segments": calibrated_pred_count,
            "predicted_segment_reduction_rate": 1.0 - calibrated_pred_count / original_pred_count if original_pred_count else 0.0,
            "original_segment_matches_iou50": original_matches,
            "calibrated_segment_matches_iou50": calibrated_matches,
            "original_segment_precision_iou50": original_precision,
            "calibrated_segment_precision_iou50": calibrated_precision,
            "original_segment_recall_iou50": original_recall,
            "calibrated_segment_recall_iou50": calibrated_recall,
            "original_segment_f1_iou50": original_f1,
            "calibrated_segment_f1_iou50": calibrated_f1,
            "segment_f1_delta_pp": 100 * (calibrated_f1 - original_f1),
            "original_start_miss_rate_5": before["start_miss_rate_5"],
            "calibrated_start_miss_rate_5": after["start_miss_rate_5"],
            "start_miss_delta_pp": 100 * (after["start_miss_rate_5"] - before["start_miss_rate_5"]),
            "original_end_miss_rate_5": before["end_miss_rate_5"],
            "calibrated_end_miss_rate_5": after["end_miss_rate_5"],
            "end_miss_delta_pp": 100 * (after["end_miss_rate_5"] - before["end_miss_rate_5"]),
            "original_segment_miss_rate_iou50": before["segment_miss_rate_iou50"],
            "calibrated_segment_miss_rate_iou50": after["segment_miss_rate_iou50"],
            "segment_miss_delta_pp": 100 * (after["segment_miss_rate_iou50"] - before["segment_miss_rate_iou50"]),
            "original_hard_error_rate": before["hard_error_rate"],
            "calibrated_hard_error_rate": after["hard_error_rate"],
            "hard_error_delta_pp": 100 * (after["hard_error_rate"] - before["hard_error_rate"]),
            "hard_errors_resolved": sum(row["calibrated_hard_resolved"] for row in split_gt_rows),
            "new_hard_errors": sum(row["calibrated_hard_new"] for row in split_gt_rows),
        })

    expected_test_predictions = sum(prediction_counts[run] for run in split_runs["test_normal"] + split_runs["test_fault"])
    redecoded_test_predictions = sum(redecoded_prediction_counts[run] for run in split_runs["test_normal"] + split_runs["test_fault"])
    original_test_predictions = sum(1 for _ in original_prediction_file.open(encoding="utf-8"))
    if expected_test_predictions != original_test_predictions:
        raise RuntimeError(f"Test prediction count changed: inferred={expected_test_predictions} original={original_test_predictions}")

    artifacts = {
        "all_segments_csv": output_root / "boundary_segment_error_audit_before_after.csv",
        "hard_segments_csv": output_root / "boundary_segment_error_audit_original_hard_refresh.csv",
        "split_summary_csv": output_root / "boundary_error_by_split.csv",
        "class_summary_csv": output_root / "boundary_error_by_class.csv",
        "run_summary_csv": output_root / "boundary_error_by_test_run.csv",
        "factors_summary_csv": output_root / "boundary_error_by_duration_and_gap.csv",
        "calibrated_hard_segments_csv": output_root / "boundary_segment_error_audit_calibrated_hard.csv",
        "calibrated_split_summary_csv": output_root / "boundary_error_calibrated_by_split.csv",
        "calibrated_class_summary_csv": output_root / "boundary_error_calibrated_by_class.csv",
        "calibrated_run_summary_csv": output_root / "boundary_error_calibrated_by_test_run.csv",
        "calibrated_factors_summary_csv": output_root / "boundary_error_calibrated_by_duration_and_gap.csv",
        "decoder_comparison_csv": output_root / "boundary_decoder_before_after.csv",
        "split_plot": output_root / "boundary_error_by_split.png",
        "class_plot": output_root / "boundary_error_by_test_class.png",
        "factor_plot": output_root / "boundary_error_by_duration_gap.png",
        "run_plot": output_root / "boundary_error_by_test_run.png",
        "calibrated_split_plot": output_root / "boundary_error_calibrated_by_split.png",
        "calibrated_class_plot": output_root / "boundary_error_calibrated_by_test_class.png",
        "calibrated_factor_plot": output_root / "boundary_error_calibrated_by_duration_gap.png",
        "calibrated_run_plot": output_root / "boundary_error_calibrated_by_test_run.png",
        "before_after_plot": output_root / "boundary_error_original_vs_calibrated.png",
        "report": output_root.parent / "BOUNDARY_SEGMENT_ERROR_ANALYSIS_2026-09-09.md",
    }
    _write_csv(artifacts["all_segments_csv"], rows)
    _write_csv(artifacts["hard_segments_csv"], hard_rows)
    _write_csv(artifacts["split_summary_csv"], split_summary)
    _write_csv(artifacts["class_summary_csv"], class_summary_all)
    _write_csv(artifacts["run_summary_csv"], test_run_summary)
    combined_factors = [{"factor": "duration", **row} for row in duration_summary] + [{"factor": "gap", **row} for row in gap_summary]
    _write_csv(artifacts["factors_summary_csv"], combined_factors)
    _write_csv(artifacts["calibrated_hard_segments_csv"], calibrated_hard_rows)
    _write_csv(artifacts["calibrated_split_summary_csv"], calibrated_split_summary)
    _write_csv(artifacts["calibrated_class_summary_csv"], calibrated_class_summary_all)
    _write_csv(artifacts["calibrated_run_summary_csv"], calibrated_test_run_summary)
    calibrated_combined_factors = [{"factor": "duration", **row} for row in calibrated_duration_summary] + [{"factor": "gap", **row} for row in calibrated_gap_summary]
    _write_csv(artifacts["calibrated_factors_summary_csv"], calibrated_combined_factors)
    _write_csv(artifacts["decoder_comparison_csv"], decoder_comparison)
    _plot_split(split_summary, artifacts["split_plot"])
    _plot_test_classes(test_class_summary, artifacts["class_plot"])
    _plot_factors(duration_summary, gap_summary, artifacts["factor_plot"])
    _plot_test_runs(test_run_summary, artifacts["run_plot"])
    _plot_split(calibrated_split_summary, artifacts["calibrated_split_plot"])
    _plot_test_classes(calibrated_test_class_summary, artifacts["calibrated_class_plot"])
    _plot_factors(calibrated_duration_summary, calibrated_gap_summary, artifacts["calibrated_factor_plot"])
    _plot_test_runs(calibrated_test_run_summary, artifacts["calibrated_run_plot"])
    _plot_before_after(split_summary, calibrated_split_summary, artifacts["before_after_plot"])

    summary_payload = {
        "scope": {
            "checkpoint": str((condition_root / "best.pth").resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "decoder": config["online"],
            "calibrated_decoder": CALIBRATED_SETTINGS,
            "calibrated_post_merge_gap_steps": CALIBRATED_MERGE_GAP,
            "calibration_selection_policy": "decoder parameters selected on the 12 training-side validation runs only",
            "gt_segments": len(rows),
            "test_gt_segments": len(test_rows),
            "original_test_predictions": original_test_predictions,
            "fresh_redecoded_test_predictions": redecoded_test_predictions,
        },
        "split_summary": split_summary,
        "calibrated_split_summary": calibrated_split_summary,
        "decoder_comparison": decoder_comparison,
        "test_class_summary": test_class_summary,
        "test_run_summary": test_run_summary,
        "duration_summary": duration_summary,
        "gap_summary": gap_summary,
        "hard_segments": len(hard_rows),
        "test_hard_segments": len(test_hard_rows),
        "calibrated_hard_segments": len(calibrated_hard_rows),
        "calibrated_test_hard_segments": len(calibrated_test_hard_rows),
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
    }
    write_json(output_root / "boundary_segment_error_summary.json", summary_payload)
    write_json(output_root / "boundary_segment_error_workbook_data.json", {
        "scope": summary_payload["scope"],
        "split_summary": split_summary,
        "calibrated_split_summary": calibrated_split_summary,
        "decoder_comparison": decoder_comparison,
        "class_summary": class_summary_all,
        "calibrated_class_summary": calibrated_class_summary_all,
        "test_class_summary": test_class_summary,
        "calibrated_test_class_summary": calibrated_test_class_summary,
        "test_run_summary": test_run_summary,
        "calibrated_test_run_summary": calibrated_test_run_summary,
        "duration_summary": duration_summary,
        "calibrated_duration_summary": calibrated_duration_summary,
        "gap_summary": gap_summary,
        "calibrated_gap_summary": calibrated_gap_summary,
        "all_segments": rows,
        "hard_segments": hard_rows,
        "calibrated_hard_segments": calibrated_hard_rows,
    })

    percent = {
        "start_miss_rate_5", "end_miss_rate_5", "any_boundary_miss_rate_5",
        "segment_miss_rate_iou50", "hard_error_rate",
    }
    top_classes = sorted(test_class_summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"], row["segments"]), reverse=True)
    calibrated_top_classes = sorted(calibrated_test_class_summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"], row["segments"]), reverse=True)
    top_runs = sorted(test_run_summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"]), reverse=True)
    calibrated_top_runs = sorted(calibrated_test_run_summary, key=lambda row: (row["segment_miss_rate_iou50"], row["any_boundary_miss_rate_5"]), reverse=True)
    top_segments = test_hard_rows[:40]
    calibrated_top_segments = calibrated_test_hard_rows[:40]
    split_lookup = {row["split"]: row for row in split_summary}
    calibrated_split_lookup = {row["split"]: row for row in calibrated_split_summary}
    comparison_lookup = {row["split"]: row for row in decoder_comparison}
    test_resolved = sum(row["calibrated_hard_resolved"] for row in test_rows)
    test_new = sum(row["calibrated_hard_new"] for row in test_rows)
    test_persistent = sum(row["hard_error"] and row["calibrated_hard_error"] for row in test_rows)
    comparison_percent = {
        "predicted_segment_reduction_rate", "original_segment_precision_iou50", "calibrated_segment_precision_iou50",
        "original_segment_recall_iou50", "calibrated_segment_recall_iou50", "original_segment_f1_iou50",
        "calibrated_segment_f1_iou50", "original_start_miss_rate_5", "calibrated_start_miss_rate_5",
        "original_end_miss_rate_5", "calibrated_end_miss_rate_5", "original_segment_miss_rate_iou50",
        "calibrated_segment_miss_rate_iou50", "original_hard_error_rate", "calibrated_hard_error_rate",
    }
    report = f"""# Boundary segment error audit

日期：2026-09-09  
范围：A-as-test、seed 1、all-runs、epoch-{int(checkpoint['epoch'])} `best.pth`、stride 1；同时审计原始decoder与validation-only校准decoder

## 1. 结论摘要

本报告对每个GT动作片段逐条检查start、end和完整片段匹配。总计{len(rows)}个GT片段，其中train {sum(1 for row in rows if row['split']=='train')}个、validation {sum(1 for row in rows if row['split']=='validation')}个、held-out A测试{len(test_rows)}个。原始与校准结果使用完全相同的GT、原始帧号、一对一事件匹配和一对一IoU匹配规则。

校准参数为`start/end/action threshold=0.85/0.85/0.65`、`start/end debounce=2/2`、`min_action_steps=5`，再把间隔不超过{CALIBRATED_MERGE_GAP}个feature steps的片段合并。参数仅由12个training-side validation runs选择，不使用held-out A标签调参。因此，校准后的held-out A分析比原始decoder更适合作为第一版系统的主要片段诊断；原始结果保留为对照。

主要结果：

- Held-out A normal中，预测片段由{comparison_lookup['test_normal']['original_predicted_segments']}降至{comparison_lookup['test_normal']['calibrated_predicted_segments']}；fault由{comparison_lookup['test_fault']['original_predicted_segments']}降至{comparison_lookup['test_fault']['calibrated_predicted_segments']}。校准显著抑制原始decoder的碎片化。
- Test-normal的Segment F1@IoU0.5由{100 * comparison_lookup['test_normal']['original_segment_f1_iou50']:.2f}%升至{100 * comparison_lookup['test_normal']['calibrated_segment_f1_iou50']:.2f}%；test-fault由{100 * comparison_lookup['test_fault']['original_segment_f1_iou50']:.2f}%升至{100 * comparison_lookup['test_fault']['calibrated_segment_f1_iou50']:.2f}%。
- 按GT片段审计，test-normal的Segment miss由{100 * split_lookup['test_normal']['segment_miss_rate_iou50']:.2f}%变为{100 * calibrated_split_lookup['test_normal']['segment_miss_rate_iou50']:.2f}%，test-fault由{100 * split_lookup['test_fault']['segment_miss_rate_iou50']:.2f}%变为{100 * calibrated_split_lookup['test_fault']['segment_miss_rate_iou50']:.2f}%。
- Held-out A的{test_resolved}个原始hard errors被校准修复，{test_persistent}个仍然困难，同时新增{test_new}个hard errors。校准并非对每个GT都单调改善，所以人工检查应优先看“persistent”和“new”两组。

## 2. 错误定义

- `Start/End miss @±N`：在原始帧号上进行一对一事件匹配，GT边界在±N帧内没有未占用预测边界。
- `Segment miss @IoU 0.5`：GT片段没有与任何预测片段完成一对一IoU≥0.5匹配。
- `Hard error`：start或end在±10帧内未命中，或者完整片段没有IoU≥0.5匹配。
- `Nearest error`：只用于诊断最近预测边界偏早或偏晚，不等同于一对一命中结果。
- `Resolved hard error`：原始decoder为hard error、校准后不再是hard error。
- `New hard error`：原始decoder不是hard error、校准后变成hard error。

## 3. 原始与校准decoder直接比较

{_markdown_table(decoder_comparison, [('Split','split'),('GT','gt_segments'),('Original pred','original_predicted_segments'),('Calibrated pred','calibrated_predicted_segments'),('Pred reduction','predicted_segment_reduction_rate'),('Original Seg F1','original_segment_f1_iou50'),('Calibrated Seg F1','calibrated_segment_f1_iou50'),('Original Seg miss','original_segment_miss_rate_iou50'),('Calibrated Seg miss','calibrated_segment_miss_rate_iou50'),('Resolved hard','hard_errors_resolved'),('New hard','new_hard_errors')], comparison_percent)}

![Original versus calibrated](boundary_segment_error_audit/boundary_error_original_vs_calibrated.png)

表中的Segment Precision/Recall/F1使用预测片段与GT片段的一对一IoU≥0.5匹配；`Segment miss`则以GT为分母。二者回答的问题不同：F1同时惩罚多余预测片段，GT miss用于定位具体哪个真实动作没有被完整检测。

## 4. 校准后Train、Validation和Test比较（主要分析）

{_markdown_table(calibrated_split_summary, [('Split','split'),('GT segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50'),('Hard error','hard_error_rate')], percent)}

![Calibrated error by split](boundary_segment_error_audit/boundary_error_calibrated_by_split.png)

这里的train仍是最终epoch-{int(checkpoint['epoch'])} checkpoint的in-sample回放，不是训练过程中逐epoch错误轨迹。Validation和Test使用同一冻结checkpoint回放；只有decoder参数来自validation选择。

## 5. 校准后Held-out A中容易出错的动作类别

{_markdown_table(calibrated_top_classes, [('Action · object','class_label'),('Segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50'),('Hard error','hard_error_rate'),('Median duration s','median_duration_seconds')], percent)}

![Calibrated error by test class](boundary_segment_error_audit/boundary_error_calibrated_by_test_class.png)

类别样本数`n`必须与错误率一起看；少量类别的高错误率不能直接外推为跨seed、跨participant稳定结论。

## 6. 校准后容易出错的Test runs

{_markdown_table(calibrated_top_runs, [('Run','sample_name'),('Type','split'),('GT segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50'),('Hard error','hard_error_rate')], percent)}

![Calibrated error by test run](boundary_segment_error_audit/boundary_error_calibrated_by_test_run.png)

## 7. 校准后片段长度和相邻background gap

{_markdown_table(calibrated_duration_summary, [('Duration','duration_bin'),('Segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50')], percent)}

{_markdown_table(calibrated_gap_summary, [('Nearest gap steps','gap_bin'),('Segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50')], percent)}

![Calibrated error by duration and gap](boundary_segment_error_audit/boundary_error_calibrated_by_duration_gap.png)

`nearest background gap`取当前GT片段与前后GT片段之间较小的background步数；0表示两个动作直接相邻。它只用于分组诊断，不进入模型或decoder。

## 8. 校准后优先人工检查的40个Test片段

下表按校准后severity排序。完整列表见Excel的`Calibrated hard`；其中`calibration_outcome`可直接筛选`persistent_hard_error`和`new_hard_error`。

{_markdown_table(calibrated_top_segments, [('Run','sample_name'),('Type','split'),('GT #','gt_segment_ordinal'),('Start timestamp','gt_start_timestamp'),('End timestamp','gt_end_timestamp'),('Action','action'),('Object','object'),('Calibration outcome','calibration_outcome'),('Calibrated error','calibrated_error_type'),('Start nearest error','calibrated_nearest_start_error_frames'),('End nearest error','calibrated_nearest_end_error_frames'),('IoU','calibrated_segment_iou'),('Gap','nearest_background_gap_steps')])}

## 9. 原始decoder结果（保留作基线）

{_markdown_table(split_summary, [('Split','split'),('GT segments','segments'),('Start miss ±5','start_miss_rate_5'),('End miss ±5','end_miss_rate_5'),('Any boundary miss ±5','any_boundary_miss_rate_5'),('Segment miss IoU 0.5','segment_miss_rate_iou50'),('Hard error','hard_error_rate')], percent)}

![Boundary error by split](boundary_segment_error_audit/boundary_error_by_split.png)

原始decoder在held-out A输出{original_test_predictions}个片段，明显多于431个GT片段，因此原始event hit可能被大量碎片抬高；不应把原始start/end miss较低直接理解为整体分割更好。

## 10. 如何人工核查

1. 先在Excel的`Calibrated hard`中筛选`phase=heldout_test`、`calibration_outcome`和`calibrated_severity_score`。
2. 用`gt_start_frame_path`、`gt_end_frame_path`直接打开真实边界图像。
3. 检查`calibrated_nearest_start/end_error_frames`的正负：负数为预测偏早，正数为预测偏晚。
4. 若start/end都在±5内但`calibrated_segment_miss_iou50=True`，重点检查两个边界是否来自不同预测片段、片段是否跨越相邻动作，或短gap合并是否过度。
5. 对`gap_bin=0`或`1-7`的记录，重点判断相邻动作是否在视觉上确实没有可分background。
6. 在`Before-after GT`中筛选`new_hard_error=True`，检查提高阈值或3-step合并是否损害了原来正确的短动作。

## 11. 数据和代码依据

- GT边界、时间戳、动作和物体：`cache/features/A_as_test/all_runs/seed_1/stride_1/*.pt`及对应frame annotation；
- 模型：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/best.pth`，epoch {int(checkpoint['epoch'])}；
- 原始decoder：`resolved_config.json`中的`online`参数；正式test基线直接使用原`predicted_segments.jsonl`的{original_test_predictions}条记录；
- 校准decoder：参数来自`tools/plot_first_round_boundary_comparison.py`和`docs/boundary_visualization_summary.json`，选择策略为12个training-side validation runs only；
- 校准后每个split均从同一checkpoint概率重新解码，再执行不超过{CALIBRATED_MERGE_GAP}步的短gap合并；held-out A标签只用于最终审计，不用于选参数；
- 生成脚本：`tools/analyze_boundary_segment_errors.py`。

## 12. 解释限制

- 这是单fold、单seed结果，类别和run排序不能外推为总体稳定困难度；
- 训练集是最终checkpoint回放，不是逐epoch训练轨迹；
- 校准结果是decoder后处理改进，不代表boundary head本身已重新训练；
- 3-step gap合并属于有界holdback语义，正式在线部署时必须计入额外输出延迟；
- 当前只分析边界检测，不评价M3 node类别是否正确。
"""
    artifacts["report"].write_text(report, encoding="utf-8")
    print(json.dumps(summary_payload["scope"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
