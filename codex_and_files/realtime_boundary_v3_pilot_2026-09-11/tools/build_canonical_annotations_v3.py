#!/usr/bin/env python
"""Build canonical annotation v3 from v2 plus reviewed Excel corrections.

The input v2 directory is read-only.  Corrections are matched by sample,
action/object, and timestamp identity, never by the potentially stale indices
shown in the review workbook.  The output is first written to a sibling staging
directory, validated, and then atomically renamed to the requested v3 path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


VERSION = "action_recognition_timestamps_canonical_v3"
BAD_FRAME = "19700101_010000_000000.jpg"
SEGMENT_FIELDS = ["No", "action", "object", "start_idx", "end_idx", "start", "end", "mark"]
FRAME_FIELDS = [
    "frame_idx", "original_frame_idx", "frame_name", "timestamp", "action", "object", "mark",
    "segment_no", "segment_start_idx", "segment_end_idx", "segment_start", "segment_end",
]
CORRECTION_PATTERN = re.compile(
    r"^(first|second)_(start|end)_frame_name\s*:\s*(.*?)\s*$", re.IGNORECASE
)
REMOVE_PATTERN = re.compile(r"^remove\s+(.+?)\s+segment\s*$", re.IGNORECASE)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def timestamp(value: str) -> str:
    return Path(str(value).strip()).stem


def frame_name(value: str) -> str:
    return f"{timestamp(value)}.jpg"


def valid_reference_frames(record: dict[str, Any]) -> list[str]:
    source_run_dir = Path(str(record["source_action_annotation"])).parent
    image_dir = source_run_dir / str(record["reference_camera"]) / "blurred_imgs"
    frames = sorted(
        path.name for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.name != BAD_FRAME
    )
    if len(frames) != int(record["reference_frame_count"]):
        raise ValueError(
            f"Reference frame count mismatch for {record['sample_name']}: "
            f"{len(frames)} != {record['reference_frame_count']}"
        )
    expected_hash = str(record.get("reference_frame_name_list_sha256", ""))
    if expected_hash and sha256_names(frames) != expected_hash:
        raise ValueError(f"Reference frame-name hash mismatch for {record['sample_name']}")
    return frames


def load_corrections(path: Path) -> tuple[str, list[dict[str, Any]], Counter]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if len(workbook.sheetnames) != 1:
        raise ValueError(f"Expected one worksheet, found {workbook.sheetnames}")
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    required = {
        "sample_id", "participant", "source_run", "first_action", "first_object",
        "first_start_idx", "first_start_frame_name", "first_end_idx", "first_end_frame_name",
        "second_action", "second_object", "second_start_idx", "second_start_frame_name",
        "second_end_idx", "second_end_frame_name", "reference_camera", "correction",
    }
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(f"Correction workbook is missing columns: {missing}")

    parsed_rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for excel_row, values in enumerate(rows, start=2):
        if all(value is None or str(value).strip() == "" for value in values):
            continue
        row = {headers[index]: values[index] if index < len(values) else None for index in range(len(headers))}
        raw = str(row["correction"] or "").strip()
        if not raw:
            raise ValueError(f"Blank correction at worksheet row {excel_row}")
        operations: list[dict[str, str]] = []
        blank_fields: list[str] = []
        for original_line in raw.replace("；", "\n").splitlines():
            line = original_line.strip().replace("：", ":")
            if not line:
                continue
            remove_match = REMOVE_PATTERN.fullmatch(line)
            if remove_match:
                operations.append({"kind": "remove_second", "label": remove_match.group(1).strip()})
                stats["remove_operations"] += 1
                continue
            boundary_match = CORRECTION_PATTERN.fullmatch(line)
            if not boundary_match:
                raise ValueError(f"Unrecognized correction at worksheet row {excel_row}: {original_line!r}")
            role, edge, value = boundary_match.groups()
            field = f"{role.lower()}_{edge.lower()}"
            if not value.strip():
                blank_fields.append(field)
                stats["blank_boundary_fields"] += 1
                continue
            corrected_timestamp = timestamp(value)
            if not re.fullmatch(r"\d{8}_\d{6}_\d{6}", corrected_timestamp):
                raise ValueError(
                    f"Invalid corrected frame timestamp at worksheet row {excel_row}: {value!r}"
                )
            operations.append({"kind": "boundary", "field": field, "timestamp": corrected_timestamp})
            stats["boundary_operations"] += 1
            stats[f"{field}_operations"] += 1
        if not operations:
            raise ValueError(f"No effective correction at worksheet row {excel_row}")
        row["excel_row"] = excel_row
        row["operations"] = operations
        row["blank_fields"] = blank_fields
        row["correction_raw"] = raw
        parsed_rows.append(row)
    stats["workbook_rows"] = len(parsed_rows)
    sheet_title = sheet.title
    workbook.close()
    if len(parsed_rows) != 35:
        raise ValueError(f"Expected 35 reviewed rows, found {len(parsed_rows)}")
    return sheet_title, parsed_rows, stats


def normalize_segment_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        current: dict[str, Any] = dict(row)
        current["start_idx"] = int(current["start_idx"])
        current["end_idx"] = int(current["end_idx"])
        current["No"] = int(current["No"])
        current["_original_no"] = int(current["No"])
        current["_original_start_idx"] = int(current["start_idx"])
        current["_original_end_idx"] = int(current["end_idx"])
        current["_original_start"] = str(current["start"])
        current["_original_end"] = str(current["end"])
        current["_removed"] = False
        current["_direct_corrections"] = {}
        result.append(current)
    return result


def pair_match(
    actions: list[dict[str, Any]], workbook_row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Locate the reviewed pair in v2 despite the known pre-v2 one-frame shift.

    Workbook indices and anchor names describe the version that was reviewed
    before the 1970/index repair.  Labels, order, and approximate position are
    therefore used only to locate the pair; correction values themselves are
    later resolved by exact timestamp identity in the canonical v2 frame list.
    """
    first_label = (
        str(workbook_row["first_action"]).strip(), str(workbook_row["first_object"]).strip()
    )
    second_label = (
        str(workbook_row["second_action"]).strip(), str(workbook_row["second_object"]).strip()
    )
    workbook_indices = [
        int(workbook_row["first_start_idx"]), int(workbook_row["first_end_idx"]),
        int(workbook_row["second_start_idx"]), int(workbook_row["second_end_idx"]),
    ]
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for first, second in zip(actions, actions[1:]):
        if (first["action"], first["object"]) != first_label:
            continue
        if (second["action"], second["object"]) != second_label:
            continue
        if int(first["end_idx"]) + 1 != int(second["start_idx"]):
            continue
        candidate_indices = [
            int(first["start_idx"]), int(first["end_idx"]),
            int(second["start_idx"]), int(second["end_idx"]),
        ]
        deltas = [abs(left - right) for left, right in zip(workbook_indices, candidate_indices)]
        candidates.append((sum(deltas), max(deltas), first, second))
    if not candidates:
        raise ValueError(
            f"Worksheet row {workbook_row['excel_row']} has no adjacent v2 pair matching "
            f"{first_label} -> {second_label}"
        )
    candidates.sort(key=lambda item: (item[0], item[1], int(item[2]["start_idx"])))
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0:2] == best[0:2]:
        raise ValueError(
            f"Worksheet row {workbook_row['excel_row']} pair position is ambiguous: "
            f"score={best[0]}, max_delta={best[1]}"
        )
    if best[1] > 2:
        raise ValueError(
            f"Worksheet row {workbook_row['excel_row']} nearest pair is too far from reviewed position: "
            f"score={best[0]}, max_delta={best[1]}"
        )
    return best[2], best[3], best[0], best[1]


def background(start_idx: int, end_idx: int, frames: list[str]) -> dict[str, Any]:
    return {
        "No": 0,
        "action": "background",
        "object": "none",
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start": timestamp(frames[start_idx - 1]),
        "end": timestamp(frames[end_idx - 1]),
        "mark": "none",
    }


def rebuild_segments(actions: list[dict[str, Any]], frames: list[str]) -> list[dict[str, Any]]:
    kept = sorted(
        (row for row in actions if not row["_removed"]),
        key=lambda row: (int(row["start_idx"]), int(row["end_idx"]), int(row["_original_no"])),
    )
    result: list[dict[str, Any]] = []
    cursor = 1
    for action in kept:
        start_idx, end_idx = int(action["start_idx"]), int(action["end_idx"])
        if not (1 <= start_idx <= end_idx <= len(frames)):
            raise ValueError(f"Invalid corrected action interval: {action}")
        if start_idx < cursor:
            raise ValueError(f"Corrected action overlap before {action['action']}/{action['object']}: {action}")
        if start_idx > cursor:
            result.append(background(cursor, start_idx - 1, frames))
        result.append({field: action[field] for field in SEGMENT_FIELDS})
        cursor = end_idx + 1
    if cursor <= len(frames):
        result.append(background(cursor, len(frames), frames))
    for number, row in enumerate(result, start=1):
        row["No"] = number
    return result


def validate_segments(rows: list[dict[str, Any]], frames: list[str]) -> None:
    expected = 1
    for number, row in enumerate(rows, start=1):
        start_idx, end_idx = int(row["start_idx"]), int(row["end_idx"])
        if int(row["No"]) != number:
            raise ValueError(f"Segment number mismatch at {number}")
        if start_idx != expected or end_idx < start_idx:
            raise ValueError(f"Non-contiguous segment: {row}")
        if str(row["start"]) != timestamp(frames[start_idx - 1]):
            raise ValueError(f"Segment start timestamp/index mismatch: {row}")
        if str(row["end"]) != timestamp(frames[end_idx - 1]):
            raise ValueError(f"Segment end timestamp/index mismatch: {row}")
        expected = end_idx + 1
    if expected != len(frames) + 1:
        raise ValueError(f"Segment coverage ends at {expected - 1}, expected {len(frames)}")


def rebuild_frame_rows(
    old_rows: list[dict[str, str]], segments: list[dict[str, Any]], frames: list[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    segment_pos = 0
    for old in old_rows:
        original_idx = int(old["original_frame_idx"])
        while int(segments[segment_pos]["end_idx"]) < original_idx:
            segment_pos += 1
        segment = segments[segment_pos]
        if not (int(segment["start_idx"]) <= original_idx <= int(segment["end_idx"])):
            raise ValueError(f"No corrected segment covers original frame {original_idx}")
        expected_name = frames[original_idx - 1]
        if old["frame_name"] != expected_name or timestamp(old["timestamp"]) != timestamp(expected_name):
            raise ValueError(f"Frame identity mismatch at original frame {original_idx}")
        result.append({
            "frame_idx": int(old["frame_idx"]),
            "original_frame_idx": original_idx,
            "frame_name": old["frame_name"],
            "timestamp": old["timestamp"],
            "action": segment["action"],
            "object": segment["object"],
            "mark": segment["mark"],
            "segment_no": segment["No"],
            "segment_start_idx": segment["start_idx"],
            "segment_end_idx": segment["end_idx"],
            "segment_start": segment["start"],
            "segment_end": segment["end"],
        })
    return result


def compare_frames(old_rows: list[dict[str, str]], new_rows: list[dict[str, Any]]) -> Counter:
    if len(old_rows) != len(new_rows):
        raise ValueError("Frame annotation row count changed")
    stats: Counter = Counter()
    for old, new in zip(old_rows, new_rows):
        if old["frame_name"] != str(new["frame_name"]):
            raise ValueError("Frame annotation identity/order changed")
        stats["frame_label_rows_changed"] += (
            old["action"] != str(new["action"]) or old["object"] != str(new["object"])
        )
        stats["frame_rows_changed"] += any(
            str(old.get(field, "")) != str(new[field]) for field in FRAME_FIELDS
        )
    return stats


def public_action_state(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_idx": int(action["start_idx"]),
        "end_idx": int(action["end_idx"]),
        "start": str(action["start"]),
        "end": str(action["end"]),
        "removed": bool(action["_removed"]),
    }


def process_run(
    annotation_root: Path,
    record: dict[str, Any],
    workbook_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], Counter
]:
    sample = str(record["sample_name"])
    seg_path = annotation_root / str(record["segmentation_annotation_csv"])
    frame_path = annotation_root / str(record["frame_annotation_csv"])
    seg_fields, old_segment_strings = read_csv(seg_path)
    frame_fields, old_frame_rows = read_csv(frame_path)
    if seg_fields != SEGMENT_FIELDS:
        raise ValueError(f"Unexpected segmentation schema in {seg_path}: {seg_fields}")
    if frame_fields != FRAME_FIELDS:
        raise ValueError(f"Unexpected frame schema in {frame_path}: {frame_fields}")
    frames = valid_reference_frames(record)
    old_segments = normalize_segment_rows(old_segment_strings)
    validate_segments(old_segments, frames)
    actions = [row for row in old_segments if row["action"] != "background"]
    original_actions = [dict(row) for row in actions]
    pair_refs: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    stats: Counter = Counter()

    for workbook_row in workbook_rows:
        first, second, mapping_score, mapping_max_delta = pair_match(actions, workbook_row)
        index_match = all(
            int(workbook_row[f"{role}_{edge}_idx"]) == int(target[f"{edge}_idx"])
            for role, target in (("first", first), ("second", second))
            for edge in ("start", "end")
        )
        name_match = all(
            timestamp(str(workbook_row[f"{role}_{edge}_frame_name"])) == str(target[edge])
            for role, target in (("first", first), ("second", second))
            for edge in ("start", "end")
        )
        stats["workbook_index_rows_matching_v2"] += index_match
        stats["workbook_index_rows_differing_from_v2"] += not index_match
        stats["workbook_name_rows_matching_v2"] += name_match
        stats["workbook_name_rows_differing_from_v2"] += not name_match
        stats["pair_mapping_score_total"] += mapping_score
        stats["pair_mapping_max_abs_delta"] = max(stats["pair_mapping_max_abs_delta"], mapping_max_delta)
        workbook_row["pair_mapping_score"] = mapping_score
        workbook_row["pair_mapping_max_abs_delta"] = mapping_max_delta
        workbook_row["workbook_names_match_v2"] = name_match
        pair_refs[int(workbook_row["excel_row"])] = (first, second, workbook_row)

    timestamp_to_idx = {timestamp(name): index for index, name in enumerate(frames, start=1)}
    if len(timestamp_to_idx) != len(frames):
        raise ValueError(f"Duplicate reference timestamp in {sample}")
    for excel_row, (first, second, workbook_row) in pair_refs.items():
        for operation in workbook_row["operations"]:
            if operation["kind"] == "remove_second":
                expected_label = f"{second['action']} {second['object']}".strip().lower()
                if operation["label"].strip().lower() != expected_label:
                    raise ValueError(
                        f"Worksheet row {excel_row} removal label {operation['label']!r} "
                        f"does not match {expected_label!r}"
                    )
                if second["_removed"]:
                    raise ValueError(f"Worksheet row {excel_row} removes an action twice")
                second["_removed"] = True
                stats["action_segments_removed"] += 1
                continue
            role, edge = str(operation["field"]).split("_", maxsplit=1)
            target = first if role == "first" else second
            corrected_timestamp = operation["timestamp"]
            if corrected_timestamp not in timestamp_to_idx:
                raise ValueError(
                    f"Worksheet row {excel_row} correction timestamp is not a valid frame: "
                    f"{corrected_timestamp}"
                )
            corrected_idx = timestamp_to_idx[corrected_timestamp]
            previous = target["_direct_corrections"].get(edge)
            if previous is not None and previous != corrected_idx:
                raise ValueError(f"Conflicting corrections for {sample} action {target['_original_no']} {edge}")
            target["_direct_corrections"][edge] = corrected_idx
            old_idx = int(target[f"{edge}_idx"])
            target[f"{edge}_idx"] = corrected_idx
            target[edge] = corrected_timestamp
            stats["boundary_fields_changed"] += old_idx != corrected_idx
            stats["boundary_fields_unchanged"] += old_idx == corrected_idx

    new_segments = rebuild_segments(actions, frames)
    validate_segments(new_segments, frames)
    new_frame_rows = rebuild_frame_rows(old_frame_rows, new_segments, frames)
    frame_stats = compare_frames(old_frame_rows, new_frame_rows)
    stats.update(frame_stats)
    stats["correction_rows"] = len(workbook_rows)
    stats["corrected_run"] = bool(workbook_rows)
    stats["segments_before"] = len(old_segments)
    stats["segments_after"] = len(new_segments)
    stats["action_segments_before"] = sum(row["action"] != "background" for row in old_segments)
    stats["action_segments_after"] = sum(row["action"] != "background" for row in new_segments)
    stats["background_segments_before"] = sum(row["action"] == "background" for row in old_segments)
    stats["background_segments_after"] = sum(row["action"] == "background" for row in new_segments)
    stats["frame_rows"] = len(new_frame_rows)

    old_by_no = {int(row["_original_no"]): row for row in original_actions}
    audit_rows: list[dict[str, Any]] = []
    for excel_row in sorted(pair_refs):
        first, second, workbook_row = pair_refs[excel_row]
        old_first = old_by_no[int(first["_original_no"])]
        old_second = old_by_no[int(second["_original_no"])]
        operation_text = []
        for operation in workbook_row["operations"]:
            if operation["kind"] == "remove_second":
                operation_text.append("remove_second")
            else:
                operation_text.append(f"{operation['field']}={operation['timestamp']}")
        gap_start = ""
        gap_end = ""
        gap_frames = 0
        if not second["_removed"]:
            gap_frames = max(0, int(second["start_idx"]) - int(first["end_idx"]) - 1)
            if gap_frames:
                gap_start = int(first["end_idx"]) + 1
                gap_end = int(second["start_idx"]) - 1
        audit_rows.append({
            "workbook_sheet": workbook_row["workbook_sheet"],
            "workbook_row": excel_row,
            "sample_id": sample,
            "participant": workbook_row["participant"],
            "source_run": workbook_row["source_run"],
            "reference_camera": workbook_row["reference_camera"],
            "correction_raw": workbook_row["correction_raw"],
            "parsed_operations": ";".join(operation_text),
            "blank_fields_no_change": ";".join(workbook_row["blank_fields"]),
            "pair_mapping_score": workbook_row["pair_mapping_score"],
            "pair_mapping_max_abs_delta": workbook_row["pair_mapping_max_abs_delta"],
            "workbook_boundary_names_match_v2": workbook_row["workbook_names_match_v2"],
            "workbook_indices_match_v2": all(
                int(workbook_row[f"{role}_{edge}_idx"]) == int(target[f"_original_{edge}_idx"])
                for role, target in (("first", first), ("second", second))
                for edge in ("start", "end")
            ),
            "first_action": first["action"],
            "first_object": first["object"],
            "first_old_start_idx": old_first["start_idx"],
            "first_old_end_idx": old_first["end_idx"],
            "first_old_start_frame_name": frame_name(old_first["start"]),
            "first_old_end_frame_name": frame_name(old_first["end"]),
            "first_new_start_idx": first["start_idx"],
            "first_new_end_idx": first["end_idx"],
            "first_new_start_frame_name": frame_name(first["start"]),
            "first_new_end_frame_name": frame_name(first["end"]),
            "second_action": second["action"],
            "second_object": second["object"],
            "second_old_start_idx": old_second["start_idx"],
            "second_old_end_idx": old_second["end_idx"],
            "second_old_start_frame_name": frame_name(old_second["start"]),
            "second_old_end_frame_name": frame_name(old_second["end"]),
            "second_new_start_idx": "" if second["_removed"] else second["start_idx"],
            "second_new_end_idx": "" if second["_removed"] else second["end_idx"],
            "second_new_start_frame_name": "" if second["_removed"] else frame_name(second["start"]),
            "second_new_end_frame_name": "" if second["_removed"] else frame_name(second["end"]),
            "second_removed": second["_removed"],
            "inserted_background_start_idx": gap_start,
            "inserted_background_end_idx": gap_end,
            "inserted_background_frame_count": gap_frames,
            "removed_segment_old_start_idx": old_second["start_idx"] if second["_removed"] else "",
            "removed_segment_old_end_idx": old_second["end_idx"] if second["_removed"] else "",
            "removed_segment_frame_count": (
                int(old_second["end_idx"]) - int(old_second["start_idx"]) + 1 if second["_removed"] else 0
            ),
        })

    run_audit = {
        "sample_name": sample,
        "participant": record["participant"],
        "source_run": record["source_run"],
        **dict(stats),
    }
    return new_segments, new_frame_rows, run_audit, audit_rows, stats


def validate_written_run(
    output_root: Path,
    record: dict[str, Any],
    frames: list[str],
    expected_frame_names: list[str],
) -> None:
    seg_fields, segments = read_csv(output_root / str(record["segmentation_annotation_csv"]))
    frame_fields, frame_rows = read_csv(output_root / str(record["frame_annotation_csv"]))
    if seg_fields != SEGMENT_FIELDS or frame_fields != FRAME_FIELDS:
        raise ValueError(f"Written schema mismatch for {record['sample_name']}")
    validate_segments(segments, frames)
    if [row["frame_name"] for row in frame_rows] != expected_frame_names:
        raise ValueError(f"Written frame identities changed for {record['sample_name']}")
    segment_pos = 0
    for row in frame_rows:
        original_idx = int(row["original_frame_idx"])
        while int(segments[segment_pos]["end_idx"]) < original_idx:
            segment_pos += 1
        segment = segments[segment_pos]
        expected = {
            "action": segment["action"], "object": segment["object"], "mark": segment["mark"],
            "segment_no": segment["No"], "segment_start_idx": segment["start_idx"],
            "segment_end_idx": segment["end_idx"], "segment_start": segment["start"],
            "segment_end": segment["end"],
        }
        if any(str(row[key]) != str(value) for key, value in expected.items()):
            raise ValueError(f"Written frame/segment inconsistency for {record['sample_name']} at {original_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical v3 annotations from v2 and manual review")
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output_root is None:
        parser.error("--output-root is required unless --dry-run is used")
    if args.output_root is not None and args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_root}")

    v2_root = args.v2_root.resolve()
    correction_path = args.corrections.resolve()
    v2_tree_before = tree_sha256(v2_root)
    correction_hash_before = sha256_file(correction_path)
    sheet_name, correction_rows, correction_stats = load_corrections(correction_path)
    for row in correction_rows:
        row["workbook_sheet"] = sheet_name
    corrections_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in correction_rows:
        corrections_by_sample[str(row["sample_id"])].append(row)

    manifest_path = v2_root / "annotation_set_manifest.jsonl"
    parent_manifest_hash = sha256_file(manifest_path)
    v2_manifest = load_jsonl(manifest_path)
    manifest_samples = {str(row["sample_name"]) for row in v2_manifest}
    unknown_samples = sorted(set(corrections_by_sample) - manifest_samples)
    if unknown_samples:
        raise ValueError(f"Correction workbook references unknown samples: {unknown_samples}")

    staging: Path | None = None
    output_root: Path | None = None
    try:
        if not args.dry_run:
            output_root = args.output_root.resolve()
            output_root.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))

        totals: Counter = Counter()
        run_audit_rows: list[dict[str, Any]] = []
        manual_audit_rows: list[dict[str, Any]] = []
        v3_manifest: list[dict[str, Any]] = []
        expected_frame_names: dict[str, list[str]] = {}
        frames_by_sample: dict[str, list[str]] = {}

        for record in v2_manifest:
            sample = str(record["sample_name"])
            workbook_rows = corrections_by_sample.get(sample, [])
            new_segments, new_frame_rows, run_audit, manual_rows, run_stats = process_run(
                v2_root, record, workbook_rows
            )
            run_audit_rows.append(run_audit)
            manual_audit_rows.extend(manual_rows)
            run_mapping_max = int(run_stats.get("pair_mapping_max_abs_delta", 0))
            totals.update({key: value for key, value in run_stats.items() if key != "pair_mapping_max_abs_delta"})
            totals["pair_mapping_max_abs_delta"] = max(
                int(totals["pair_mapping_max_abs_delta"]), run_mapping_max
            )
            totals["runs"] += 1
            frames = valid_reference_frames(record)
            frames_by_sample[sample] = frames
            expected_frame_names[sample] = [str(row["frame_name"]) for row in new_frame_rows]

            seg_name = str(record["segmentation_annotation_csv"])
            frame_csv_name = str(record["frame_annotation_csv"])
            if staging is not None:
                if workbook_rows:
                    write_csv(staging / seg_name, SEGMENT_FIELDS, new_segments)
                    write_csv(staging / frame_csv_name, FRAME_FIELDS, new_frame_rows)
                else:
                    shutil.copy2(v2_root / seg_name, staging / seg_name)
                    shutil.copy2(v2_root / frame_csv_name, staging / frame_csv_name)

            v2_parent_seg = record.get("parent_segmentation_sha256", "")
            v2_parent_frame = record.get("parent_frame_annotation_sha256", "")
            new_record = {
                **record,
                "version_name": VERSION,
                "parent_annotation_version": str(record.get("version_name", v2_root.name)),
                "boundary_policy": (
                    "canonical_v2_timestamp_identity plus reviewed manual frame-name corrections; "
                    "one_based_inclusive_indices"
                ),
                "short_background_policy": "background_is_exact_complement_of_manually_corrected_action_intervals",
                "overlap_policy": "reject_corrected_action_overlap",
                "manual_correction_workbook": str(correction_path),
                "manual_correction_workbook_sha256": correction_hash_before,
                "manual_correction_sheet": sheet_name,
                "manual_correction_rows_for_run": len(workbook_rows),
                "v2_parent_segmentation_sha256": v2_parent_seg,
                "v2_parent_frame_annotation_sha256": v2_parent_frame,
                "parent_segmentation_sha256": sha256_file(v2_root / seg_name),
                "parent_frame_annotation_sha256": sha256_file(v2_root / frame_csv_name),
            }
            if staging is not None:
                new_record["segmentation_annotation_sha256"] = sha256_file(staging / seg_name)
                new_record["frame_annotation_sha256"] = sha256_file(staging / frame_csv_name)
            v3_manifest.append(new_record)

        if len(manual_audit_rows) != len(correction_rows):
            raise ValueError(
                f"Not all workbook rows were audited: {len(manual_audit_rows)} != {len(correction_rows)}"
            )
        corrected_runs = sum(bool(corrections_by_sample.get(str(row["sample_name"]))) for row in v2_manifest)
        manual_gap_frames = sum(int(row["inserted_background_frame_count"]) for row in manual_audit_rows)
        removed_frames = sum(int(row["removed_segment_frame_count"]) for row in manual_audit_rows)
        summary = {
            "version_name": VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_annotation_version": v2_root.name,
            "parent_manifest_sha256": parent_manifest_hash,
            "parent_tree_sha256_before": v2_tree_before,
            "correction_workbook": str(correction_path),
            "correction_workbook_sha256": correction_hash_before,
            "correction_worksheet": sheet_name,
            "correction_semantics": {
                "reviewed_pair_locator": (
                    "sample plus adjacent action/object order and nearest reviewed indices; "
                    "the workbook locator names/indices predate the canonical-v2 one-frame repair"
                ),
                "named_nonblank_boundary": "replace the named action endpoint by exact frame timestamp identity",
                "named_blank_boundary": "leave that endpoint unchanged",
                "remove_segment": "remove the named action segment and relabel its frames as background",
                "background": "exact complement of retained corrected action intervals",
            },
            "index_semantics": "one-based; start_idx and end_idx are inclusive valid-reference-frame positions",
            "timestamp_semantics": "start/end are timestamps of the first/last included frame",
            "invalid_frame_policy": f"inherit v2; exclude {BAD_FRAME} before indexing",
            "totals": {
                **dict(sorted(totals.items())),
                "corrected_runs": corrected_runs,
                "manual_gap_frames_across_reviewed_pairs": manual_gap_frames,
                "removed_action_frames": removed_frames,
                **dict(sorted(correction_stats.items())),
            },
            "validation": {
                "all_35_workbook_rows_parsed": len(correction_rows) == 35,
                "all_workbook_pairs_mapped_uniquely_to_v2": True,
                "pair_mapping_max_abs_index_delta_at_most_2": totals["pair_mapping_max_abs_delta"] <= 2,
                "all_correction_timestamps_resolved_uniquely": True,
                "all_corrected_actions_non_overlapping": True,
                "all_segments_contiguous_full_run": True,
                "all_segment_timestamps_match_indices": True,
                "all_frame_identities_and_order_preserved": True,
                "all_frame_rows_have_exactly_one_segment": True,
                "v1_and_v2_not_modified": True,
            },
        }

        if staging is not None and output_root is not None:
            run_fields: list[str] = []
            for row in run_audit_rows:
                for field in row:
                    if field not in run_fields:
                        run_fields.append(field)
            for row in run_audit_rows:
                for field in run_fields:
                    row.setdefault(field, 0)
            write_csv(staging / "generation_audit.csv", run_fields, run_audit_rows)
            manual_fields = list(manual_audit_rows[0])
            write_csv(staging / "manual_correction_audit.csv", manual_fields, manual_audit_rows)
            with (staging / "annotation_set_manifest.jsonl").open("w", encoding="utf-8") as handle:
                for record in v3_manifest:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            (staging / "generation_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (staging / "README.md").write_text(
                "# Canonical boundary annotations v3\n\n"
                "This version starts from `action_recognition_timestamps_canonical_v2` and applies all 35 "
                "reviewed rows in `adjacent_action_boundary_frame_names_2026-09-09.xlsx`. Corrections are "
                "matched by exact frame timestamp identity. A nonblank named endpoint replaces that endpoint; "
                "a named but blank endpoint is unchanged; `remove ... segment` removes that action and relabels "
                "its interval as background. Background is rebuilt as the exact complement of all retained "
                "action intervals. Indices remain one-based and inclusive, and the invalid 1970 placeholder "
                "policy is inherited from v2. Existing v1 and v2 files are not modified.\n\n"
                f"Correction workbook SHA-256: `{correction_hash_before}`\n",
                encoding="utf-8",
            )

            for record in v3_manifest:
                sample = str(record["sample_name"])
                validate_written_run(
                    staging, record, frames_by_sample[sample], expected_frame_names[sample]
                )
            if len(list(staging.glob("*_segmentation_annotation.csv"))) != len(v2_manifest):
                raise ValueError("Written segmentation file count mismatch")
            if len(list(staging.glob("*_frame_annotation.csv"))) != len(v2_manifest):
                raise ValueError("Written frame annotation file count mismatch")

        v2_tree_after = tree_sha256(v2_root)
        correction_hash_after = sha256_file(correction_path)
        if v2_tree_after != v2_tree_before:
            raise RuntimeError("v2 input tree changed during v3 generation")
        if correction_hash_after != correction_hash_before:
            raise RuntimeError("Correction workbook changed during v3 generation")
        summary["parent_tree_sha256_after"] = v2_tree_after
        summary["correction_workbook_sha256_after"] = correction_hash_after

        if staging is not None and output_root is not None:
            (staging / "generation_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            staging.replace(output_root)
            staging = None
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    except Exception:
        if staging is not None and staging.exists():
            resolved_parent = staging.parent.resolve()
            expected_parent = args.output_root.resolve().parent if args.output_root is not None else None
            if expected_parent is not None and resolved_parent == expected_parent and staging.name.startswith(
                f".{args.output_root.name}.building-"
            ):
                shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
