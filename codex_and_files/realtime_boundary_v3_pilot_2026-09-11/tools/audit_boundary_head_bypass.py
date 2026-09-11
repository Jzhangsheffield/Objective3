from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_run_index
from boundary_experiment.engine import infer_run, load_boundary_checkpoint
from boundary_experiment.features import load_feature_cache
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import read_csv, resolve_device, write_json


def _gt_fields(cache: dict[str, Any], index: int, prefix: str = "gt") -> dict[str, Any]:
    exact_start = bool(float(cache["exact_start"][index]) > 0)
    exact_end = bool(float(cache["exact_end"][index]) > 0)
    state = bool(int(cache["state"][index]))
    if exact_start and exact_end:
        label = "start+end"
    elif exact_start:
        label = "start"
    elif exact_end:
        label = "end"
    elif state:
        label = "action"
    else:
        label = "background"
    return {
        f"{prefix}_class": label,
        f"{prefix}_exact_start": exact_start,
        f"{prefix}_exact_end": exact_end,
        f"{prefix}_action_state": state,
        f"{prefix}_action": cache["action"][index],
        f"{prefix}_object": cache["object"][index],
        f"{prefix}_segment_no": int(cache["segment_no"][index]),
        f"{prefix}_training_start_target": float(cache["start"][index]),
        f"{prefix}_training_end_target": float(cache["end"][index]),
    }


def _frame_fields(cache: dict[str, Any], rows: list[dict[str, str]], camera_dir: Path, index: int, prefix: str) -> dict[str, Any]:
    row_index = int(cache["anchor_row_index"][index])
    row = rows[row_index]
    return {
        f"{prefix}_anchor_index": index,
        f"{prefix}_frame_idx": int(cache["frame_idx"][index]),
        f"{prefix}_original_frame_idx": int(cache["original_frame_idx"][index]),
        f"{prefix}_frame_name": row["frame_name"],
        f"{prefix}_frame_path": str((camera_dir / row["frame_name"]).resolve()),
        f"{prefix}_timestamp": cache["timestamps"][index],
    }


def _probability_fields(probabilities: dict[str, np.ndarray], index: int, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_start_probability": float(probabilities["start_probability"][index]),
        f"{prefix}_end_probability": float(probabilities["end_probability"][index]),
        f"{prefix}_action_probability": float(probabilities["state_probability"][index]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit state-head bypasses in the original test-all decoder output")
    parser.add_argument("--condition-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--camera-id", default="001484412812")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    condition_root = Path(args.condition_root)
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((condition_root / "resolved_config.json").read_text(encoding="utf-8"))
    online = config["online"]
    start_threshold = float(online["start_threshold"])
    end_threshold = float(online["end_threshold"])
    action_threshold = float(online["action_threshold"])
    device = resolve_device(args.device)
    model, checkpoint = load_boundary_checkpoint(condition_root / "best.pth", device)
    run_index = load_run_index(args.dataset_root, args.annotation_root, args.camera_id)
    protocol_runs = load_protocol_runs(args.protocol)

    prediction_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in protocol_runs}
    with (condition_root / "evaluation" / "test_all" / "predicted_segments.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prediction_rows.setdefault(row["sample_name"], []).append(row)

    start_rows: list[dict[str, Any]] = []
    end_rows: list[dict[str, Any]] = []
    total_segments = 0
    for run_name in protocol_runs:
        cache_path = cache_root / f"{run_name}.pt"
        cache = load_feature_cache(cache_path)
        probabilities = infer_run(model, cache, device)
        info = run_index[run_name]
        annotation_rows = read_csv(info.frame_annotation)
        segments = prediction_rows[run_name]
        total_segments += len(segments)
        for ordinal, segment in enumerate(segments, 1):
            common = {
                "sample_name": run_name,
                "predicted_segment_ordinal": ordinal,
                "predicted_start_anchor_index": int(segment["start_index"]),
                "predicted_end_anchor_index": int(segment["end_index"]),
                "emitted_at_anchor_index": int(segment["emitted_at_index"]),
                "recorded_start_score": float(segment["start_score"]),
                "recorded_end_score": float(segment["end_score"]),
                "annotation_file": str(info.frame_annotation.resolve()),
                "feature_cache": str(cache_path.resolve()),
                "checkpoint": str((condition_root / "best.pth").resolve()),
            }
            if float(segment["start_score"]) < start_threshold:
                decision = int(segment["start_index"])
                confirmation = min(decision + int(online["start_debounce"]) - 1, len(cache["state"]) - 1)
                row = {
                    "audit_id": f"START-{len(start_rows) + 1:04d}",
                    "bypass_type": "start_head_bypassed_by_action_state",
                    **common,
                    "start_threshold": start_threshold,
                    "action_start_threshold": action_threshold,
                    **_frame_fields(cache, annotation_rows, info.camera_dir, decision, "decision"),
                    **_probability_fields(probabilities, decision, "decision"),
                    **_gt_fields(cache, decision, "decision_gt"),
                    **_frame_fields(cache, annotation_rows, info.camera_dir, confirmation, "confirmation"),
                    **_probability_fields(probabilities, confirmation, "confirmation"),
                    **_gt_fields(cache, confirmation, "confirmation_gt"),
                }
                row["decision_start_head_met"] = row["decision_start_probability"] >= start_threshold
                row["decision_action_state_met"] = row["decision_action_probability"] >= action_threshold
                row["confirmation_start_head_met"] = row["confirmation_start_probability"] >= start_threshold
                row["confirmation_action_state_met"] = row["confirmation_action_probability"] >= action_threshold
                start_rows.append(row)
            if float(segment["end_score"]) < end_threshold:
                boundary = int(segment["end_index"])
                decision = min(int(segment["emitted_at_index"]), len(cache["state"]) - 1)
                row = {
                    "audit_id": f"END-{len(end_rows) + 1:04d}",
                    "bypass_type": "end_head_bypassed_by_low_action_state",
                    **common,
                    "end_threshold": end_threshold,
                    "action_end_threshold": 1.0 - action_threshold,
                    **_frame_fields(cache, annotation_rows, info.camera_dir, decision, "decision"),
                    **_probability_fields(probabilities, decision, "decision"),
                    **_gt_fields(cache, decision, "decision_gt"),
                    **_frame_fields(cache, annotation_rows, info.camera_dir, boundary, "predicted_boundary"),
                    **_probability_fields(probabilities, boundary, "predicted_boundary"),
                    **_gt_fields(cache, boundary, "predicted_boundary_gt"),
                }
                row["decision_end_head_met"] = row["decision_end_probability"] >= end_threshold
                row["decision_low_action_state_met"] = row["decision_action_probability"] < (1.0 - action_threshold)
                end_rows.append(row)

    summary = {
        "scope": "original A-as-test, seed-1, all-runs, test_all output",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_runs": len(protocol_runs),
        "predicted_segments": total_segments,
        "start_bypass_rows": len(start_rows),
        "start_bypass_percent": 100.0 * len(start_rows) / max(total_segments, 1),
        "end_bypass_rows": len(end_rows),
        "end_bypass_percent": 100.0 * len(end_rows) / max(total_segments, 1),
        "thresholds": {
            "start": start_threshold,
            "end": end_threshold,
            "action_start": action_threshold,
            "low_action_end": 1.0 - action_threshold,
        },
        "classification_rule": "exact start > exact end > action interior > background; start+end is retained if both flags are true",
        "important_note": "End decision_frame is emitted_at_index (the frame that completes end debounce); predicted_boundary_frame is end_index (the first frame in that debounce run).",
    }
    _write_csv(output_root / "start_head_bypass_frames.csv", start_rows)
    _write_csv(output_root / "end_head_bypass_frames.csv", end_rows)
    write_json(output_root / "boundary_head_bypass_audit.json", {"summary": summary, "start_rows": start_rows, "end_rows": end_rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
