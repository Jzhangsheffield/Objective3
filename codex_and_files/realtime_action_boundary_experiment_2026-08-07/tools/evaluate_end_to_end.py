from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_frame_table, load_run_index
from boundary_experiment.config import format_path, load_config
from boundary_experiment.engine import infer_run, load_boundary_checkpoint
from boundary_experiment.features import build_frozen_backbone, extract_closed_segment_feature, load_feature_cache
from boundary_experiment.m3_adapter import M3AtomicTailOnlineAdapter
from boundary_experiment.metrics import segment_iou
from boundary_experiment.online import run_state_machine
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import read_jsonl, resolve_device, write_json


def _node_rows_by_run(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[(str(row["participant"]), str(row["run"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["annotation_row_index"]))
    return grouped


def _ground_truth_segments(table: dict, node_rows: list[dict]) -> list[dict]:
    result = []
    action_segment_ids = []
    for segment_id in table["segment_no"]:
        segment_id = int(segment_id)
        if segment_id not in action_segment_ids and np.any((table["segment_no"] == segment_id) & (table["state"] == 1)):
            action_segment_ids.append(segment_id)
    if len(action_segment_ids) != len(node_rows):
        raise ValueError(f"Action/node count mismatch: annotation={len(action_segment_ids)} node_manifest={len(node_rows)}")
    for segment_id, node in zip(action_segment_ids, node_rows):
        indices = np.flatnonzero((table["segment_no"] == segment_id) & (table["state"] == 1))
        result.append({
            "start_row": int(indices[0]), "end_row": int(indices[-1]),
            "start_frame": int(table["original_frame_idx"][indices[0]]),
            "end_frame": int(table["original_frame_idx"][indices[-1]]),
            "node_idx": int(node["node_idx"]), "annotation_row_index": int(node["annotation_row_index"]),
        })
    return result


def _match(gt: list[dict], predictions: list[dict], threshold: float) -> dict[int, int]:
    candidates = []
    for pred_index, pred in enumerate(predictions):
        p = (pred["start_original_frame_idx"], pred["end_original_frame_idx"])
        for gt_index, target in enumerate(gt):
            g = (target["start_frame"], target["end_frame"])
            iou = segment_iou(p, g)
            if iou >= threshold:
                candidates.append((-iou, pred_index, gt_index))
    used_pred, used_gt, matches = set(), set(), {}
    for _, pred_index, gt_index in sorted(candidates):
        if pred_index not in used_pred and gt_index not in used_gt:
            used_pred.add(pred_index); used_gt.add(gt_index); matches[pred_index] = gt_index
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end detected-segment M3 Node Accuracy without oracle boundaries/history")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--splits", nargs="+", default=["test_normal", "test_fault", "test_all"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    experiment_output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    boundary_model, _ = load_boundary_checkpoint(experiment_output / "best.pth", device)
    backbone_checkpoint = format_path(cfg["paths"]["backbone_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    backbone, _ = build_frozen_backbone(cfg["paths"]["atomic_project_root"], backbone_checkpoint, device)
    m3_checkpoint = format_path(cfg["paths"]["m3_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    atomic_protocol = format_path(cfg["paths"]["atomic_protocol_template"], heldout=args.heldout, scope=args.scope)
    node_rows = _node_rows_by_run(atomic_protocol)
    run_index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    cache_root = format_path(cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed, scope=args.scope, stride=cfg["features"]["stride_frames"])

    for split in args.splits:
        split_protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / f"{split}.jsonl"
        runs = load_protocol_runs(split_protocol)
        all_records = []
        totals = {"gt": 0, "pred": 0, "matched": 0, "correct": 0}
        recognizer = M3AtomicTailOnlineAdapter(
            cfg["paths"]["atomic_project_root"], m3_checkpoint, device,
            max_history=int(cfg["m3"]["max_history"]), task_graph_path=cfg["paths"]["task_graph"],
        )
        for run_position, run in enumerate(runs, 1):
            recognizer.reset()
            info = run_index[run]
            table = load_frame_table(info)
            gt = _ground_truth_segments(table, node_rows[(info.participant, info.source_run)])
            cache = load_feature_cache(cache_root / f"{run}.pt")
            probabilities = infer_run(boundary_model, cache, device)
            detected = run_state_machine(**probabilities, settings=cfg["online"])
            predictions = []
            for segment_number, segment in enumerate(detected, 1):
                start_row = int(cache["anchor_row_index"][segment["start_index"]])
                end_row = int(cache["anchor_row_index"][segment["end_index"]])
                feature = extract_closed_segment_feature(table["frame_paths"], start_row, end_row, backbone, device, cfg["features"])
                predictions.append({
                    "sample_name": run, "detected_segment_no": segment_number, **segment,
                    "start_original_frame_idx": int(table["original_frame_idx"][start_row]),
                    "end_original_frame_idx": int(table["original_frame_idx"][end_row]),
                    **recognizer.predict(feature),
                })
            matches = _match(gt, predictions, float(cfg["evaluation"]["node_matching_iou"]))
            for pred_index, prediction in enumerate(predictions):
                gt_index = matches.get(pred_index)
                prediction["matched_gt_index"] = gt_index
                prediction["target_node_idx"] = gt[gt_index]["node_idx"] if gt_index is not None else None
                prediction["node_correct"] = gt_index is not None and prediction["node_idx"] == gt[gt_index]["node_idx"]
                all_records.append(prediction)
            totals["gt"] += len(gt); totals["pred"] += len(predictions); totals["matched"] += len(matches)
            totals["correct"] += sum(int(row["node_correct"]) for row in predictions)
            print(f"[{run_position}/{len(runs)}] {split} {run}: gt={len(gt)} pred={len(predictions)} matched={len(matches)}")
        metrics = {
            **totals,
            "detection_precision_at_iou": totals["matched"] / totals["pred"] if totals["pred"] else 0.0,
            "detection_recall_at_iou": totals["matched"] / totals["gt"] if totals["gt"] else 0.0,
            "conditional_node_accuracy": totals["correct"] / totals["matched"] if totals["matched"] else 0.0,
            "end_to_end_node_accuracy": totals["correct"] / totals["gt"] if totals["gt"] else 0.0,
            "matching_iou": float(cfg["evaluation"]["node_matching_iou"]),
        }
        target = experiment_output / "end_to_end" / split
        write_json(target / "metrics.json", metrics)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "predicted_nodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_records:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(split, metrics)


if __name__ == "__main__":
    main()
