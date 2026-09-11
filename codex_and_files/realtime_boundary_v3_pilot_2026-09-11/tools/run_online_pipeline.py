from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_frame_table, load_run_index
from boundary_experiment.config import format_path, load_config
from boundary_experiment.engine import infer_run, load_boundary_checkpoint
from boundary_experiment.features import build_frozen_backbone, extract_closed_segment_feature, load_feature_cache
from boundary_experiment.m3_adapter import M3AtomicTailOnlineAdapter
from boundary_experiment.online import run_state_machine
from boundary_experiment.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal boundary detection followed by closed-segment M3 node recognition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--run", required=True, help="Structured run name, e.g. run_sample_000001")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    output_root = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    boundary_model, _ = load_boundary_checkpoint(output_root / "best.pth", device)
    backbone_checkpoint = format_path(cfg["paths"]["backbone_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    backbone, _ = build_frozen_backbone(cfg["paths"]["atomic_project_root"], backbone_checkpoint, device)
    m3_checkpoint = format_path(cfg["paths"]["m3_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    recognizer = M3AtomicTailOnlineAdapter(
        cfg["paths"]["atomic_project_root"], m3_checkpoint, device,
        max_history=int(cfg["m3"]["max_history"]), task_graph_path=cfg["paths"]["task_graph"],
    )
    cache_root = format_path(cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed, scope=args.scope, stride=cfg["features"]["stride_frames"])
    cache = load_feature_cache(cache_root / f"{args.run}.pt")
    probabilities = infer_run(boundary_model, cache, device)
    segments = run_state_machine(**probabilities, settings=cfg["online"])
    run_index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    frame_table = load_frame_table(run_index[args.run])
    records = []
    for segment_number, segment in enumerate(segments, 1):
        start_row = int(cache["anchor_row_index"][segment["start_index"]])
        end_row = int(cache["anchor_row_index"][segment["end_index"]])
        feature = extract_closed_segment_feature(frame_table["frame_paths"], start_row, end_row, backbone, device, cfg["features"])
        node = recognizer.predict(feature)
        records.append({
            "sample_name": args.run, "detected_segment_no": segment_number, **segment,
            "start_original_frame_idx": int(frame_table["original_frame_idx"][start_row]),
            "end_original_frame_idx": int(frame_table["original_frame_idx"][end_row]),
            "classification_available_at_anchor": int(segment["emitted_at_index"]), **node,
        })
    target = output_root / "online_pipeline" / f"{args.run}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(records)} detected/classified segments to {target}")


if __name__ == "__main__":
    main()
