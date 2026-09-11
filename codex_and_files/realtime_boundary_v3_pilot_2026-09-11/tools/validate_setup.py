from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_frame_table, load_run_index
from boundary_experiment.config import format_path, load_config
from boundary_experiment.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation of annotations, frames, protocols and checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--deep", action="store_true", help="Check every annotated frame; default checks endpoints")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    problems: list[str] = []
    action_frames = background_frames = 0
    for info in run_index.values():
        try:
            table = load_frame_table(info) if args.deep else None
            if table:
                action_frames += int(table["state"].sum())
                background_frames += int(len(table["state"]) - table["state"].sum())
            else:
                if not info.frame_annotation.is_file() or not info.camera_dir.is_dir():
                    problems.append(f"Missing annotation/camera directory for {info.sample_name}")
        except Exception as error:
            problems.append(f"{info.sample_name}: {error}")
    checkpoint_count = 0
    m3_checkpoint_count = 0
    for heldout in cfg["data"]["participants"]:
        for seed in cfg["data"]["seeds"]:
            for scope in cfg["data"]["train_scopes"]:
                path = format_path(cfg["paths"]["backbone_checkpoint_template"], heldout=heldout, seed=seed, scope=scope)
                if path.is_file(): checkpoint_count += 1
                else: problems.append(f"Missing backbone checkpoint: {path}")
                m3_path = format_path(cfg["paths"]["m3_checkpoint_template"], heldout=heldout, seed=seed, scope=scope)
                if m3_path.is_file(): m3_checkpoint_count += 1
                else: problems.append(f"Missing M3 checkpoint: {m3_path}")
    report = {
        "status": "ok" if not problems else "failed", "runs": len(run_index),
        "deep": args.deep, "action_frames": action_frames, "background_frames": background_frames,
        "backbone_checkpoints": checkpoint_count, "m3_checkpoints": m3_checkpoint_count,
        "problems": problems,
    }
    validation_root = Path(cfg["paths"].get("validation_root", Path(cfg["paths"]["experiment_root"]) / "validation"))
    target = validation_root / "setup_validation.json"
    write_json(target, report)
    print(report)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
