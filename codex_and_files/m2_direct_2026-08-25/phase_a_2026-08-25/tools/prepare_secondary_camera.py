from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.config import load_config
from phase_a.paths import protocol_dir, secondary_backbone_dir, secondary_feature_cache


def run(command: list[str], execute: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if execute:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/extract the independent second-camera RGB backbone")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only print the commands")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    graph_root = Path(config["m2_project_root"])
    protocol = protocol_dir(config, args.participant)
    backbone = secondary_backbone_dir(config, args.participant, args.seed)
    common = ["--device", args.device, "--num-workers", str(args.num_workers)]
    overwrite = ["--overwrite"] if args.overwrite else []
    train_command = [
        sys.executable, str(graph_root / "tools" / "train_backbone.py"),
        "--dataset-root", config["dataset_root"], "--protocol-root", str(protocol.parent),
        "--train-scope", config["train_scope"], "--output-dir", str(backbone),
        "--camera-id", config["secondary_camera_id"], "--seed", str(args.seed),
        "--epochs", "100", "--batch-size", "16", *common, *overwrite,
    ]
    if args.resume and (backbone / "last.pth").is_file():
        print(f"SKIP completed backbone: {backbone / 'last.pth'}", flush=True)
    else:
        run(train_command, args.execute)
    for split, manifest_name in (("train", "train.jsonl"), ("test", "test_all.jsonl")):
        output = secondary_feature_cache(config, args.participant, args.seed, split)
        if args.resume and output.is_file():
            print(f"SKIP completed feature cache: {output}", flush=True)
            continue
        extraction = [
            sys.executable, str(graph_root / "tools" / "extract_features.py"),
            "--dataset-root", config["dataset_root"], "--manifest", str(protocol / manifest_name),
            "--checkpoint", str(backbone / "last.pth"), "--output", str(output),
            "--camera-id", config["secondary_camera_id"], "--seed", str(args.seed),
            "--batch-size", "16", *common, *overwrite,
        ]
        run(extraction, args.execute)


if __name__ == "__main__":
    main()
