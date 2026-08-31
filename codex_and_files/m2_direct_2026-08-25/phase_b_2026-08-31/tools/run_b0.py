from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.io import read_json, write_json
from phase_b.paths import b0_condition_root, b0_secondary_backbone


def run(command: list[str], execute: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if execute:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the B0 two-camera A1/A2 baseline")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Without this flag only print commands")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    phase_a = Path(config["phase_a_root"])
    template = read_json(PACKAGE_ROOT / "config" / config["b0"]["phase_a_compatible_config"])
    template["dataset_root"] = config["dataset_root"]
    template["m2_project_root"] = config["m2_project_root"]
    template["output_root"] = str(Path(config["output_root"]) / "B0_phase_a")
    runtime_config = Path(config["output_root"]) / "runtime_configs" / "b0_phase_a.json"
    write_json(runtime_config, template)
    common = ["--config", str(runtime_config), "--participant", args.participant,
              "--seed", str(args.seed)]
    signal_cache = Path(config["output_root"]) / "B0_phase_a" / "signal_cache" / f"{args.participant}_as_test"
    if not (signal_cache / "right_signal_stats.json").is_file() or args.overwrite:
        command = [sys.executable, str(phase_a / "tools" / "build_signal_cache.py"),
                   "--config", str(runtime_config), "--participant", args.participant]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.execute)
    else:
        print("SKIP B0 compatibility signal cache", flush=True)
    if not b0_secondary_backbone(config, args.participant, args.seed).joinpath("last.pth").is_file() or args.overwrite:
        command = [sys.executable, str(phase_a / "tools" / "prepare_secondary_camera.py"), *common,
                   "--device", args.device, "--num-workers", str(args.num_workers), "--execute"]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.execute)
    else:
        print("SKIP B0 secondary backbone/features", flush=True)
    a1 = b0_condition_root(config, "A1", args.participant, args.seed)
    if not (a1 / "completed.json").is_file() or args.overwrite:
        command = [sys.executable, str(phase_a / "tools" / "train_condition.py"),
                   "--condition", "A1", *common, "--device", args.device,
                   "--num-workers", str(args.num_workers)]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.execute)
    else:
        print("SKIP B0 A1", flush=True)
    a2 = b0_condition_root(config, "A2", args.participant, args.seed)
    if not (a2 / "completed.json").is_file() or args.overwrite:
        command = [sys.executable, str(phase_a / "tools" / "evaluate_a2_late_fusion.py"), *common]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.execute)
    else:
        print("SKIP B0 A2", flush=True)


if __name__ == "__main__":
    main()
