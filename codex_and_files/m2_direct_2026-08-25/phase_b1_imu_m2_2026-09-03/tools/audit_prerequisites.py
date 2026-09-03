from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, inner_phase_b_root, inner_protocol, load_config,
    m2_project_root, outer_imu_token_cache, outer_protocol, output_root,
    phase_b_output_root, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable Phase B prerequisites for B1_IMU_M2")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    missing: list[str] = []
    checked: list[str] = []

    def require(path: Path) -> None:
        checked.append(str(path))
        if not path.is_file():
            missing.append(str(path))

    for outer in config["participants"]:
        for inner in config["participants"]:
            if inner == outer:
                continue
            for seed in config["seeds"]:
                base = inner_phase_b_root(config, outer, inner, int(seed))
                protocol = inner_protocol(config, outer, inner)
                for path in (
                    base / "imu_cache" / "train_imu.pt",
                    base / "imu_cache" / "test_imu.pt",
                    base / "imu_direct_node" / "last.pth",
                    base / "cam0_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
                    base / "cam1_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
                    protocol / "train.jsonl", protocol / "test_all.jsonl",
                    protocol / "test_normal.jsonl", protocol / "test_fault.jsonl",
                ):
                    require(path)
    for outer in config["participants"]:
        for seed in config["seeds"]:
            protocol = outer_protocol(config, outer)
            paths = [
                outer_imu_token_cache(config, outer, int(seed), "train"),
                outer_imu_token_cache(config, outer, int(seed), "test"),
                protocol / "train.jsonl", protocol / "test_all.jsonl",
                protocol / "test_normal.jsonl", protocol / "test_fault.jsonl",
            ]
            for split in config["evaluation"]["splits"]:
                paths.extend([
                    phase_b_output_root(config) / "B1" / f"{outer}_as_test" / f"seed_{seed}"
                    / "test_results" / f"{split}_metrics.json",
                    phase_b_output_root(config) / "B0_phase_a" / "A2" / f"{outer}_as_test"
                    / f"seed_{seed}" / "test_results" / f"{split}_metrics.json",
                    phase_b_output_root(config) / "outer_experts" / f"{outer}_as_test"
                    / f"seed_{seed}" / "imu_direct_node" / "test_results" / f"{split}_metrics.json",
                    m2_project_root(config) / "outputs" / f"{outer}_as_test"
                    / f"cam_{config['_phase_b_config']['primary_camera_id']}" / f"seed_{seed}"
                    / "history_models" / "direct_head_fusion" / "all_runs" / "m2_direct"
                    / "test_results" / f"{split}_probabilities.pt",
                    phase_b_output_root(config) / "B0_phase_a" / "A1" / f"{outer}_as_test"
                    / f"seed_{seed}" / "test_results" / f"{split}_probabilities.pt",
                ])
            for path in paths:
                require(path)

    payload = {
        "experiment": config["experiment"], "complete": not missing,
        "checked_files": len(checked), "missing_files": missing,
        "phase_b_root": config["_phase_b_root"],
        "phase_b_output_root": str(phase_b_output_root(config)),
        "m2_project_root": str(m2_project_root(config)),
        "extension_output_root": str(output_root(config)),
        "inner_units": 36, "outer_units": 12,
        "original_phase_b_write_required": False,
    }
    destination = output_root(config) / "audit" / "prerequisite_audit.json"
    write_json(destination, payload)
    print(f"Saved: {destination}")
    if missing:
        raise SystemExit(f"Prerequisite audit failed: {len(missing)} missing files")
    print(f"PASS: {len(checked)} immutable Phase B prerequisite files found")


if __name__ == "__main__":
    main()
