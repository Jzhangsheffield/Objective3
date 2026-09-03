from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, fusion_root, inner_feature_root, inner_m2_root,
    load_config, outer_m2_root, output_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the independent B1_IMU_M2 job matrix")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "scripts" / "job_matrix.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    jobs: list[dict[str, str | int]] = []

    def add(stage: str, outer: str, inner: str, seed: str, expected: Path | None, command: list[str]) -> None:
        jobs.append({
            "job_id": len(jobs) + 1, "stage": stage, "outer": outer,
            "inner": inner, "seed": seed,
            "expected_output": "" if expected is None else str(expected.resolve()),
            "command_json": json.dumps(command, ensure_ascii=False),
        })

    add("00_audit", "", "", "", None, [
        "python", str(PACKAGE_ROOT / "tools" / "audit_prerequisites.py"), "--config", config_path,
    ])
    for outer in config["participants"]:
        for inner in config["participants"]:
            if inner == outer:
                continue
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                common = ["--config", config_path, "--outer", outer, "--inner", inner, "--seed", str(seed)]
                add("01_inner_imu_features", outer, inner, str(seed),
                    inner_feature_root(config, outer, inner, seed) / "completed.json", [
                        "python", str(PACKAGE_ROOT / "tools" / "extract_inner_imu_features.py"),
                        *common, "--device", args.device,
                    ])
                add("02_inner_imu_m2", outer, inner, str(seed),
                    inner_m2_root(config, outer, inner, seed) / "completed.json", [
                        "python", str(PACKAGE_ROOT / "tools" / "train_imu_m2.py"),
                        "--scope", "inner", *common, "--device", args.device,
                        "--num-workers", str(args.num_workers),
                    ])
    for outer in config["participants"]:
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            add("03_outer_imu_m2", outer, "", str(seed),
                outer_m2_root(config, outer, seed) / "completed.json", [
                    "python", str(PACKAGE_ROOT / "tools" / "train_imu_m2.py"),
                    "--config", config_path, "--scope", "outer", "--outer", outer,
                    "--seed", str(seed), "--device", args.device,
                    "--num-workers", str(args.num_workers),
                ])
            add("04_fit_b1_imu_m2", outer, "", str(seed),
                fusion_root(config, outer, seed) / "completed.json", [
                    "python", str(PACKAGE_ROOT / "tools" / "fit_fusion.py"),
                    "--config", config_path, "--outer", outer, "--seed", str(seed),
                ])
    add("05_summarize", "", "", "", None, [
        "python", str(PACKAGE_ROOT / "tools" / "summarize.py"), "--config", config_path,
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    metadata = {
        "jobs": len(jobs), "audit": 1, "inner_feature_extraction": 36,
        "inner_imu_m2": 36, "outer_imu_m2": 12, "fusion": 12, "summary": 1,
        "device": args.device, "num_workers": args.num_workers,
    }
    (output.with_suffix(".summary.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(jobs)} jobs to {output}")


if __name__ == "__main__":
    main()
