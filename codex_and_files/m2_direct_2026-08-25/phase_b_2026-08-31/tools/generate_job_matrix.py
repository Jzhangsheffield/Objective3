from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.paths import (
    b0_condition_root, b0_secondary_backbone, crossfit_protocol, crossfit_root,
    experiment_root, expert_root, outer_protocol, primary_backbone, temporal_cache_root,
)


def command(*values: object) -> str:
    return subprocess.list2cmdline([str(value) for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete resumable B0-B5 experiment matrix")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    py, graph = args.python, Path(config["m2_project_root"])
    jobs: list[dict[str, object]] = []

    def add(stage: str, expected: Path, cmd: str, outer: str = "", inner: str = "", seed: object = "") -> None:
        jobs.append({
            "job_id": len(jobs) + 1, "stage": stage, "outer": outer, "inner": inner,
            "seed": seed, "expected_output": str(expected), "command": cmd,
        })

    summary = Path(config["output_root"]) / "crossfit_protocols" / "summary.json"
    add("00_crossfit_protocols", summary, command(
        py, PACKAGE_ROOT / "tools" / "prepare_crossfit_protocols.py", "--config", args.config,
    ))
    for outer in config["participants"]:
        for seed in config["seeds"]:
            add("01_B0", b0_condition_root(config, "A2", outer, seed) / "completed.json", command(
                py, PACKAGE_ROOT / "tools" / "run_b0.py", "--config", args.config,
                "--participant", outer, "--seed", seed, "--device", args.device,
                "--num-workers", args.num_workers, "--execute",
            ), outer=outer, seed=seed)

    for outer in config["participants"]:
        for inner in config["participants"]:
            if inner == outer:
                continue
            protocol = crossfit_protocol(config, outer, inner)
            protocol_parent = protocol.parent
            for seed in config["seeds"]:
                root = crossfit_root(config, outer, inner, seed)
                for name, camera_id in (("cam0", config["primary_camera_id"]),
                                        ("cam1", config["secondary_camera_id"])):
                    backbone = root / f"{name}_backbone"
                    features = root / f"{name}_features"
                    add("02_inner_camera_backbone", backbone / "last.pth", command(
                        py, graph / "tools" / "train_backbone.py", "--dataset-root", config["dataset_root"],
                        "--protocol-root", protocol_parent, "--train-scope", "all_runs",
                        "--output-dir", backbone, "--camera-id", camera_id,
                        "--epochs", config["crossfit"]["camera_backbone_epochs"],
                        "--batch-size", config["camera"]["backbone_batch_size"],
                        "--num-workers", args.num_workers, "--seed", seed, "--device", args.device,
                    ), outer, inner, seed)
                    for split, manifest in (("train", "train.jsonl"), ("test", "test_all.jsonl")):
                        add("03_inner_camera_features", features / f"{split}_all.pt", command(
                            py, graph / "tools" / "extract_features.py", "--dataset-root", config["dataset_root"],
                            "--manifest", protocol / manifest, "--checkpoint", backbone / "last.pth",
                            "--output", features / f"{split}_all.pt", "--camera-id", camera_id,
                            "--batch-size", config["camera"]["backbone_batch_size"],
                            "--num-workers", args.num_workers, "--seed", seed, "--device", args.device,
                        ), outer, inner, seed)
                    camera_output = root / f"{name}_m2"
                    add("04_inner_camera_m2", camera_output / "all_runs" / "m2_direct" / "completed.json", command(
                        py, PACKAGE_ROOT / "tools" / "train_camera_context_expert.py",
                        "--config", args.config, "--protocol-parent", protocol_parent,
                        "--train-cache", features / "train_all.pt", "--test-cache", features / "test_all.pt",
                        "--output-root", camera_output, "--seed", seed, "--device", args.device,
                        "--num-workers", args.num_workers,
                    ), outer, inner, seed)
                imu_cache = root / "imu_cache"
                add("05_inner_imu_cache", imu_cache / "metadata.json", command(
                    py, PACKAGE_ROOT / "tools" / "build_imu_cache.py", "--config", args.config,
                    "--train-manifest", protocol / "train.jsonl", "--test-manifest", protocol / "test_all.jsonl",
                    "--output-dir", imu_cache,
                ), outer, inner, seed)
                imu_output = root / "imu_direct_node"
                add("06_inner_imu_expert", imu_output / "completed.json", command(
                    py, PACKAGE_ROOT / "tools" / "train_imu_expert.py", "--config", args.config,
                    "--cache-dir", imu_cache, "--protocol-dir", protocol, "--output-dir", imu_output,
                    "--seed", seed, "--device", args.device, "--num-workers", args.num_workers,
                ), outer, inner, seed)

    for outer in config["participants"]:
        protocol = outer_protocol(config, outer)
        imu_cache = Path(config["output_root"]) / "outer_imu_cache" / f"{outer}_as_test"
        add("07_outer_imu_cache", imu_cache / "metadata.json", command(
            py, PACKAGE_ROOT / "tools" / "build_imu_cache.py", "--config", args.config,
            "--train-manifest", protocol / "train.jsonl", "--test-manifest", protocol / "test_all.jsonl",
            "--output-dir", imu_cache,
        ), outer=outer)
        for seed in config["seeds"]:
            imu_output = expert_root(config, "outer_experts", outer, seed, "imu_direct_node")
            add("08_outer_imu_expert", imu_output / "completed.json", command(
                py, PACKAGE_ROOT / "tools" / "train_imu_expert.py", "--config", args.config,
                "--cache-dir", imu_cache, "--protocol-dir", protocol, "--output-dir", imu_output,
                "--seed", seed, "--device", args.device, "--num-workers", args.num_workers,
            ), outer=outer, seed=seed)
            token_root = temporal_cache_root(config, outer, seed)
            camera_checkpoints = {
                "cam0": primary_backbone(config, outer, seed),
                "cam1": b0_secondary_backbone(config, outer, seed) / "last.pth",
            }
            for name, camera_id in (("cam0", config["primary_camera_id"]),
                                    ("cam1", config["secondary_camera_id"])):
                for split, manifest in (("train", "train.jsonl"), ("test", "test_all.jsonl")):
                    add("09_outer_camera_tokens", token_root / f"{name}_{split}.pt", command(
                        py, PACKAGE_ROOT / "tools" / "extract_camera_temporal_tokens.py",
                        "--config", args.config, "--manifest", protocol / manifest,
                        "--checkpoint", camera_checkpoints[name], "--camera-id", camera_id,
                        "--output", token_root / f"{name}_{split}.pt", "--seed", seed,
                        "--batch-size", 8, "--num-workers", args.num_workers, "--device", args.device,
                    ), outer=outer, seed=seed)
            for split, manifest, cache_file in (
                ("train", "train.jsonl", "train_imu.pt"), ("test", "test_all.jsonl", "test_imu.pt")
            ):
                add("10_outer_imu_tokens", token_root / f"imu_{split}.pt", command(
                    py, PACKAGE_ROOT / "tools" / "extract_imu_temporal_tokens.py",
                    "--config", args.config, "--cache", imu_cache / cache_file,
                    "--manifest", protocol / manifest, "--checkpoint", imu_output / "last.pth",
                    "--output", token_root / f"imu_{split}.pt", "--device", args.device,
                    "--num-workers", args.num_workers,
                ), outer=outer, seed=seed)
            add("11_B1_B2", experiment_root(config, "B2", outer, seed) / "completed.json", command(
                py, PACKAGE_ROOT / "tools" / "fit_decision_fusion.py", "--config", args.config,
                "--participant", outer, "--seed", seed,
            ), outer=outer, seed=seed)
            for condition in ("B3", "B4", "B5"):
                add(f"12_{condition}", experiment_root(config, condition, outer, seed) / "completed.json", command(
                    py, PACKAGE_ROOT / "tools" / "train_joint_fusion.py", "--config", args.config,
                    "--condition", condition, "--participant", outer, "--seed", seed,
                    "--device", args.device, "--num-workers", args.num_workers,
                ), outer=outer, seed=seed)
    add("13_summary", Path(config["output_root"]) / "summary" / "completeness.json", command(
        py, PACKAGE_ROOT / "tools" / "summarize_phase_b.py", "--config", args.config,
    ))
    matrix = PACKAGE_ROOT / "scripts" / "phase_b_job_matrix.csv"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    with matrix.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    print(f"Wrote {len(jobs)} jobs to {matrix}")


if __name__ == "__main__":
    main()
