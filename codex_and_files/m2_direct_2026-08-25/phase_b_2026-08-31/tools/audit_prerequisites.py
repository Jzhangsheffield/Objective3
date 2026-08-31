from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.io import read_jsonl, write_json
from phase_b.paths import (
    a0_checkpoint,
    a0_probabilities,
    b0_condition_root,
    b0_secondary_backbone,
    outer_protocol,
    primary_global_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all Phase B inputs without modifying upstream data")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--load-tensors", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        import torch
        torch_runtime = {
            "available": True, "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError as exc:
        torch_runtime = {"available": False, "error": str(exc)}
    report: dict = {
        "config": config["config_path"],
        "dataset_root": config["dataset_root"],
        "dataset_root_exists": Path(config["dataset_root"]).is_dir(),
        "m2_project_root_exists": Path(config["m2_project_root"]).is_dir(),
        "phase_a_root_exists": Path(config["phase_a_root"]).is_dir(),
        "torch_runtime": torch_runtime,
        "folds": [],
        "blocking": [],
        "generated_upstream_required": [],
    }
    if not report["dataset_root_exists"]:
        report["blocking"].append("dataset_root_missing_for_generation_and_training")
    if not torch_runtime["available"]:
        report["blocking"].append("pytorch_runtime_missing")
    for participant in config["participants"]:
        protocol = outer_protocol(config, participant)
        protocol_files = {name: (protocol / f"{name}.jsonl").is_file() for name in (
            "train", "test_all", "test_normal", "test_fault"
        )}
        counts = {}
        if all(protocol_files.values()):
            counts = {name: len(read_jsonl(protocol / f"{name}.jsonl")) for name in protocol_files}
        for seed in config["seeds"]:
            entry = {
                "participant": participant,
                "seed": seed,
                "protocol_files": protocol_files,
                "protocol_counts": counts,
                "a0_checkpoint": a0_checkpoint(config, participant, seed).is_file(),
                "a0_test_all_probabilities": a0_probabilities(config, participant, seed, "test_all").is_file(),
                "primary_train_cache": primary_global_cache(config, participant, seed, "train").is_file(),
                "primary_test_cache": primary_global_cache(config, participant, seed, "test").is_file(),
                "b0_secondary_checkpoint": b0_secondary_backbone(config, participant, seed).joinpath("last.pth").is_file(),
                "b0_a1_complete": b0_condition_root(config, "A1", participant, seed).joinpath("completed.json").is_file(),
                "b0_a2_complete": b0_condition_root(config, "A2", participant, seed).joinpath("completed.json").is_file(),
            }
            required = (
                all(protocol_files.values())
                and entry["a0_checkpoint"]
                and entry["a0_test_all_probabilities"]
                and entry["primary_train_cache"]
                and entry["primary_test_cache"]
            )
            entry["fixed_upstream_ready"] = required
            if not required:
                report["blocking"].append(f"fixed_upstream_missing:{participant}:seed_{seed}")
            if not entry["b0_a1_complete"]:
                report["generated_upstream_required"].append(f"B0_A1:{participant}:seed_{seed}")
            report["folds"].append(entry)
    if args.load_tensors:
        try:
            import torch

            checked = []
            for entry in report["folds"]:
                participant, seed = entry["participant"], entry["seed"]
                for name, path in (
                    ("primary_train", primary_global_cache(config, participant, seed, "train")),
                    ("primary_test", primary_global_cache(config, participant, seed, "test")),
                    ("a0_probabilities", a0_probabilities(config, participant, seed, "test_all")),
                ):
                    value = torch.load(path, map_location="cpu", weights_only=False)
                    tensor = value.get("features", value.get("node_probabilities"))
                    checked.append({
                        "participant": participant,
                        "seed": seed,
                        "name": name,
                        "shape": list(tensor.shape),
                        "finite": bool(torch.isfinite(tensor).all()),
                    })
            report["tensor_checks"] = checked
        except Exception as exc:
            report["blocking"].append(f"tensor_audit_failed:{type(exc).__name__}:{exc}")
    report["fixed_upstream_complete"] = not any(
        value.startswith("fixed_upstream_missing") for value in report["blocking"]
    )
    report["formal_run_ready"] = (
        report["fixed_upstream_complete"] and report["dataset_root_exists"]
        and torch_runtime["available"]
    )
    output = Path(config["output_root"]) / "audit" / "prerequisite_audit.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
