from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sequence_disjoint_exp.common import load_package_config, resolve_paths


def parse_values(raw: str | None, defaults: list[Any], cast=str) -> list[Any]:
    if raw is None:
        return list(defaults)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def amp_flag(enabled: bool) -> str:
    return "--amp" if enabled else "--no-amp"


def build_commands(
    config: dict[str, Any], participant: str, seed: int, dataset_root: Path, device: str
) -> dict[str, list[str]]:
    paths = resolve_paths(config, participant, seed)
    scope_root = Path(paths["protocol_root"]) / str(config["grid"]["train_scope"])
    backbone = config["backbone_training"]
    features = config["feature_extraction"]
    backbone_command = [
        sys.executable,
        str(paths["legacy_train_backbone"]),
        "--dataset-root", str(dataset_root),
        "--protocol-root", str(paths["protocol_root"]),
        "--train-scope", str(config["grid"]["train_scope"]),
        "--output-dir", str(paths["backbone_output"]),
        "--camera-id", str(config["grid"]["camera_id"]),
        "--epochs", str(backbone["epochs"]),
        "--batch-size", str(backbone["batch_size"]),
        "--num-workers", str(backbone["num_workers"]),
        "--learning-rate", str(backbone["learning_rate"]),
        "--weight-decay", str(backbone["weight_decay"]),
        "--seed", str(seed),
        "--device", device,
        amp_flag(bool(backbone["amp"])),
    ]
    train_feature_command = [
        sys.executable,
        str(paths["legacy_extract_features"]),
        "--dataset-root", str(dataset_root),
        "--manifest", str(scope_root / str(features["train_manifest"])),
        "--checkpoint", str(paths["backbone_checkpoint"]),
        "--output", str(paths["train_cache"]),
        "--camera-id", str(config["grid"]["camera_id"]),
        "--batch-size", str(features["batch_size"]),
        "--num-workers", str(features["num_workers"]),
        "--seed", str(seed),
        "--device", device,
        amp_flag(bool(features["amp"])),
    ]
    test_feature_command = [
        sys.executable,
        str(paths["legacy_extract_features"]),
        "--dataset-root", str(dataset_root),
        "--manifest", str(scope_root / str(features["test_manifest"])),
        "--checkpoint", str(paths["backbone_checkpoint"]),
        "--output", str(paths["test_cache"]),
        "--completion-marker", str(Path(paths["feature_root"]) / "completed.json"),
        "--camera-id", str(config["grid"]["camera_id"]),
        "--batch-size", str(features["batch_size"]),
        "--num-workers", str(features["num_workers"]),
        "--seed", str(seed),
        "--device", device,
        amp_flag(bool(features["amp"])),
    ]
    return {
        "backbone": backbone_command,
        "extract_train": train_feature_command,
        "extract_test": test_feature_command,
    }


def validate_inputs(
    config: dict[str, Any], participant: str, seed: int, dataset_root: Path
) -> tuple[list[str], list[str]]:
    paths = resolve_paths(config, participant, seed)
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("legacy_train_backbone", "legacy_extract_features"):
        if not Path(paths[key]).is_file():
            errors.append(f"missing {key}: {paths[key]}")
    scope_root = Path(paths["protocol_root"]) / str(config["grid"]["train_scope"])
    for name in ("train.jsonl", "test_normal.jsonl", "test_fault.jsonl", "test_all.jsonl"):
        if not (scope_root / name).is_file():
            errors.append(f"missing manifest: {scope_root / name}")
    if not dataset_root.is_dir():
        warnings.append(f"dataset root is not available in this environment: {dataset_root}")
    return errors, warnings


def run_command(command: list[str], overwrite: bool) -> None:
    actual = list(command)
    if overwrite:
        actual.append("--overwrite")
    subprocess.run(actual, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain sequence-disjoint RGB backbones and extract fresh features")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participants", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--stage", choices=["backbone", "features", "all"], default="all")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    config = load_package_config(args.config)
    participants = parse_values(args.participants, list(config["grid"]["participants"]), str)
    seeds = parse_values(args.seeds, list(config["grid"]["seeds"]), int)
    first_paths = resolve_paths(config, participants[0], seeds[0])
    dataset_root = Path(args.dataset_root or first_paths["dataset_root"]).resolve()
    device = str(args.device or config["backbone_training"]["device"])
    jobs = []
    errors: list[str] = []
    warnings: list[str] = []
    for participant in participants:
        for seed in seeds:
            paths = resolve_paths(config, participant, seed)
            commands = build_commands(config, participant, seed, dataset_root, device)
            job_errors, job_warnings = validate_inputs(config, participant, seed, dataset_root)
            errors.extend(f"{participant}/seed_{seed}: {item}" for item in job_errors)
            warnings.extend(f"{participant}/seed_{seed}: {item}" for item in job_warnings)
            jobs.append({
                "participant": participant,
                "seed": seed,
                "backbone_checkpoint": str(paths["backbone_checkpoint"]),
                "train_cache": str(paths["train_cache"]),
                "test_cache": str(paths["test_cache"]),
                "commands": commands,
            })
    plan = {
        "fold_seed_jobs": len(jobs),
        "stage": args.stage,
        "participants": participants,
        "seeds": seeds,
        "dataset_root": str(dataset_root),
        "backbone_retrained_from_scratch": True,
        "feature_caches_reused_from_old_experiment": False,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "jobs": jobs,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if errors:
        return 2
    if args.dry_run or args.validate_only:
        return 0
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    failures = []
    for index, job in enumerate(jobs, 1):
        participant = str(job["participant"])
        seed = int(job["seed"])
        paths = resolve_paths(config, participant, seed)
        try:
            if args.stage in {"backbone", "all"}:
                marker = Path(paths["backbone_output"]) / "completed.json"
                checkpoint = Path(paths["backbone_checkpoint"])
                if marker.is_file() and not checkpoint.is_file():
                    raise RuntimeError(f"Backbone marker exists but checkpoint is missing: {checkpoint}")
                if marker.is_file() and checkpoint.is_file() and not args.overwrite:
                    print(f"[SKIP backbone {index}/{len(jobs)}] {participant} seed={seed}")
                else:
                    print(f"[RUN backbone {index}/{len(jobs)}] {participant} seed={seed}")
                    run_command(job["commands"]["backbone"], args.overwrite)
            if args.stage in {"features", "all"}:
                marker = Path(paths["feature_root"]) / "completed.json"
                train_cache = Path(paths["train_cache"])
                test_cache = Path(paths["test_cache"])
                if marker.is_file() and (not train_cache.is_file() or not test_cache.is_file()):
                    raise RuntimeError(
                        f"Feature marker exists but cache is incomplete: train={train_cache}, test={test_cache}"
                    )
                if marker.is_file() and train_cache.is_file() and test_cache.is_file() and not args.overwrite:
                    print(f"[SKIP features {index}/{len(jobs)}] {participant} seed={seed}")
                else:
                    if not Path(paths["backbone_checkpoint"]).is_file():
                        raise FileNotFoundError(f"Missing new backbone checkpoint: {paths['backbone_checkpoint']}")
                    print(f"[RUN train features {index}/{len(jobs)}] {participant} seed={seed}")
                    run_command(job["commands"]["extract_train"], args.overwrite)
                    print(f"[RUN test features {index}/{len(jobs)}] {participant} seed={seed}")
                    run_command(job["commands"]["extract_test"], args.overwrite)
        except Exception as error:
            failures.append({"participant": participant, "seed": seed, "error": repr(error)})
            if not args.continue_on_error:
                break
    if failures:
        print(json.dumps({"failed_jobs": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
