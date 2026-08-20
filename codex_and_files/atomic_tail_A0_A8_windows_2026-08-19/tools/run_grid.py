from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.config import (
    load_config,
    normalize_experiment_ids,
    parse_csv_values,
    run_spec,
)


def validate_run_inputs(spec: dict) -> list[str]:
    missing = []
    paths = spec["paths"]
    for key in ("task_graph", "relation_matrix", "train_cache", "test_cache"):
        if not Path(paths[key]).is_file():
            missing.append(f"{key}: {paths[key]}")
    protocol_root = Path(paths["protocol_root"]) / spec["scope"]
    for name in ("train", "test_normal", "test_fault", "test_all"):
        path = protocol_root / f"{name}.jsonl"
        if not path.is_file():
            missing.append(f"manifest: {path}")
    if spec.get("reuse_shared_a0_checkpoint") and (
        spec["experiment_id"] == "A0" or spec.get("warm_start") == "A0"
    ):
        checkpoint = Path(spec["shared_a0_checkpoint"])
        if not checkpoint.is_file():
            missing.append(f"shared_a0_checkpoint: {checkpoint}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a selectable atomic-tail experiment grid, including DualPos experiments.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--experiments", default=None, help="Comma-separated subset, e.g. A3-DualPos,A4-DualPos")
    parser.add_argument("--participants", default=None, help="Comma-separated outer test folds")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--scopes", default=None, help="Comma-separated train scopes")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print jobs without importing PyTorch")
    parser.add_argument("--validate-only", action="store_true", help="Validate all selected input paths, then exit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = normalize_experiment_ids(args.experiments, config)
    participants = parse_csv_values(args.participants, config["grid"]["participants"], str)
    seeds = parse_csv_values(args.seeds, config["grid"]["seeds"], int)
    scopes = parse_csv_values(args.scopes, config["grid"]["train_scopes"], str)
    jobs = [run_spec(config, experiment, participant, seed, scope) for experiment in experiments for scope in scopes for participant in participants for seed in seeds]
    errors = []
    for spec in jobs:
        missing = validate_run_inputs(spec)
        if missing:
            errors.extend(f"{spec['experiment_id']}/{spec['scope']}/{spec['participant']}/seed_{spec['seed']}: {item}" for item in missing)
    plan = {
        "job_count": len(jobs),
        "experiments": experiments,
        "participants": participants,
        "seeds": seeds,
        "scopes": scopes,
        "auto_added_dependencies": args.experiments is not None and any(
            item.casefold() not in [part.strip().casefold() for part in args.experiments.split(",")]
            for item in experiments
        ),
        "deferred_selected": [
            item for item in experiments
            if config["experiments"][item].get("status") == "deferred"
        ],
        "input_errors": errors,
        "jobs": [
            {
                "experiment": spec["experiment_id"],
                "participant": spec["participant"],
                "seed": spec["seed"],
                "scope": spec["scope"],
                "output": spec["output_dir"],
                "warm_start": spec["warm_start_checkpoint"],
                "reuses_shared_a0_checkpoint": bool(spec.get("reuse_shared_a0_checkpoint"))
                and (spec["experiment_id"] == "A0" or spec.get("warm_start") == "A0"),
                "shared_a0_checkpoint": spec.get("shared_a0_checkpoint"),
            }
            for spec in jobs
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if errors:
        print("\nInput validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    if args.dry_run or args.validate_only:
        return 0

    from atomic_tail_exp.training import run_training

    failed = []
    for index, spec in enumerate(jobs, 1):
        marker = Path(spec["output_dir"]) / "completed.json"
        if marker.is_file() and config["grid"].get("skip_completed", True) and not args.overwrite:
            print(f"[SKIP {index}/{len(jobs)}] {spec['experiment_id']} {spec['participant']} seed={spec['seed']} {spec['scope']}")
            continue
        print(f"[RUN {index}/{len(jobs)}] {spec['experiment_id']} {spec['participant']} seed={spec['seed']} {spec['scope']}")
        try:
            run_training(config, spec, overwrite=args.overwrite)
        except Exception as error:
            failed.append({"spec": spec, "error": repr(error)})
            traceback.print_exc()
            if not args.continue_on_error:
                break
    if failed:
        print(json.dumps({"failed_jobs": failed}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print("Selected atomic-tail experiment grid completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
