from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sequence_disjoint_exp.common import load_package_config, read_json, resolve_paths


def parse_values(raw: str | None, defaults: list[Any], cast=str) -> list[Any]:
    if raw is None:
        return list(defaults)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def experiment_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in config["experiments"]}


def load_training_config(package_config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    first_participant = str(package_config["grid"]["participants"][0])
    first_seed = int(package_config["grid"]["seeds"][0])
    resolved = resolve_paths(package_config, first_participant, first_seed)
    legacy_root = Path(resolved["legacy_atomic_package_root"])
    legacy_config_path = Path(resolved["legacy_atomic_config"])
    legacy_config = read_json(legacy_config_path)
    legacy_config["training"].update(package_config["training_overrides"])
    legacy_config["training"]["reuse_shared_a0_checkpoint"] = False
    legacy_config["grid"]["test_splits"] = list(package_config["grid"]["test_splits"])
    legacy_config["grid"]["train_scopes"] = [package_config["grid"]["train_scope"]]
    return legacy_config, legacy_root


def build_spec(
    package_config: dict[str, Any],
    training_config: dict[str, Any],
    experiment_definition: dict[str, Any],
    participant: str,
    seed: int,
) -> dict[str, Any]:
    experiment_id = str(experiment_definition["id"])
    base_experiment = str(experiment_definition["base_experiment"])
    spec = copy.deepcopy(training_config["experiments"][base_experiment])
    refresh_interval = experiment_definition.get("refresh_interval")
    if refresh_interval is not None:
        spec["refresh_interval"] = refresh_interval
    paths = resolve_paths(package_config, participant, seed)
    output_dir = (
        Path(paths["history_output_root"])
        / experiment_id
        / str(package_config["grid"]["train_scope"])
        / f"{participant}_as_test"
        / f"seed_{seed}"
    )
    shared_checkpoint = output_dir / "_unused_shared_a0_checkpoint.pth"
    spec.update({
        "experiment_id": experiment_id,
        "source_experiment_id": base_experiment,
        "description": experiment_definition["description"],
        "participant": participant,
        "seed": int(seed),
        "scope": str(package_config["grid"]["train_scope"]),
        "paths": {key: str(value) for key, value in paths.items()},
        "output_dir": str(output_dir),
        "warm_start": None,
        "warm_start_checkpoint": None,
        "reuse_shared_a0_checkpoint": False,
        "shared_a0_checkpoint": str(shared_checkpoint),
        "raw_training_sequence_policy": "sequence_disjoint_real_runs",
        "augmentation_may_match_test_order": bool(
            package_config["sequence_isolation"]["allow_augmentation_to_match_test_order"]
        ),
        "test_guided_sampling": False,
    })
    return spec


def validate_spec(spec: dict[str, Any], package_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    missing_upstream: list[str] = []
    paths = spec["paths"]
    for key in ("task_graph", "relation_matrix"):
        if not Path(paths[key]).is_file():
            missing.append(f"{key}: {paths[key]}")
    for key in ("train_cache", "test_cache"):
        if not Path(paths[key]).is_file():
            missing_upstream.append(f"{key}: {paths[key]}")
    protocol_scope = Path(paths["protocol_root"]) / spec["scope"]
    for name in ("train", *package_config["grid"]["test_splits"]):
        path = protocol_scope / f"{name}.jsonl"
        if not path.is_file():
            missing.append(f"manifest: {path}")
    return missing, missing_upstream


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sequence-disjoint M2/A1-Legacy/A3-DualPos grid")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--experiments", default=None, help="Comma-separated package experiment IDs")
    parser.add_argument("--participants", default=None, help="Comma-separated held-out participants")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    package_config = load_package_config(args.config)
    training_config, legacy_root = load_training_config(package_config)
    definitions = experiment_map(package_config)
    selected_ids = parse_values(
        args.experiments, list(package_config["grid"]["default_experiments"]), str
    )
    unknown = [value for value in selected_ids if value not in definitions]
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {unknown}; valid={list(definitions)}")
    participants = parse_values(args.participants, list(package_config["grid"]["participants"]), str)
    seeds = parse_values(args.seeds, list(package_config["grid"]["seeds"]), int)
    jobs = [
        build_spec(package_config, training_config, definitions[experiment_id], participant, seed)
        for experiment_id in selected_ids
        for participant in participants
        for seed in seeds
    ]
    errors = []
    upstream_missing = []
    for spec in jobs:
        spec_errors, spec_upstream = validate_spec(spec, package_config)
        errors.extend(
            f"{spec['experiment_id']}/{spec['participant']}/seed_{spec['seed']}: {item}"
            for item in spec_errors
        )
        upstream_missing.extend(
            f"{spec['participant']}/seed_{spec['seed']}: {item}"
            for item in spec_upstream
        )
    plan = {
        "job_count": len(jobs),
        "experiments": selected_ids,
        "participants": participants,
        "seeds": seeds,
        "scope": package_config["grid"]["train_scope"],
        "M2_uses_shuffle": False,
        "augmentation_may_naturally_match_test_order": True,
        "test_guided_sampling": False,
        "position_embedding_reinitialized_on_refresh": False,
        "optimizer_state_reinitialized_on_refresh": False,
        "legacy_training_code": str(legacy_root),
        "input_errors": errors,
        "missing_upstream_feature_caches": sorted(set(upstream_missing)),
        "jobs": [
            {
                "experiment": spec["experiment_id"],
                "source_experiment": spec["source_experiment_id"],
                "participant": spec["participant"],
                "seed": spec["seed"],
                "refresh_interval": spec.get("refresh_interval"),
                "train_view": spec["train_view"],
                "position_mode": spec["position_mode"],
                "active_tail_only": spec["active_tail_only"],
                "refresh_replaces_previous_view": spec["experiment_id"].endswith("Every10-Replace")
                or spec["experiment_id"].endswith("Every10"),
                "output": spec["output_dir"],
            }
            for spec in jobs
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if errors or (upstream_missing and not args.dry_run):
        return 2
    if args.dry_run or args.validate_only:
        return 0

    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from atomic_tail_exp.training import run_training

    failed = []
    for index, spec in enumerate(jobs, 1):
        marker = Path(spec["output_dir"]) / "completed.json"
        if marker.is_file() and package_config["grid"].get("skip_completed", True) and not args.overwrite:
            print(f"[SKIP {index}/{len(jobs)}] {spec['experiment_id']} {spec['participant']} seed={spec['seed']}")
            continue
        print(f"[RUN {index}/{len(jobs)}] {spec['experiment_id']} {spec['participant']} seed={spec['seed']}")
        try:
            run_training(training_config, spec, overwrite=args.overwrite)
        except Exception as error:
            failed.append({"experiment": spec["experiment_id"], "participant": spec["participant"], "seed": spec["seed"], "error": repr(error)})
            traceback.print_exc()
            if not args.continue_on_error:
                break
    if failed:
        print(json.dumps({"failed_jobs": failed}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
