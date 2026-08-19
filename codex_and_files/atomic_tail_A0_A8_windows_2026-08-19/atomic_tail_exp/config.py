from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_IDS = tuple(f"A{i}" for i in range(9))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    config["_config_path"] = str(config_path)
    config["_package_root"] = str(config_path.parents[1])
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in ("paths", "grid", "model", "training", "augmentation", "experiments") if key not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")
    if tuple(config["experiments"].keys()) != EXPERIMENT_IDS:
        raise ValueError(f"experiments must be ordered exactly as {EXPERIMENT_IDS}")
    for experiment_id, experiment in config["experiments"].items():
        dependency = experiment.get("warm_start")
        if dependency is not None and dependency not in EXPERIMENT_IDS:
            raise ValueError(f"{experiment_id}.warm_start is invalid: {dependency}")
        if experiment.get("position_mode") not in {"presented", "actual_recency"}:
            raise ValueError(f"{experiment_id}.position_mode is invalid")
        if experiment.get("sampling") not in {"none", "uniform", "plausibility_weighted"}:
            raise ValueError(f"{experiment_id}.sampling is invalid")


def parse_csv_values(raw: str | None, default: Iterable[Any], cast=str) -> list[Any]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [cast(value) for value in values]


def normalize_experiment_ids(raw: str | None, config: dict[str, Any]) -> list[str]:
    selected = parse_csv_values(raw, config["experiments"].keys(), str)
    selected = [value.upper() for value in selected]
    unknown = sorted(set(selected) - set(EXPERIMENT_IDS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; valid IDs are A0-A8")
    ordered = [item for item in EXPERIMENT_IDS if item in selected]
    if config["grid"].get("auto_run_dependencies", True):
        dependencies = {
            config["experiments"][item].get("warm_start")
            for item in ordered
            if config["experiments"][item].get("warm_start")
        }
        ordered = [item for item in EXPERIMENT_IDS if item in set(ordered) | dependencies]
    return ordered


def _format_until_stable(template: str, values: dict[str, Any]) -> str:
    result = str(template)
    for _ in range(4):
        previous = result
        result = result.format_map(values)
        if result == previous:
            return result
    return result


def resolved_paths(
    config: dict[str, Any], participant: str, seed: int, scope: str
) -> dict[str, Path | str]:
    values: dict[str, Any] = {
        "package_root": config["_package_root"],
        "participant": participant,
        "seed": int(seed),
        "scope": scope,
        "cache_scope": scope,
        "camera_id": config["grid"]["camera_id"],
    }
    for key, template in config["paths"].items():
        values[key] = _format_until_stable(str(template), values)
    result: dict[str, Path | str] = {}
    for key in config["paths"]:
        value = values[key]
        result[key] = value if key == "python_executable" else Path(str(value)).resolve()
    return result


def run_output_dir(
    config: dict[str, Any], experiment_id: str, participant: str, seed: int, scope: str
) -> Path:
    paths = resolved_paths(config, participant, seed, scope)
    return Path(paths["output_root"]) / experiment_id / scope / f"{participant}_as_test" / f"seed_{seed}"


def run_spec(
    config: dict[str, Any], experiment_id: str, participant: str, seed: int, scope: str
) -> dict[str, Any]:
    spec = deepcopy(config["experiments"][experiment_id])
    spec.update(
        {
            "experiment_id": experiment_id,
            "participant": participant,
            "seed": int(seed),
            "scope": scope,
            "paths": {key: str(value) for key, value in resolved_paths(config, participant, seed, scope).items()},
            "output_dir": str(run_output_dir(config, experiment_id, participant, seed, scope)),
        }
    )
    dependency = spec.get("warm_start")
    reuse_shared_a0 = bool(config["training"].get("reuse_shared_a0_checkpoint", False))
    spec["shared_a0_checkpoint"] = str(spec["paths"]["shared_a0_checkpoint"])
    spec["reuse_shared_a0_checkpoint"] = reuse_shared_a0
    if dependency == "A0" and reuse_shared_a0:
        spec["warm_start_checkpoint"] = spec["shared_a0_checkpoint"]
    else:
        spec["warm_start_checkpoint"] = (
            str(run_output_dir(config, dependency, participant, seed, scope) / "last.pth")
            if dependency
            else None
        )
    return spec
