from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


EXPERIMENTS = {f"B{i}" for i in range(6)}


def _resolve(base: Path, value: str) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_json(path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Phase B config schema_version must be 1")
    package_root = path.parents[1]
    config["config_path"] = str(path)
    config["package_root"] = str(package_root)
    for key in ("dataset_root", "m2_project_root", "phase_a_root", "output_root"):
        config[key] = _resolve(package_root, str(config[key]))
    if sorted(config["participants"]) != sorted("ADJM"):
        raise ValueError("The preregistered outer participants must be A/D/J/M")
    if sorted(int(value) for value in config["seeds"]) != [1, 2, 42]:
        raise ValueError("The preregistered seeds must be 1/2/42")
    unknown = set(config["experiments"]) - EXPERIMENTS
    if unknown:
        raise ValueError(f"Unknown Phase B experiments: {sorted(unknown)}")
    if int(config["b2"]["max_optimizer_steps"]) <= 0:
        raise ValueError("B2 max_optimizer_steps must be positive")
    joint = config["joint_fusion"]
    if int(joint["batch_size"]) != int(joint["effective_batch_size"]):
        raise ValueError("B3-B5 direct batch training requires batch_size == effective_batch_size")
    if int(joint.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("B3-B5 gradient_accumulation_steps must be 1")
    return config


def validate_experiment(value: str) -> str:
    value = value.upper()
    if value not in EXPERIMENTS:
        raise ValueError(f"Unknown Phase B experiment: {value}")
    return value
