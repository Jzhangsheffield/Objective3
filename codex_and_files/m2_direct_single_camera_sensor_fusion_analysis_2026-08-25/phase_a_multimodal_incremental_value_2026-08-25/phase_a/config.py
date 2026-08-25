from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_json(path)
    config["config_path"] = str(path)
    package_root = path.parents[1]
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = package_root / output_root
    config["output_root"] = str(output_root.resolve())
    for key in ("dataset_root", "m2_project_root"):
        config[key] = str(Path(config[key]).resolve())
    return config


def validate_condition(condition: str) -> str:
    condition = condition.upper()
    if condition not in {f"A{i}" for i in range(8)}:
        raise ValueError(f"Unknown Phase A condition: {condition}")
    return condition
