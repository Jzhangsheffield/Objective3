from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .utils import read_json


REQUIRED = {"paths", "data", "sensor", "model", "training", "online", "evaluation"}


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if "%SBE_" in result:
            raise EnvironmentError(
                f"Unresolved Windows variable in {value!r}. Call config_windows.bat first."
            )
        return result
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = read_json(path)
    if "extends" in cfg:
        parent = Path(cfg.pop("extends"))
        if not parent.is_absolute():
            parent = path.parent / parent
        cfg = _merge(load_config(parent), cfg)
    missing = REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
    return copy.deepcopy(_expand(cfg))


def format_path(template: str, **values: Any) -> Path:
    return Path(template.format(**values))
