from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .utils import read_json


REQUIRED_SECTIONS = {"paths", "data", "features", "model", "training", "online", "evaluation"}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
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
        cfg = _deep_update(load_config(parent), cfg)
    missing = REQUIRED_SECTIONS - set(cfg)
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
    return copy.deepcopy(cfg)


def format_path(template: str, **values: Any) -> Path:
    return Path(template.format(**values))
