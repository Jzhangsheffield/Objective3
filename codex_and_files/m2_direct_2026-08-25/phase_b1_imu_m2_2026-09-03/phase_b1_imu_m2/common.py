from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "experiment.json"


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported experiment config schema")
    config["_config_path"] = str(config_path)
    config["_package_root"] = str(PACKAGE_ROOT)
    config["_phase_b_root"] = str(_resolve(PACKAGE_ROOT, config["phase_b_root"]))
    config["_output_root"] = str(_resolve(PACKAGE_ROOT, config["output_root"]))
    phase_b_root = Path(config["_phase_b_root"])
    phase_b_config_path = phase_b_root / "config" / "phase_b.json"
    phase_b_config = read_json(phase_b_config_path)
    for key in ("dataset_root", "m2_project_root", "phase_a_root", "output_root"):
        phase_b_config[key] = str(_resolve(phase_b_root, phase_b_config[key]))
    phase_b_config["config_path"] = str(phase_b_config_path)
    phase_b_config["package_root"] = str(phase_b_root)
    config["_phase_b_config"] = phase_b_config
    return config


def phase_b_output_root(config: dict[str, Any]) -> Path:
    phase_b_root = Path(config["_phase_b_root"])
    value = Path(config["_phase_b_config"]["output_root"])
    return value.resolve() if value.is_absolute() else (phase_b_root / value).resolve()


def m2_project_root(config: dict[str, Any]) -> Path:
    phase_b_root = Path(config["_phase_b_root"])
    value = Path(config["_phase_b_config"]["m2_project_root"])
    return value.resolve() if value.is_absolute() else (phase_b_root / value).resolve()


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["_output_root"])


def outer_protocol(config: dict[str, Any], outer: str) -> Path:
    camera_id = config["_phase_b_config"]["primary_camera_id"]
    return m2_project_root(config) / "outputs" / f"{outer}_as_test" / f"cam_{camera_id}" / "protocols" / "all_runs"


def inner_protocol(config: dict[str, Any], outer: str, inner: str) -> Path:
    return phase_b_output_root(config) / "crossfit_protocols" / f"outer_{outer}" / f"heldout_{inner}" / "all_runs"


def inner_phase_b_root(config: dict[str, Any], outer: str, inner: str, seed: int) -> Path:
    return phase_b_output_root(config) / "crossfit" / f"outer_{outer}" / f"heldout_{inner}" / f"seed_{seed}"


def inner_feature_root(config: dict[str, Any], outer: str, inner: str, seed: int) -> Path:
    return output_root(config) / "imu_features" / "inner" / f"outer_{outer}" / f"heldout_{inner}" / f"seed_{seed}"


def inner_m2_root(config: dict[str, Any], outer: str, inner: str, seed: int) -> Path:
    return output_root(config) / "imu_m2" / "inner" / f"outer_{outer}" / f"heldout_{inner}" / f"seed_{seed}"


def outer_m2_root(config: dict[str, Any], outer: str, seed: int) -> Path:
    return output_root(config) / "imu_m2" / "outer" / f"{outer}_as_test" / f"seed_{seed}"


def fusion_root(config: dict[str, Any], outer: str, seed: int) -> Path:
    return output_root(config) / "B1_IMU_M2" / f"{outer}_as_test" / f"seed_{seed}"


def outer_imu_token_cache(config: dict[str, Any], outer: str, seed: int, split: str) -> Path:
    return phase_b_output_root(config) / "temporal_caches" / f"{outer}_as_test" / f"seed_{seed}" / f"imu_{split}.pt"


def add_phase_b_to_path(config: dict[str, Any]) -> None:
    value = config["_phase_b_root"]
    if value not in sys.path:
        sys.path.insert(0, value)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def safe_torch_load(path: str | Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
