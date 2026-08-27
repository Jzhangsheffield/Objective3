from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .io import read_json
from .paths import protocol_dir, signal_cache


SUPPLEMENTARY_IDS = tuple(f"S{index}" for index in range(1, 13))


def load_supplementary_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    supplementary = read_json(path)
    base_path = path.parent / supplementary.get("base_phase_a_config", "phase_a.json")
    base = load_config(base_path)
    experiments = supplementary.get("experiments", {})
    missing = set(SUPPLEMENTARY_IDS) - set(experiments)
    extra = set(experiments) - set(SUPPLEMENTARY_IDS)
    if missing or extra:
        raise ValueError(f"Supplementary experiment IDs mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for condition, spec in experiments.items():
        if spec.get("modality") not in {"emg", "imu"}:
            raise ValueError(f"{condition}: unsupported modality {spec.get('modality')}")
        if spec.get("encoder") not in {"resnet10_1d", "dilated_conv1d"}:
            raise ValueError(f"{condition}: unsupported encoder {spec.get('encoder')}")
        if spec.get("task") not in {"m2_node", "direct_node", "direct_tier3"}:
            raise ValueError(f"{condition}: unsupported task {spec.get('task')}")
        if spec.get("task") == "m2_node" and spec.get("upstream") not in experiments:
            raise ValueError(f"{condition}: invalid upstream {spec.get('upstream')}")
    output_root = Path(base["output_root"]) / supplementary.get("output_subdirectory", "supplementary")
    supplementary["config_path"] = str(path)
    supplementary["base"] = base
    supplementary["output_root"] = str(output_root.resolve())
    return supplementary


def validate_supplementary_condition(config: dict[str, Any], condition: str) -> str:
    condition = condition.upper()
    if condition not in config["experiments"]:
        raise ValueError(f"Unknown supplementary experiment: {condition}")
    return condition


def experiment_spec(config: dict[str, Any], condition: str) -> dict[str, Any]:
    return dict(config["experiments"][validate_supplementary_condition(config, condition)])


def supplementary_model_dir(config: dict[str, Any], condition: str, participant: str, seed: int) -> Path:
    return Path(config["output_root"]) / condition / f"{participant}_as_test" / f"seed_{seed}"


def supplementary_feature_dir(config: dict[str, Any], upstream: str, participant: str, seed: int) -> Path:
    return Path(config["output_root"]) / "signal_features" / upstream / f"{participant}_as_test" / f"seed_{seed}"


def supplementary_feature_cache(
    config: dict[str, Any], upstream: str, participant: str, seed: int, split: str
) -> Path:
    if split == "train":
        filename = "train_features.pt"
    elif split == "test":
        filename = "test_features.pt"
    else:
        filename = f"{split}_features.pt"
    return supplementary_feature_dir(config, upstream, participant, seed) / filename


def base_protocol_dir(config: dict[str, Any], participant: str) -> Path:
    return protocol_dir(config["base"], participant)


def base_signal_cache(config: dict[str, Any], participant: str, split: str) -> Path:
    return signal_cache(config["base"], participant, split)


def bilateral_cache_dir(config: dict[str, Any], participant: str) -> Path:
    subdirectory = str(config.get("signal_data", {}).get("cache_subdirectory", "bilateral_signal_cache"))
    return Path(config["base"]["output_root"]) / subdirectory / f"{participant}_as_test"


def training_signal_cache(config: dict[str, Any], participant: str) -> Path:
    if config.get("signal_data", {}).get("scope") == "bilateral":
        return bilateral_cache_dir(config, participant) / "train_bilateral_signals.pt"
    return base_signal_cache(config, participant, "train")


def evaluation_protocols(config: dict[str, Any], participant: str) -> list[dict[str, Any]]:
    signal_data = config.get("signal_data", {})
    if signal_data.get("scope") != "bilateral":
        return [{
            "name": "default",
            "cache": base_signal_cache(config, participant, "test"),
            "manifest_dir": base_protocol_dir(config, participant),
            "feature_split": "test",
            "result_subdirectory": None,
        }]
    root = bilateral_cache_dir(config, participant)
    result = []
    for name, spec in signal_data["test_protocols"].items():
        result.append({
            "name": str(name),
            "cache": root / str(spec["cache_filename"]),
            "manifest_dir": root / "evaluation_protocols" / str(name),
            "feature_split": f"test_{name}",
            "result_subdirectory": str(name),
        })
    return result


def evaluation_protocol(config: dict[str, Any], participant: str, name: str) -> dict[str, Any]:
    matches = [item for item in evaluation_protocols(config, participant) if item["name"] == name]
    if not matches:
        available = [item["name"] for item in evaluation_protocols(config, participant)]
        raise ValueError(f"Unknown evaluation protocol {name}; available={available}")
    return matches[0]


def evaluation_result_dir(model_root: str | Path, protocol: dict[str, Any], base_name: str = "test_results") -> Path:
    result = Path(model_root) / base_name
    return result if protocol["result_subdirectory"] is None else result / protocol["result_subdirectory"]


def signal_channels(modality: str, config: dict[str, Any] | None = None) -> int:
    if config is not None:
        configured = config.get("signal_data", {}).get("channels", {}).get(modality)
        if configured is not None:
            return int(configured)
    if modality == "emg":
        return 8
    if modality == "imu":
        return 6
    raise ValueError(modality)


def signal_length(config: dict[str, Any], modality: str) -> int:
    key = "emg_target_length" if modality == "emg" else "imu_target_length"
    return int(config["base"][key])


def direct_num_classes(task: str) -> int:
    if task == "direct_node":
        return 35
    if task == "direct_tier3":
        return 31
    raise ValueError(f"Not a direct task: {task}")
