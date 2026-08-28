from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .io import read_jsonl, write_json


def safe_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resample_lc_to_cl(value: torch.Tensor, channels: int, length: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim != 2 or value.shape[1] != channels:
        raise ValueError(f"{name} must be [L,{channels}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return F.interpolate(
        value.transpose(0, 1).unsqueeze(0), size=length, mode="linear", align_corners=False
    ).squeeze(0).contiguous()


def _load_right_signals(dataset_root: Path, row: dict[str, Any], emg_len: int, imu_len: int):
    loaded = safe_load(dataset_root / row["mindrove"])
    if not isinstance(loaded, dict):
        raise TypeError(f"MindRove file for {row['sample_name']} is not a dict")
    emg = resample_lc_to_cl(loaded["right_emg"], 8, emg_len, "right_emg")
    acc = torch.as_tensor(loaded["right_acc"]).float()
    gyro = torch.as_tensor(loaded["right_gyro"]).float()
    if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"Invalid right IMU shape for {row['sample_name']}: {acc.shape}, {gyro.shape}")
    imu = resample_lc_to_cl(torch.cat([acc, gyro], dim=1), 6, imu_len, "right_imu")
    return emg, imu


def _channel_stats(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    merged = torch.cat(values, dim=1).double()
    mean = merged.mean(dim=1).float()
    std = merged.std(dim=1, unbiased=False).float().clamp_min(1e-6)
    return mean, std


def build_signal_caches(
    dataset_root: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    emg_length: int = 512,
    imu_length: int = 256,
    overwrite: bool = False,
) -> dict[str, str]:
    """Compute train-only normalization and cache standardized right-hand tensors."""
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: output_dir / f"{split}_right_signals.pt" for split in ("train", "test")}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"Signal cache already exists under {output_dir}")
    rows_by_split = {"train": read_jsonl(train_manifest), "test": read_jsonl(test_manifest)}
    raw: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for split, rows in rows_by_split.items():
        emg_values, imu_values = [], []
        for index, row in enumerate(rows, 1):
            emg, imu = _load_right_signals(dataset_root, row, emg_length, imu_length)
            emg_values.append(emg)
            imu_values.append(imu)
            if index % 250 == 0:
                print(f"{split}: loaded {index}/{len(rows)}", flush=True)
        raw[split] = emg_values, imu_values
    emg_mean, emg_std = _channel_stats(raw["train"][0])
    imu_mean, imu_std = _channel_stats(raw["train"][1])
    stats = {
        "right_emg_mean": emg_mean.tolist(), "right_emg_std": emg_std.tolist(),
        "right_imu_mean": imu_mean.tolist(), "right_imu_std": imu_std.tolist(),
        "emg_target_length": emg_length, "imu_target_length": imu_length,
        "normalization_source": str(Path(train_manifest).resolve()),
    }
    for split, rows in rows_by_split.items():
        emg_values, imu_values = raw[split]
        emg = torch.stack([(x - emg_mean[:, None]) / emg_std[:, None] for x in emg_values])
        imu = torch.stack([(x - imu_mean[:, None]) / imu_std[:, None] for x in imu_values])
        torch.save({"records": rows, "right_emg": emg, "right_imu": imu, "stats": stats}, outputs[split])
    write_json(output_dir / "right_signal_stats.json", stats)
    return {key: str(value) for key, value in outputs.items()}


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    cache = safe_load(path)
    required = {"features", "records"}
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"Invalid RGB feature cache: {path}")
    if len(cache["records"]) != int(cache["features"].shape[0]):
        raise ValueError(f"Record/feature mismatch: {path}")
    return cache


def load_signal_cache(path: str | Path) -> dict[str, Any]:
    cache = safe_load(path)
    if not isinstance(cache, dict) or not {"records", "stats"}.issubset(cache):
        raise ValueError(f"Invalid signal cache: {path}")
    emg_key = "emg" if "emg" in cache else "right_emg"
    imu_key = "imu" if "imu" in cache else "right_imu"
    if emg_key not in cache or imu_key not in cache:
        raise ValueError(f"Signal cache has no compatible EMG/IMU tensors: {path}")
    count = len(cache["records"])
    if count != cache[emg_key].shape[0] or count != cache[imu_key].shape[0]:
        raise ValueError(f"Signal record/tensor mismatch: {path}")
    cache["_emg_key"] = emg_key
    cache["_imu_key"] = imu_key
    return cache
