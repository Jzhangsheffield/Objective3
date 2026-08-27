from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .cache import safe_load
from .io import read_jsonl, write_json


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _interpolate_to_grid(value: Any, source_timestamps: Any, channels: int,
                         target_grid: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).float()
    source = torch.as_tensor(source_timestamps).double().flatten()
    if value.ndim != 2 or int(value.shape[1]) != channels:
        raise ValueError(f"{name} must be [L,{channels}], got {tuple(value.shape)}")
    if source.numel() != value.shape[0] or value.shape[0] < 2:
        raise ValueError(f"{name} value/timestamp length mismatch: {tuple(value.shape)} vs {source.numel()}")
    if not torch.isfinite(value).all() or not torch.isfinite(source).all() or torch.any(source[1:] < source[:-1]):
        raise ValueError(f"Invalid {name} values or board timestamps")
    # The PT was already sliced with the common annotation/RGB clip interval. A device's
    # first/last sample can sit just inside that interval; nearest-edge fill preserves the
    # clip boundary without cropping both hands to their timestamp intersection.
    target = target_grid.to(dtype=torch.float64).clamp(float(source[0]), float(source[-1]))
    upper = torch.searchsorted(source, target, right=False).clamp(1, source.numel() - 1)
    lower = upper - 1
    fraction = ((target - source[lower]) / (source[upper] - source[lower]).clamp_min(1e-12)).float()
    interpolated = value[lower] + (value[upper] - value[lower]) * fraction[:, None]
    return interpolated.transpose(0, 1).contiguous()


def _load_bilateral(dataset_root: Path, row: dict[str, Any], emg_len: int, imu_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    loaded = safe_load(dataset_root / str(row["mindrove"]))
    if not isinstance(loaded, dict):
        raise TypeError(f"MindRove file for {row['sample_name']} is not a dict")
    required = {"left_emg", "right_emg", "left_acc", "left_gyro", "right_acc", "right_gyro",
                "left_board_ts", "right_board_ts", "meta"}
    missing = sorted(required - set(loaded))
    if missing:
        raise KeyError(f"MindRove file for {row['sample_name']} misses {missing}")
    meta = loaded["meta"]
    shared_start = float(meta["annotation_start_sec"])
    shared_end = float(meta["annotation_end_sec"])
    if not shared_end > shared_start:
        raise ValueError(f"Invalid RGB-aligned annotation interval for {row['sample_name']}")
    emg_grid = torch.linspace(shared_start, shared_end, emg_len, dtype=torch.float64)
    imu_grid = torch.linspace(shared_start, shared_end, imu_len, dtype=torch.float64)
    left_emg = _interpolate_to_grid(loaded["left_emg"], loaded["left_board_ts"], 8, emg_grid, "left_emg")
    right_emg = _interpolate_to_grid(loaded["right_emg"], loaded["right_board_ts"], 8, emg_grid, "right_emg")
    left_imu = torch.cat([torch.as_tensor(loaded["left_acc"]), torch.as_tensor(loaded["left_gyro"])], dim=1)
    right_imu = torch.cat([torch.as_tensor(loaded["right_acc"]), torch.as_tensor(loaded["right_gyro"])], dim=1)
    left_imu = _interpolate_to_grid(left_imu, loaded["left_board_ts"], 6, imu_grid, "left_imu")
    right_imu = _interpolate_to_grid(right_imu, loaded["right_board_ts"], 6, imu_grid, "right_imu")
    return torch.cat([left_emg, right_emg], dim=0), torch.cat([left_imu, right_imu], dim=0)


def _stats(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not values:
        raise ValueError("Cannot compute normalization statistics from zero clips")
    merged = torch.cat(values, dim=1).double()
    return merged.mean(1).float(), merged.std(1, unbiased=False).float().clamp_min(1e-6)


def _normalise(values: list[torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.stack([(value - mean[:, None]) / std[:, None] for value in values])


def _serialise_stats(mean: torch.Tensor, std: torch.Tensor) -> dict[str, list[float]]:
    return {"mean": mean.tolist(), "std": std.tolist()}


def build_bilateral_signal_caches(dataset_root: str | Path, train_manifest: str | Path,
                                  test_all_manifest: str | Path, test_normal_manifest: str | Path,
                                  test_fault_manifest: str | Path, output_dir: str | Path,
                                  participant: str, signal_config: dict[str, Any],
                                  emg_length: int = 512, imu_length: int = 256,
                                  overwrite: bool = False) -> dict[str, str]:
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_bilateral_signals.pt"
    protocol_specs = dict(signal_config["test_protocols"])
    test_paths = {name: output_dir / str(spec["cache_filename"]) for name, spec in protocol_specs.items()}
    outputs = [train_path, *test_paths.values(), output_dir / "bilateral_signal_stats.json"]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Bilateral signal cache already exists under {output_dir}")

    train_rows = read_jsonl(train_manifest)
    test_rows = read_jsonl(test_all_manifest)
    normal_names = {str(row["sample_name"]) for row in read_jsonl(test_normal_manifest)}
    fault_names = {str(row["sample_name"]) for row in read_jsonl(test_fault_manifest)}
    train_raw, test_raw = [], []
    for split, rows, target in (("train", train_rows, train_raw), ("test", test_rows, test_raw)):
        for index, row in enumerate(rows, 1):
            target.append(_load_bilateral(dataset_root, row, emg_length, imu_length))
            if index % 250 == 0:
                print(f"{split}: loaded bilateral {index}/{len(rows)}", flush=True)

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(train_rows):
        grouped.setdefault(str(row["participant"]), []).append(index)
    participant_stats: dict[str, dict[str, Any]] = {}
    train_emg = torch.empty((len(train_rows), 16, emg_length), dtype=torch.float32)
    train_imu = torch.empty((len(train_rows), 12, imu_length), dtype=torch.float32)
    for person, indices in sorted(grouped.items()):
        emg_mean, emg_std = _stats([train_raw[index][0] for index in indices])
        imu_mean, imu_std = _stats([train_raw[index][1] for index in indices])
        for index in indices:
            train_emg[index] = (train_raw[index][0] - emg_mean[:, None]) / emg_std[:, None]
            train_imu[index] = (train_raw[index][1] - imu_mean[:, None]) / imu_std[:, None]
        participant_stats[person] = {"emg": _serialise_stats(emg_mean, emg_std),
                                     "imu": _serialise_stats(imu_mean, imu_std), "clips": len(indices)}
    pooled_emg_mean, pooled_emg_std = _stats([value[0] for value in train_raw])
    pooled_imu_mean, pooled_imu_std = _stats([value[1] for value in train_raw])

    calibration = dict(signal_config["calibration"])
    explicit = list(calibration.get("explicit_runs", {}).get(participant, []))
    available_runs = sorted({str(row["run"]) for row in test_rows})
    calibration_runs = explicit or available_runs[:int(calibration.get("runs_per_participant", 1))]
    unknown = sorted(set(calibration_runs) - set(available_runs))
    if unknown:
        raise ValueError(f"Configured calibration runs are absent for {participant}: {unknown}")
    calibration_indices = [index for index, row in enumerate(test_rows) if str(row["run"]) in calibration_runs]
    if not calibration_indices:
        raise ValueError(f"No calibration clips selected for held-out participant {participant}")
    if not bool(calibration.get("exclude_calibration_runs_from_evaluation", True)):
        raise ValueError("This package requires calibration runs to be excluded from scored evaluation")
    evaluation_indices = [index for index, row in enumerate(test_rows) if str(row["run"]) not in calibration_runs]
    if not evaluation_indices:
        raise ValueError(f"Calibration selection leaves no evaluation clips for {participant}")
    evaluation_rows = [test_rows[index] for index in evaluation_indices]
    calibration_emg_mean, calibration_emg_std = _stats([test_raw[index][0] for index in calibration_indices])
    calibration_imu_mean, calibration_imu_std = _stats([test_raw[index][1] for index in calibration_indices])

    common_metadata = {
        "signal_scope": "bilateral", "emg_channels": 16, "imu_channels": 12,
        "emg_target_length": emg_length, "imu_target_length": imu_length,
        "channel_order": signal_config["channel_order"],
        "shared_time_grid": signal_config["shared_time_grid"],
        "train_manifest": str(Path(train_manifest).resolve()),
    }
    torch.save({"records": train_rows, "emg": train_emg, "imu": train_imu,
                "stats": {**common_metadata, "normalization": signal_config["train_normalization"]}}, train_path)
    protocol_tensors = {
        "pooled_train": (
            _normalise([test_raw[index][0] for index in evaluation_indices], pooled_emg_mean, pooled_emg_std),
            _normalise([test_raw[index][1] for index in evaluation_indices], pooled_imu_mean, pooled_imu_std),
        ),
        "participant_calibrated": (
            _normalise([test_raw[index][0] for index in evaluation_indices], calibration_emg_mean, calibration_emg_std),
            _normalise([test_raw[index][1] for index in evaluation_indices], calibration_imu_mean, calibration_imu_std),
        ),
    }
    for protocol, path in test_paths.items():
        emg, imu = protocol_tensors[protocol]
        torch.save({"records": evaluation_rows, "emg": emg, "imu": imu,
                    "stats": {**common_metadata, "normalization": protocol_specs[protocol]["normalization"],
                              "calibration_runs": calibration_runs}}, path)
        protocol_root = output_dir / "evaluation_protocols" / protocol
        _write_jsonl(protocol_root / "test_all.jsonl", evaluation_rows)
        _write_jsonl(protocol_root / "test_normal.jsonl",
                     [row for row in evaluation_rows if str(row["sample_name"]) in normal_names])
        _write_jsonl(protocol_root / "test_fault.jsonl",
                     [row for row in evaluation_rows if str(row["sample_name"]) in fault_names])

    stats = {
        **common_metadata,
        "participant": participant,
        "training_participant_stats": participant_stats,
        "pooled_training": {"emg": _serialise_stats(pooled_emg_mean, pooled_emg_std),
                            "imu": _serialise_stats(pooled_imu_mean, pooled_imu_std)},
        "heldout_calibration": {"runs": calibration_runs, "clips": len(calibration_indices),
                                "emg": _serialise_stats(calibration_emg_mean, calibration_emg_std),
                                "imu": _serialise_stats(calibration_imu_mean, calibration_imu_std)},
        "evaluation_clips": len(evaluation_rows),
        "evaluation_runs": sorted({str(row["run"]) for row in evaluation_rows}),
    }
    write_json(output_dir / "bilateral_signal_stats.json", stats)
    return {"train": str(train_path), **{name: str(path) for name, path in test_paths.items()}}
