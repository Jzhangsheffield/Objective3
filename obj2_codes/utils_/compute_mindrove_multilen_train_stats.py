from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_DATASETS = (
    "Only_falut_run_as_test_J",
    "Only_falut_run_as_test_M",
    "A_as_test",
    "D_as_test",
    "J_as_test",
    "M_as_test",
    "Only_falut_run_as_test_A",
    "Only_falut_run_as_test_D",
)
EMG_LENGTHS = (256, 512, 1024, 2048)
IMU_LENGTHS = (64, 128, 256, 512)
PAIRED_LENGTHS = tuple(zip(EMG_LENGTHS, IMU_LENGTHS))
EMG_CHANNELS = tuple(f"emg_{i}" for i in range(8))
IMU_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


@dataclass
class ChannelMoments:
    channels: int
    count: int = 0
    num_samples: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.mean = torch.zeros(self.channels, dtype=torch.float64)
        self.m2 = torch.zeros(self.channels, dtype=torch.float64)

    def update(self, x_cf: torch.Tensor) -> None:
        if x_cf.ndim != 2 or x_cf.shape[0] != self.channels:
            raise ValueError(
                f"Expected [{self.channels}, L], got {tuple(x_cf.shape)}"
            )
        x = x_cf.to(dtype=torch.float64)
        batch_count = int(x.shape[1])
        batch_mean = x.mean(dim=1)
        centered = x - batch_mean[:, None]
        batch_m2 = (centered * centered).sum(dim=1)

        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.count = batch_count
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean.add_(delta * (batch_count / total))
            self.m2.add_(batch_m2 + delta.square() * (self.count * batch_count / total))
            self.count = total
        self.num_samples += 1

    def finalize(self, channel_names: tuple[str, ...]) -> dict[str, Any]:
        if self.count <= 0 or self.num_samples <= 0:
            raise RuntimeError("Cannot finalize empty statistics")
        var = torch.clamp(self.m2 / self.count, min=0.0)
        std = torch.sqrt(var)
        mean_values = [float(v) for v in self.mean.tolist()]
        var_values = [float(v) for v in var.tolist()]
        std_values = [float(v) for v in std.tolist()]
        return {
            "num_samples": self.num_samples,
            "num_channels": self.channels,
            "count_per_channel": self.count,
            "channel_order": list(channel_names),
            "mean": mean_values,
            "std": std_values,
            "var": var_values,
            "per_channel": [
                {
                    "channel": i,
                    "name": channel_names[i],
                    "count": self.count,
                    "mean": mean_values[i],
                    "std": std_values[i],
                    "var": var_values[i],
                }
                for i in range(self.channels)
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute multi-length MindRove train-set per-channel mean/std."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--manifest-name", default="train_manifest.jsonl")
    parser.add_argument("--output-name", default="mindrove_train_normalization_stats.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional mirrored output root, primarily for validation runs.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def load_manifest(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def require_lc_tensor(obj: dict[str, Any], key: str, channels: int, context: str) -> torch.Tensor:
    if key not in obj:
        raise KeyError(f"{context}: missing key {key!r}")
    value = obj[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{context}: {key} is not a tensor: {type(value)}")
    if value.ndim != 2 or value.shape[1] != channels or value.shape[0] <= 0:
        raise ValueError(
            f"{context}: {key} must have shape [L,{channels}] with L>0; "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{context}: {key} contains NaN or Inf")
    return value.to(dtype=torch.float32)


def resample_lc_to_cf(x_lc: torch.Tensor, target_len: int) -> torch.Tensor:
    x = x_lc.transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    y = y.squeeze(0).contiguous()
    if not bool(torch.isfinite(y).all()):
        raise ValueError("Resampled tensor contains NaN or Inf")
    return y


def make_accumulators() -> dict[str, dict[int, dict[str, ChannelMoments]]]:
    return {
        "emg": {
            length: {
                "left_emg": ChannelMoments(8),
                "right_emg": ChannelMoments(8),
            }
            for length in EMG_LENGTHS
        },
        "imu": {
            length: {
                "left_imu": ChannelMoments(6),
                "right_imu": ChannelMoments(6),
            }
            for length in IMU_LENGTHS
        },
    }


def process_dataset(
    dataset_root: Path,
    dataset_name: str,
    manifest_name: str,
    output_name: str,
    output_root: Path | None,
    max_records: int | None,
    progress_every: int,
) -> Path:
    dataset_dir = dataset_root / dataset_name
    manifest_path = dataset_dir / manifest_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = load_manifest(manifest_path, max_records)
    accumulators = make_accumulators()
    started = time.time()

    for index, record in enumerate(records, start=1):
        sample_name = str(record.get("sample_name", f"record_{index}"))
        rel_path = record.get("mindrove")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError(f"{dataset_name}/{sample_name}: missing mindrove path")
        mindrove_path = Path(rel_path)
        if not mindrove_path.is_absolute():
            mindrove_path = dataset_root / mindrove_path
        if not mindrove_path.is_file():
            raise FileNotFoundError(
                f"{dataset_name}/{sample_name}: MindRove file not found: {mindrove_path}"
            )

        loaded = torch.load(mindrove_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError(
                f"{dataset_name}/{sample_name}: expected dict in {mindrove_path}, "
                f"got {type(loaded)}"
            )
        context = f"{dataset_name}/{sample_name} ({mindrove_path})"
        left_emg = require_lc_tensor(loaded, "left_emg", 8, context)
        right_emg = require_lc_tensor(loaded, "right_emg", 8, context)
        left_acc = require_lc_tensor(loaded, "left_acc", 3, context)
        right_acc = require_lc_tensor(loaded, "right_acc", 3, context)
        left_gyro = require_lc_tensor(loaded, "left_gyro", 3, context)
        right_gyro = require_lc_tensor(loaded, "right_gyro", 3, context)

        if left_acc.shape[0] != left_gyro.shape[0]:
            raise ValueError(f"{context}: left acc/gyro lengths differ")
        if right_acc.shape[0] != right_gyro.shape[0]:
            raise ValueError(f"{context}: right acc/gyro lengths differ")
        left_imu = torch.cat((left_acc, left_gyro), dim=1)
        right_imu = torch.cat((right_acc, right_gyro), dim=1)

        for target_len in EMG_LENGTHS:
            accumulators["emg"][target_len]["left_emg"].update(
                resample_lc_to_cf(left_emg, target_len)
            )
            accumulators["emg"][target_len]["right_emg"].update(
                resample_lc_to_cf(right_emg, target_len)
            )
        for target_len in IMU_LENGTHS:
            accumulators["imu"][target_len]["left_imu"].update(
                resample_lc_to_cf(left_imu, target_len)
            )
            accumulators["imu"][target_len]["right_imu"].update(
                resample_lc_to_cf(right_imu, target_len)
            )

        if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
            elapsed = time.time() - started
            print(
                f"[{dataset_name}] {index}/{len(records)} samples "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    emg_output: dict[str, Any] = {}
    for length in EMG_LENGTHS:
        emg_output[str(length)] = {
            "target_len": length,
            "groups": {
                "left_emg": accumulators["emg"][length]["left_emg"].finalize(EMG_CHANNELS),
                "right_emg": accumulators["emg"][length]["right_emg"].finalize(EMG_CHANNELS),
            },
        }

    imu_output: dict[str, Any] = {}
    for length in IMU_LENGTHS:
        imu_output[str(length)] = {
            "target_len": length,
            "groups": {
                "left_imu": accumulators["imu"][length]["left_imu"].finalize(IMU_CHANNELS),
                "right_imu": accumulators["imu"][length]["right_imu"].finalize(IMU_CHANNELS),
            },
        }

    configurations: dict[str, Any] = {}
    for emg_len, imu_len in PAIRED_LENGTHS:
        left_emg_stats = emg_output[str(emg_len)]["groups"]["left_emg"]
        right_emg_stats = emg_output[str(emg_len)]["groups"]["right_emg"]
        left_imu_stats = imu_output[str(imu_len)]["groups"]["left_imu"]
        right_imu_stats = imu_output[str(imu_len)]["groups"]["right_imu"]
        configurations[f"emg{emg_len}_imu{imu_len}"] = {
            "emg_target_len": emg_len,
            "imu_target_len": imu_len,
            "left_emg_mean": left_emg_stats["mean"],
            "left_emg_std": left_emg_stats["std"],
            "right_emg_mean": right_emg_stats["mean"],
            "right_emg_std": right_emg_stats["std"],
            "left_imu_mean": left_imu_stats["mean"],
            "left_imu_std": left_imu_stats["std"],
            "right_imu_mean": right_imu_stats["mean"],
            "right_imu_std": right_imu_stats["std"],
        }

    output = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "source_manifest": str(manifest_path),
        "num_manifest_records": len(records),
        "loaded_mindrove_files": len(records),
        "statistics": "population mean/variance/std over all resampled time points",
        "variance_ddof": 0,
        "resample_method": "torch.nn.functional.interpolate(mode='linear', align_corners=False)",
        "emg_target_lengths": list(EMG_LENGTHS),
        "imu_target_lengths": list(IMU_LENGTHS),
        "paired_configurations": [
            {"emg_target_len": emg_len, "imu_target_len": imu_len}
            for emg_len, imu_len in PAIRED_LENGTHS
        ],
        "imu_channel_order": list(IMU_CHANNELS),
        "emg": emg_output,
        "imu": imu_output,
        "normalization_by_configuration": configurations,
        "elapsed_seconds": time.time() - started,
    }

    destination_dir = (output_root / dataset_name) if output_root is not None else dataset_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / output_name
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, destination)
    print(f"[Saved] {destination}", flush=True)
    return destination


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else None
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive")
    outputs = []
    for dataset_name in args.datasets:
        outputs.append(
            process_dataset(
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                manifest_name=args.manifest_name,
                output_name=args.output_name,
                output_root=output_root,
                max_records=args.max_records,
                progress_every=args.progress_every,
            )
        )
    print(f"Completed {len(outputs)} dataset(s).", flush=True)


if __name__ == "__main__":
    main()
