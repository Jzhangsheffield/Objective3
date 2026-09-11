from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .annotations import RunInfo, load_annotation, targets_on_grid


IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
EMG_CHANNELS = [f"emg{i}" for i in range(1, 9)]


def _read_sensor(path: Path, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path, usecols=["board_ts", *channels])
    table = table.apply(pd.to_numeric, errors="coerce").dropna()
    table = table.sort_values("board_ts").drop_duplicates("board_ts", keep="last")
    times = table["board_ts"].to_numpy(np.float64)
    values = table[channels].to_numpy(np.float32)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid sensor timestamps: {path}")
    return times, values


def _causal_previous_sample(times: np.ndarray, values: np.ndarray, grid: np.ndarray, max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """
    寻找在grid之前，距离grid最近且距离小于max_gap的传感器值
    """
    indices = np.searchsorted(times, grid, side="right") - 1
    clipped = np.clip(indices, 0, len(times) - 1)
    age = grid - times[clipped]
    valid = (indices >= 0) & (age >= 0) & (age <= max_gap)
    return values[clipped], valid.astype(np.float32)


def _filter_uniform(values: np.ndarray, modality: str, sample_rate: float, cfg: dict[str, Any]) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    order = int(cfg.get("filter_order", 4))
    if modality == "imu" and cfg.get("lowpass_hz"):
        sos = signal.butter(order, float(cfg["lowpass_hz"]), btype="lowpass", fs=sample_rate, output="sos")
        result = signal.sosfilt(sos, result, axis=0)
    if modality == "emg":
        low, high = [float(x) for x in cfg["bandpass_hz"]]
        if high >= sample_rate / 2:
            raise ValueError(f"EMG high cutoff {high} must be below Nyquist {sample_rate / 2}")
        sos = signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        result = signal.sosfilt(sos, result, axis=0)
        notch = cfg.get("notch_hz")
        if notch:
            b, a = signal.iirnotch(float(notch), float(cfg.get("notch_q", 30.0)), fs=sample_rate)
            result = signal.lfilter(b, a, result, axis=0)
    return result.astype(np.float32)


def _make_windows(signals: np.ndarray, decision_indices: np.ndarray, window_samples: int) -> np.ndarray:
    pad = np.zeros((window_samples - 1, signals.shape[1]), dtype=np.float32)
    padded = np.concatenate([pad, signals], axis=0)
    view = np.lib.stride_tricks.sliding_window_view(padded, window_samples, axis=0)
    return np.ascontiguousarray(view[decision_indices].transpose(0, 2, 1))


def extract_run(info: RunInfo, sensor_cfg: dict[str, Any]) -> dict[str, Any]:
    modality = str(sensor_cfg["modality"]).lower()
    if modality not in {"imu", "emg"}:
        raise ValueError(f"Unsupported modality: {modality}")
    source_channels = IMU_CHANNELS if modality == "imu" else EMG_CHANNELS
    left_t, left_x = _read_sensor(info.left_csv, source_channels)
    right_t, right_x = _read_sensor(info.right_csv, source_channels)
    annotation = load_annotation(info)
    sample_rate = float(sensor_cfg["sample_rate_hz"])
    decision_rate = float(sensor_cfg["decision_rate_hz"])
    start = max(annotation["frame_start"], float(left_t[0]), float(right_t[0]))
    end = min(annotation["frame_end"], float(left_t[-1]), float(right_t[-1]))
    if end <= start:
        raise ValueError(f"No common sensor/annotation time range for {info.sample_name}")
    count = int(np.floor((end - start) * sample_rate)) + 1
    raw_grid = start + np.arange(count, dtype=np.float64) / sample_rate
    max_gap = float(sensor_cfg["max_gap_seconds"])
    left, left_valid = _causal_previous_sample(left_t, left_x, raw_grid, max_gap)
    right, right_valid = _causal_previous_sample(right_t, right_x, raw_grid, max_gap)
    left = _filter_uniform(left, modality, sample_rate, sensor_cfg)
    right = _filter_uniform(right, modality, sample_rate, sensor_cfg)
    signals = np.concatenate([left, right], axis=1)
    physical_channels = [f"left_{x}" for x in source_channels] + [f"right_{x}" for x in source_channels]
    if bool(sensor_cfg.get("append_validity_channels", True)):
        signals = np.concatenate([signals, left_valid[:, None], right_valid[:, None]], axis=1)
        channel_names = physical_channels + ["left_valid", "right_valid"]
    else:
        channel_names = physical_channels
    stride = sample_rate / decision_rate
    if abs(stride - round(stride)) > 1e-6:
        raise ValueError("sample_rate_hz must be an integer multiple of decision_rate_hz")
    decision_indices = np.arange(0, len(raw_grid), int(round(stride)), dtype=np.int64)
    decision_times = raw_grid[decision_indices]
    window_samples = int(round(float(sensor_cfg["local_window_seconds"]) * sample_rate))
    windows = _make_windows(signals, decision_indices, window_samples)
    radius_steps = int(round(float(sensor_cfg["boundary_label_radius_ms"]) * decision_rate / 1000.0))
    targets = targets_on_grid(annotation, decision_times, radius_steps)
    dtype = np.float16 if sensor_cfg.get("storage_dtype", "float16") == "float16" else np.float32
    validity_indices = list(range(len(physical_channels), len(channel_names)))
    return {
        "windows": windows.astype(dtype),
        "times": decision_times,
        **{key: targets[key] for key in ("state", "start", "end", "exact_start", "exact_end", "segment_id")},
        "metadata": {
            "sample_name": info.sample_name,
            "participant": info.participant,
            "source_run": info.source_run,
            "modality": modality,
            "channel_names": channel_names,
            "validity_channel_indices": validity_indices,
            "sample_rate_hz": sample_rate,
            "decision_rate_hz": decision_rate,
            "window_samples": window_samples,
            "duration_seconds": float(decision_times[-1] - decision_times[0]),
            "decision_steps": int(len(decision_times)),
            "left_valid_fraction": float(left_valid.mean()),
            "right_valid_fraction": float(right_valid.mean()),
            "stat_sum": signals.astype(np.float64).sum(axis=0).tolist(),
            "stat_sumsq": np.square(signals.astype(np.float64)).sum(axis=0).tolist(),
            "stat_count": int(len(signals)),
            "action_names": {str(k): v for k, v in targets["action_names"].items()},
        },
    }
