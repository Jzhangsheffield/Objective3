from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .cache import load_signal_cache, safe_load
from .io import read_jsonl


def _record_index(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {str(row["sample_name"]): index for index, row in enumerate(records)}
    if len(result) != len(records):
        raise ValueError("Duplicate sample_name in cache")
    return result


def zero_shift(signal: torch.Tensor, fraction: float) -> torch.Tensor:
    amount = int(round(signal.shape[-1] * float(fraction)))
    if amount == 0:
        return signal
    shifted = torch.zeros_like(signal)
    if 0 < amount < signal.shape[-1]:
        shifted[..., amount:] = signal[..., :-amount]
    elif -signal.shape[-1] < amount < 0:
        shifted[..., :amount] = signal[..., -amount:]
    return shifted


class SignalClipDataset(Dataset):
    """Current-clip EMG or IMU dataset for right-hand or config-selected bilateral caches."""

    def __init__(
        self,
        cache_path: str | Path,
        manifest_path: str | Path,
        modality: str,
        training: bool = False,
        time_shift_probability: float = 0.0,
        time_shift_max_fraction: float = 0.0,
        fixed_offset_fraction: float = 0.0,
        zero_signal: bool = False,
    ) -> None:
        self.cache = load_signal_cache(cache_path)
        self.rows = read_jsonl(manifest_path)
        self.modality = modality
        self.training = bool(training)
        self.time_shift_probability = float(time_shift_probability)
        self.time_shift_max_fraction = float(time_shift_max_fraction)
        self.fixed_offset_fraction = float(fixed_offset_fraction)
        self.zero_signal = bool(zero_signal)
        self.lookup = _record_index(self.cache["records"])
        missing = [row["sample_name"] for row in self.rows if row["sample_name"] not in self.lookup]
        if missing:
            raise KeyError(f"Signal cache misses {len(missing)} manifest samples, e.g. {missing[:5]}")
        if modality not in {"emg", "imu"}:
            raise ValueError(modality)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cache_index = self.lookup[str(row["sample_name"])]
        key = self.cache["_emg_key"] if self.modality == "emg" else self.cache["_imu_key"]
        signal = self.cache[key][cache_index].float().clone()
        offset = self.fixed_offset_fraction
        if self.training and torch.rand(()) < self.time_shift_probability:
            offset += float(torch.empty(()).uniform_(-self.time_shift_max_fraction, self.time_shift_max_fraction))
        signal = zero_shift(signal, offset)
        if self.zero_signal:
            signal.zero_()
        return {
            "signal": signal,
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
        }


def collate_signal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"signal": torch.stack([row["signal"] for row in batch])}
    for key in ("node_target", "tier3_target", "stage_id"):
        result[key] = torch.tensor([row[key] for row in batch], dtype=torch.long)
    for key in ("sample_name", "participant", "run", "annotation_row_index"):
        result[key] = [row[key] for row in batch]
    return result


@dataclass(frozen=True)
class FeatureExample:
    current: str
    history: tuple[str, ...]
    row: dict[str, Any]


class SignalFeatureHistoryDataset(Dataset):
    """M2 dataset over frozen 512-D features extracted by a Tier3 signal encoder."""

    def __init__(self, feature_cache_path: str | Path, manifest_path: str | Path) -> None:
        cache = safe_load(feature_cache_path)
        if not isinstance(cache, dict) or not {"records", "features"}.issubset(cache):
            raise ValueError(f"Invalid signal feature cache: {feature_cache_path}")
        if len(cache["records"]) != int(cache["features"].shape[0]):
            raise ValueError("Signal feature record/tensor mismatch")
        self.cache = cache
        self.features = cache["features"].float()
        self.lookup = _record_index(cache["records"])
        rows = read_jsonl(manifest_path)
        missing = [row["sample_name"] for row in rows if row["sample_name"] not in self.lookup]
        if missing:
            raise KeyError(f"Feature cache misses {len(missing)} samples, e.g. {missing[:5]}")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((str(row["participant"]), str(row["run"])), []).append(row)
        examples = []
        for run_rows in grouped.values():
            run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
            for position, row in enumerate(run_rows):
                examples.append(FeatureExample(
                    str(row["sample_name"]),
                    tuple(str(previous["sample_name"]) for previous in run_rows[:position]),
                    row,
                ))
        self.examples = sorted(examples, key=lambda item: (
            str(item.row["participant"]), str(item.row["run"]), int(item.row["annotation_row_index"])
        ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        current = self.features[self.lookup[example.current]]
        history = [self.features[self.lookup[name]] for name in example.history]
        row = example.row
        return {
            "current_feature": current,
            "history_features": history,
            "history_position_ids": torch.arange(len(history), 0, -1, dtype=torch.long),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
        }


def collate_feature_history(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_history = max(len(row["history_features"]) for row in batch)
    feature_dim = int(batch[0]["current_feature"].shape[-1])
    history = torch.zeros((size, max_history, feature_dim), dtype=torch.float32)
    positions = torch.zeros((size, max_history), dtype=torch.long)
    padding = torch.ones((size, max_history), dtype=torch.bool)
    for row_index, row in enumerate(batch):
        length = len(row["history_features"])
        if length:
            history[row_index, :length] = torch.stack(row["history_features"])
            positions[row_index, :length] = row["history_position_ids"]
            padding[row_index, :length] = False
    result: dict[str, Any] = {
        "current_feature": torch.stack([row["current_feature"] for row in batch]),
        "history_features": history,
        "history_position_ids": positions,
        "history_padding_mask": padding,
    }
    for key in ("node_target", "tier3_target", "stage_id"):
        result[key] = torch.tensor([row[key] for row in batch], dtype=torch.long)
    for key in ("sample_name", "participant", "run", "annotation_row_index"):
        result[key] = [row[key] for row in batch]
    return result
