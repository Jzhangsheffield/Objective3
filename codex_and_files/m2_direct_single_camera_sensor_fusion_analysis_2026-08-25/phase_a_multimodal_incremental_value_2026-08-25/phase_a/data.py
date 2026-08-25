from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .cache import load_feature_cache, load_signal_cache
from .io import read_jsonl


@dataclass(frozen=True)
class Example:
    current: str
    history: tuple[str, ...]
    row: dict[str, Any]


def _index(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {str(row["sample_name"]): index for index, row in enumerate(records)}
    if len(result) != len(records):
        raise ValueError("A cache contains duplicate sample_name values")
    return result


def _zero_shift(signal: torch.Tensor, fraction: float) -> torch.Tensor:
    amount = int(round(signal.shape[-1] * fraction))
    if amount == 0:
        return signal
    shifted = torch.zeros_like(signal)
    if amount > 0 and amount < signal.shape[-1]:
        shifted[..., amount:] = signal[..., :-amount]
    elif amount < 0 and -amount < signal.shape[-1]:
        shifted[..., :amount] = signal[..., -amount:]
    return shifted


class MultimodalHistoryDataset(Dataset):
    def __init__(
        self,
        primary_cache: str | Path,
        secondary_cache: str | Path,
        signal_cache: str | Path,
        selection_manifest: str | Path,
        drop_modalities: tuple[str, ...] = (),
        sensor_offset_fraction: float = 0.0,
        emg_offset_fraction: float | None = None,
        imu_offset_fraction: float | None = None,
        training: bool = False,
        time_shift_augmentation_probability: float = 0.0,
        time_shift_augmentation_max_fraction: float = 0.0,
    ) -> None:
        self.primary = load_feature_cache(primary_cache)
        self.secondary = load_feature_cache(secondary_cache)
        self.signals = load_signal_cache(signal_cache)
        self.rows = read_jsonl(selection_manifest)
        self.drop_modalities = set(drop_modalities)
        self.emg_offset_fraction = float(sensor_offset_fraction if emg_offset_fraction is None else emg_offset_fraction)
        self.imu_offset_fraction = float(sensor_offset_fraction if imu_offset_fraction is None else imu_offset_fraction)
        self.training = bool(training)
        self.time_shift_augmentation_probability = float(time_shift_augmentation_probability)
        self.time_shift_augmentation_max_fraction = float(time_shift_augmentation_max_fraction)
        self.lookup = {
            "primary": _index(self.primary["records"]),
            "secondary": _index(self.secondary["records"]),
            "signal": _index(self.signals["records"]),
        }
        sample_names = {str(row["sample_name"]) for row in self.rows}
        for name, lookup in self.lookup.items():
            missing = sorted(sample_names - set(lookup))
            if missing:
                raise KeyError(f"{name} cache misses {len(missing)} samples, e.g. {missing[:5]}")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            grouped.setdefault((str(row["participant"]), str(row["run"])), []).append(row)
        self.examples: list[Example] = []
        for run_rows in grouped.values():
            run_rows.sort(key=lambda value: int(value["annotation_row_index"]))
            for position, row in enumerate(run_rows):
                self.examples.append(Example(str(row["sample_name"]), tuple(
                    str(previous["sample_name"]) for previous in run_rows[:position]
                ), row))
        self.examples.sort(key=lambda item: (
            str(item.row["participant"]), str(item.row["run"]), int(item.row["annotation_row_index"])
        ))

    def __len__(self) -> int:
        return len(self.examples)

    def _one(self, sample_name: str) -> dict[str, torch.Tensor]:
        p = self.lookup["primary"][sample_name]
        s = self.lookup["secondary"][sample_name]
        m = self.lookup["signal"][sample_name]
        emg_offset, imu_offset = self.emg_offset_fraction, self.imu_offset_fraction
        if self.training and torch.rand(()) < self.time_shift_augmentation_probability:
            emg_offset += float(torch.empty(()).uniform_(
                -self.time_shift_augmentation_max_fraction, self.time_shift_augmentation_max_fraction
            ))
        if self.training and torch.rand(()) < self.time_shift_augmentation_probability:
            imu_offset += float(torch.empty(()).uniform_(
                -self.time_shift_augmentation_max_fraction, self.time_shift_augmentation_max_fraction
            ))
        emg = _zero_shift(self.signals["right_emg"][m].float(), emg_offset)
        imu = _zero_shift(self.signals["right_imu"][m].float(), imu_offset)
        return {
            "primary": self.primary["features"][p].float(),
            "secondary": self.secondary["features"][s].float(),
            "emg": emg,
            "imu": imu,
            "secondary_available": torch.tensor("secondary" not in self.drop_modalities),
            "emg_available": torch.tensor("emg" not in self.drop_modalities),
            "imu_available": torch.tensor("imu" not in self.drop_modalities),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        current = self._one(example.current)
        history = [self._one(sample_name) for sample_name in example.history]
        row = example.row
        return {
            "current": current, "history": history,
            "history_position_ids": torch.arange(len(history), 0, -1, dtype=torch.long),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]), "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]), "participant": str(row["participant"]),
            "run": str(row["run"]), "annotation_row_index": int(row["annotation_row_index"]),
        }


def collate_multimodal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_history = max(len(item["history"]) for item in batch)
    current_keys = list(batch[0]["current"])
    result: dict[str, Any] = {}
    for key in current_keys:
        result[f"current_{key}"] = torch.stack([item["current"][key] for item in batch])
        shape = tuple(batch[0]["current"][key].shape)
        history_tensor = torch.zeros((size, max_history, *shape), dtype=result[f"current_{key}"].dtype)
        for row_index, item in enumerate(batch):
            if item["history"]:
                history_tensor[row_index, :len(item["history"])] = torch.stack(
                    [entry[key] for entry in item["history"]]
                )
        result[f"history_{key}"] = history_tensor
    result["history_position_ids"] = torch.zeros((size, max_history), dtype=torch.long)
    result["history_padding_mask"] = torch.ones((size, max_history), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = len(item["history"])
        if length:
            result["history_position_ids"][row_index, :length] = item["history_position_ids"]
            result["history_padding_mask"][row_index, :length] = False
    for key in ("node_target", "tier3_target", "stage_id"):
        result[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    for key in ("sample_name", "participant", "run", "annotation_row_index"):
        result[key] = [item[key] for item in batch]
    return result
