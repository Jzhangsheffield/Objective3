from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .io import read_jsonl, write_json


def safe_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _lookup(records: list[dict]) -> dict[str, int]:
    result = {str(row["sample_name"]): index for index, row in enumerate(records)}
    if len(result) != len(records):
        raise ValueError("Duplicate sample_name in cache")
    return result


def resample_lc_to_cl(value: torch.Tensor, channels: int, length: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim != 2 or value.shape[1] != channels or value.shape[0] < 1:
        raise ValueError(f"Invalid {name} shape: {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return F.interpolate(value.T.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)


def load_right_imu(dataset_root: str | Path, row: dict, target_length: int) -> torch.Tensor:
    loaded = safe_load(Path(dataset_root) / str(row["mindrove"]))
    if not isinstance(loaded, dict) or not {"right_acc", "right_gyro"}.issubset(loaded):
        raise ValueError(f"Invalid right IMU file for {row['sample_name']}")
    acc = torch.as_tensor(loaded["right_acc"])
    gyro = torch.as_tensor(loaded["right_gyro"])
    if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"Invalid right IMU shape for {row['sample_name']}: {acc.shape}, {gyro.shape}")
    return resample_lc_to_cl(torch.cat([acc, gyro], dim=1), 6, target_length, "right_imu")


def build_imu_caches(
    dataset_root: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    target_length: int = 256,
    overwrite: bool = False,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    train_output = output_dir / "train_imu.pt"
    test_output = output_dir / "test_imu.pt"
    if (train_output.exists() or test_output.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite IMU cache in {output_dir}")
    train_rows, test_rows = read_jsonl(train_manifest), read_jsonl(test_manifest)
    train_raw = [load_right_imu(dataset_root, row, target_length) for row in train_rows]
    test_raw = [load_right_imu(dataset_root, row, target_length) for row in test_rows]
    merged = torch.cat(train_raw, dim=1)
    mean = merged.mean(dim=1).float()
    std = merged.std(dim=1, unbiased=False).float().clamp_min(1e-6)
    normalise = lambda values: torch.stack([(value - mean[:, None]) / std[:, None] for value in values])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"imu": normalise(train_raw), "records": train_rows}, train_output)
    torch.save({"imu": normalise(test_raw), "records": test_rows}, test_output)
    write_json(output_dir / "metadata.json", {
        "train_manifest": str(Path(train_manifest).resolve()),
        "test_manifest": str(Path(test_manifest).resolve()),
        "target_length": target_length,
        "channels": 6,
        "normalization": "train-only channel z-score",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
    })
    return {"train": str(train_output), "test": str(test_output)}


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


class IMUDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        manifest: str | Path,
        training: bool = False,
        shift_probability: float = 0.0,
        shift_max_fraction: float = 0.0,
        fixed_offset: float = 0.0,
    ) -> None:
        cache = safe_load(cache_path)
        self.imu = cache["imu"].float()
        self.records = cache["records"]
        self.lookup = _lookup(self.records)
        self.rows = read_jsonl(manifest)
        self.training = training
        self.shift_probability = float(shift_probability)
        self.shift_max_fraction = float(shift_max_fraction)
        self.fixed_offset = float(fixed_offset)
        missing = {str(row["sample_name"]) for row in self.rows} - set(self.lookup)
        if missing:
            raise KeyError(f"IMU cache misses samples: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        signal = self.imu[self.lookup[str(row["sample_name"])]].clone()
        offset = self.fixed_offset
        if self.training and torch.rand(()) < self.shift_probability:
            offset += float(torch.empty(()).uniform_(-self.shift_max_fraction, self.shift_max_fraction))
        return {
            "imu": zero_shift(signal, offset),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "row": row,
        }


@dataclass(frozen=True)
class HistoryExample:
    current: str
    history: tuple[str, ...]
    row: dict[str, Any]


def make_history_examples(rows: list[dict]) -> list[HistoryExample]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["participant"]), str(row["run"])), []).append(row)
    examples = []
    for run_rows in grouped.values():
        run_rows.sort(key=lambda value: int(value["annotation_row_index"]))
        for index, row in enumerate(run_rows):
            examples.append(HistoryExample(
                str(row["sample_name"]),
                tuple(str(value["sample_name"]) for value in run_rows[:index]),
                row,
            ))
    return sorted(examples, key=lambda value: (
        str(value.row["participant"]), str(value.row["run"]), int(value.row["annotation_row_index"])
    ))


class GlobalFeatureHistoryDataset(Dataset):
    def __init__(self, cache_path: str | Path, manifest: str | Path) -> None:
        cache = safe_load(cache_path)
        self.features = cache["features"].float()
        self.lookup = _lookup(cache["records"])
        self.examples = make_history_examples(read_jsonl(manifest))
        missing = {value.current for value in self.examples} - set(self.lookup)
        if missing:
            raise KeyError(f"Global feature cache misses samples: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        history = torch.stack([self.features[self.lookup[name]] for name in example.history]) \
            if example.history else self.features.new_zeros((0, self.features.shape[1]))
        return {
            "current": self.features[self.lookup[example.current]],
            "history": history,
            "node_target": int(example.row["node_idx"]) - 1,
            "tier3_target": int(example.row["tier3_id"]),
            "stage_id": int(example.row["stage_id"]),
            "row": example.row,
        }


def collate_global_history(batch: list[dict]) -> dict[str, Any]:
    batch_size = len(batch)
    max_history = max(value["history"].shape[0] for value in batch)
    feature_dim = batch[0]["current"].shape[-1]
    history = torch.zeros((batch_size, max_history, feature_dim), dtype=torch.float32)
    padding = torch.ones((batch_size, max_history), dtype=torch.bool)
    positions = torch.zeros((batch_size, max_history), dtype=torch.long)
    for index, value in enumerate(batch):
        length = value["history"].shape[0]
        if length:
            history[index, :length] = value["history"]
            padding[index, :length] = False
            positions[index, :length] = torch.arange(length, 0, -1)
    return {
        "current": torch.stack([value["current"] for value in batch]),
        "history": history,
        "history_padding_mask": padding,
        "history_position_ids": positions,
        "node_target": torch.tensor([value["node_target"] for value in batch]),
        "tier3_target": torch.tensor([value["tier3_target"] for value in batch]),
        "stage_id": torch.tensor([value["stage_id"] for value in batch]),
        "rows": [value["row"] for value in batch],
    }


class TokenHistoryDataset(Dataset):
    MODALITIES = ("cam0", "cam1", "imu")

    def __init__(self, cache_paths: dict[str, str | Path], manifest: str | Path, use_history: bool) -> None:
        self.caches = {name: safe_load(path) for name, path in cache_paths.items()}
        self.lookups = {name: _lookup(cache["records"]) for name, cache in self.caches.items()}
        self.examples = make_history_examples(read_jsonl(manifest))
        self.use_history = bool(use_history)
        expected = {value.current for value in self.examples}
        for name in self.MODALITIES:
            missing = expected - set(self.lookups[name])
            if missing:
                raise KeyError(f"{name} temporal cache misses samples: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.examples)

    def _tokens(self, modality: str, sample_name: str) -> dict[str, torch.Tensor]:
        cache = self.caches[modality]
        index = self.lookups[modality][sample_name]
        return {
            "global": cache["global_features"][index].float(),
            "temporal": cache["temporal_tokens"][index].float(),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        history_names = example.history if self.use_history else ()
        current = {name: self._tokens(name, example.current) for name in self.MODALITIES}
        history = {}
        for name in self.MODALITIES:
            if history_names:
                values = [self._tokens(name, sample) for sample in history_names]
                history[name] = {
                    "global": torch.stack([value["global"] for value in values]),
                    "temporal": torch.stack([value["temporal"] for value in values]),
                }
            else:
                history[name] = {
                    "global": current[name]["global"].new_zeros((0, *current[name]["global"].shape)),
                    "temporal": current[name]["temporal"].new_zeros((0, *current[name]["temporal"].shape)),
                }
        return {
            "current": current,
            "history": history,
            "node_target": int(example.row["node_idx"]) - 1,
            "tier3_target": int(example.row["tier3_id"]),
            "stage_id": int(example.row["stage_id"]),
            "row": example.row,
        }


def collate_token_history(batch: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    max_history = max(value["history"]["cam0"]["global"].shape[0] for value in batch)
    for modality in TokenHistoryDataset.MODALITIES:
        for kind in ("global", "temporal"):
            result[f"current_{modality}_{kind}"] = torch.stack(
                [value["current"][modality][kind] for value in batch]
            )
            shape = batch[0]["current"][modality][kind].shape
            history = torch.zeros((len(batch), max_history, *shape), dtype=torch.float32)
            for index, value in enumerate(batch):
                length = value["history"][modality][kind].shape[0]
                if length:
                    history[index, :length] = value["history"][modality][kind]
            result[f"history_{modality}_{kind}"] = history
    result["history_padding_mask"] = torch.ones((len(batch), max_history), dtype=torch.bool)
    result["history_position_ids"] = torch.zeros((len(batch), max_history), dtype=torch.long)
    for index, value in enumerate(batch):
        length = value["history"]["cam0"]["global"].shape[0]
        if length:
            result["history_padding_mask"][index, :length] = False
            result["history_position_ids"][index, :length] = torch.arange(length, 0, -1)
    result["available"] = torch.ones((len(batch), 3), dtype=torch.bool)
    result["node_target"] = torch.tensor([value["node_target"] for value in batch])
    result["tier3_target"] = torch.tensor([value["tier3_target"] for value in batch])
    result["stage_id"] = torch.tensor([value["stage_id"] for value in batch])
    result["rows"] = [value["row"] for value in batch]
    return result
