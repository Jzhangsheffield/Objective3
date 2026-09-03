from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .common import read_jsonl, safe_torch_load


@dataclass(frozen=True)
class HistoryExample:
    current: str
    history: tuple[str, ...]
    row: dict[str, Any]


class IMUFeatureHistoryDataset(Dataset):
    """Actual same-run history over a frozen IMU 512-D feature cache."""

    def __init__(self, cache_path: str | Path, manifest_path: str | Path) -> None:
        cache = safe_torch_load(cache_path)
        feature_key = "features" if "features" in cache else "global_features"
        if feature_key not in cache or "records" not in cache:
            raise ValueError(f"Invalid IMU feature cache: {cache_path}")
        self.features = torch.as_tensor(cache[feature_key]).float()
        records = cache["records"]
        if len(records) != self.features.shape[0] or self.features.ndim != 2:
            raise ValueError(f"Feature/record mismatch in {cache_path}")
        self.lookup = {str(row["sample_name"]): index for index, row in enumerate(records)}
        if len(self.lookup) != len(records):
            raise ValueError(f"Duplicate sample_name in {cache_path}")
        rows = read_jsonl(manifest_path)
        missing = [str(row["sample_name"]) for row in rows if str(row["sample_name"]) not in self.lookup]
        if missing:
            raise KeyError(f"Cache misses {len(missing)} samples, e.g. {missing[:5]}")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((str(row["participant"]), str(row["run"])), []).append(row)
        examples: list[HistoryExample] = []
        for run_rows in grouped.values():
            run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
            for position, row in enumerate(run_rows):
                examples.append(HistoryExample(
                    current=str(row["sample_name"]),
                    history=tuple(str(value["sample_name"]) for value in run_rows[:position]),
                    row=row,
                ))
        self.examples = sorted(examples, key=lambda item: (
            str(item.row["participant"]), str(item.row["run"]), int(item.row["annotation_row_index"])
        ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        history = [self.features[self.lookup[name]] for name in example.history]
        row = example.row
        return {
            "current_feature": self.features[self.lookup[example.current]],
            "history_features": history,
            "history_position_ids": torch.arange(len(history), 0, -1, dtype=torch.long),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "row": row,
        }


def collate_history(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_history = max(len(row["history_features"]) for row in batch)
    feature_dim = int(batch[0]["current_feature"].shape[-1])
    history = torch.zeros((size, max_history, feature_dim), dtype=torch.float32)
    positions = torch.zeros((size, max_history), dtype=torch.long)
    padding = torch.ones((size, max_history), dtype=torch.bool)
    for index, row in enumerate(batch):
        length = len(row["history_features"])
        if length:
            history[index, :length] = torch.stack(row["history_features"])
            positions[index, :length] = row["history_position_ids"]
            padding[index, :length] = False
    return {
        "current_feature": torch.stack([row["current_feature"] for row in batch]),
        "history_features": history,
        "history_position_ids": positions,
        "history_padding_mask": padding,
        "node_target": torch.tensor([row["node_target"] for row in batch]),
        "tier3_target": torch.tensor([row["tier3_target"] for row in batch]),
        "stage_id": torch.tensor([row["stage_id"] for row in batch]),
        "rows": [row["row"] for row in batch],
    }
