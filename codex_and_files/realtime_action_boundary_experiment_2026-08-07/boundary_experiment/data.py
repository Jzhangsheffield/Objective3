from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .features import load_feature_cache


class BoundaryChunkDataset(Dataset):
    def __init__(self, cache_root: str | Path, run_names: list[str], chunk_length: int, chunk_overlap: int):
        self.cache_root = Path(cache_root)
        self.chunk_length = int(chunk_length)
        step = self.chunk_length - int(chunk_overlap)
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_length")
        self.caches = {name: load_feature_cache(self.cache_root / f"{name}.pt") for name in run_names}
        self.index: list[tuple[str, int, int]] = []
        for name in run_names:
            length = int(self.caches[name]["features"].shape[0])
            for start in range(0, length, step):
                end = min(length, start + self.chunk_length)
                self.index.append((name, start, end))
                if end == length:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, start, end = self.index[index]
        cache = self.caches[name]
        return {
            "sample_name": name,
            "start_offset": start,
            "features": cache["features"][start:end].float(),
            "state": cache["state"][start:end].long(),
            "start": cache["start"][start:end].float(),
            "end": cache["end"][start:end].float(),
        }


def collate_chunks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(row["features"].shape[0] for row in rows)
    feature_dim = rows[0]["features"].shape[1]
    batch = len(rows)
    features = torch.zeros(batch, max_len, feature_dim)
    state = torch.zeros(batch, max_len, dtype=torch.long)
    start = torch.zeros(batch, max_len)
    end = torch.zeros(batch, max_len)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, row in enumerate(rows):
        length = row["features"].shape[0]
        features[i, :length] = row["features"]
        state[i, :length] = row["state"]
        start[i, :length] = row["start"]
        end[i, :length] = row["end"]
        mask[i, :length] = True
    return {
        "features": features,
        "state": state,
        "start": start,
        "end": end,
        "mask": mask,
        "sample_name": [row["sample_name"] for row in rows],
        "start_offset": [row["start_offset"] for row in rows],
    }
