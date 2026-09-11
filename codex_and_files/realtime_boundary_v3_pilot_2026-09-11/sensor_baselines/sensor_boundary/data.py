from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import read_json, safe_torch_load


class SensorChunkDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        run_names: list[str],
        chunk_length: int,
        chunk_overlap: int,
        max_cached_runs: int = 2,
    ):
        self.cache_root = Path(cache_root)
        self.chunk_length = int(chunk_length)
        self.chunk_overlap = int(chunk_overlap)
        if self.chunk_overlap < 0: raise ValueError("negative overlap")
        self.max_cached_runs = int(max_cached_runs)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        step = self.chunk_length - int(chunk_overlap)
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_length")
        self.index: list[tuple[str, int, int]] = []
        for name in run_names:
            meta = read_json(self.cache_root / f"{name}.json")
            length = int(meta["decision_steps"])
            for start in range(0, length, step):
                end = min(length, start + self.chunk_length)
                self.index.append((name, start, end))
                if end == length:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, name: str) -> dict[str, Any]:
        if name in self._cache:
            value = self._cache.pop(name)
            self._cache[name] = value
            return value
        value = safe_torch_load(self.cache_root / f"{name}.pt")
        self._cache[name] = value
        while len(self._cache) > self.max_cached_runs:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, start, end = self.index[index]
        cache = self._load(name)
        return {
            "sample_name": name,
            "start_offset": start,
            "context_steps": 0 if start == 0 else min(self.chunk_overlap, end-start),
            "windows": cache["windows"][start:end].float(),
            "state": cache["state"][start:end].long(),
            "start": cache["start"][start:end].float(),
            "end": cache["end"][start:end].float(),
        }


def collate_chunks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(row["windows"].shape[0] for row in rows)
    width, channels = rows[0]["windows"].shape[1:]
    batch = len(rows)
    windows = torch.zeros(batch, max_len, width, channels)
    state = torch.zeros(batch, max_len, dtype=torch.long)
    start = torch.zeros(batch, max_len)
    end = torch.zeros(batch, max_len)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, row in enumerate(rows):
        length = row["windows"].shape[0]
        windows[i, :length] = row["windows"]
        state[i, :length] = row["state"]
        start[i, :length] = row["start"]
        end[i, :length] = row["end"]
        mask[i, int(row.get("context_steps", 0)):length] = True
    return {
        "windows": windows,
        "state": state,
        "start": start,
        "end": end,
        "mask": mask,
        "sample_name": [row["sample_name"] for row in rows],
        "start_offset": [row["start_offset"] for row in rows],
    }
