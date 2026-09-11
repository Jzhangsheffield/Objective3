from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .constants import DEFAULT_CAMERA_ID, DEFAULT_RGB_MEAN, DEFAULT_RGB_STD
from .graph import TaskGraphSpec, randomized_graph_valid_history, stable_sample_seed
from .utils import read_jsonl, resolve_manifest, run_key


def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def uniform_frame_indices(num_frames: int, target_frames: int) -> list[int]:
    if num_frames <= 0 or target_frames <= 0:
        raise ValueError(f"Invalid frame counts: num_frames={num_frames}, target={target_frames}")
    return np.linspace(0, num_frames - 1, target_frames).astype(np.int64).tolist()


class RGBVideoTransform:
    def __init__(
        self,
        train: bool,
        size: int = 224,
        mean: tuple[float, float, float] = DEFAULT_RGB_MEAN,
        std: tuple[float, float, float] = DEFAULT_RGB_STD,
    ) -> None:
        if train:
            self.transform = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        size=(size, size),
                        scale=(0.6, 1.0),
                        ratio=(0.75, 1.3333333333),
                        antialias=True,
                    ),
                    v2.RandomHorizontalFlip(p=0.0),
                    v2.RandomVerticalFlip(p=0.0),
                    v2.RandomApply(
                        [v2.ColorJitter(brightness=0.24, contrast=0.24, saturation=0.24, hue=0.16)],
                        p=0.5,
                    ),
                    v2.RandomGrayscale(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(size, size), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, video_tchw: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.transform(tv_tensors.Video(video_tchw))).contiguous()


class RGBClipDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        camera_id: str = DEFAULT_CAMERA_ID,
        n_frames: int = 16,
        rgb_size: int = 224,
        train: bool = False,
        verify_paths: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = resolve_manifest(self.dataset_root, manifest)
        self.camera_id = str(camera_id)
        self.n_frames = int(n_frames)
        self.rows = read_jsonl(self.manifest_path)
        self.transform = RGBVideoTransform(train=train, size=rgb_size)
        if verify_paths:
            missing: list[str] = []
            field = f"{self.camera_id}_rgb"
            for row in self.rows:
                rel = row.get(field)
                if not rel or not (self.dataset_root / str(rel)).is_file():
                    missing.append(str(row.get("sample_name", "<unknown>")))
                    if len(missing) >= 10:
                        break
            if missing:
                raise FileNotFoundError(
                    f"Missing camera {self.camera_id} RGB tensors for examples: {missing}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel_path = row[f"{self.camera_id}_rgb"]
        obj = safe_torch_load(self.dataset_root / rel_path)
        video = obj["frames"] if isinstance(obj, dict) else obj
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(
                f"Invalid RGB tensor for {row['sample_name']}: {type(video)} {getattr(video, 'shape', None)}"
            )
        indices = uniform_frame_indices(int(video.shape[0]), self.n_frames)
        video = self.transform(video[indices])
        # ResNet3D input is [C,T,H,W] per sample.
        video = video.permute(1, 0, 2, 3).contiguous()
        return {
            "video": video,
            "tier3_target": int(row["tier3_id"]),
            "node_target": int(row["node_idx"]) - 1,
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
        }


@dataclass(frozen=True)
class HistoryExample:
    current_cache_index: int
    history_cache_indices: tuple[int, ...]
    current_row: dict[str, Any]
    history_rows: tuple[dict[str, Any], ...]


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    cache = safe_torch_load(path)
    required = {"features", "tier3_logits", "records", "metadata"}
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"Feature cache {path} does not contain {sorted(required)}")
    if len(cache["records"]) != int(cache["features"].shape[0]):
        raise ValueError("Feature cache record/feature count mismatch")
    return cache


class FeatureHistoryDataset(Dataset):
    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        history_order: str,
        graph: TaskGraphSpec | None = None,
        shuffle_seed: int = 1,
    ) -> None:
        if history_order not in {"actual", "graph_valid"}:
            raise ValueError(f"Unsupported history_order: {history_order}")
        self.cache = load_feature_cache(feature_cache_path)
        self.features: torch.Tensor = self.cache["features"].float()
        self.selection_rows = read_jsonl(selection_manifest)
        self.history_order = history_order
        self.graph = graph
        self.shuffle_seed = int(shuffle_seed)
        if history_order == "graph_valid" and graph is None:
            raise ValueError("graph_valid history requires a TaskGraphSpec")

        cache_lookup = {
            str(row["sample_name"]): index
            for index, row in enumerate(self.cache["records"])
        }
        missing = [
            str(row["sample_name"])
            for row in self.selection_rows
            if str(row["sample_name"]) not in cache_lookup
        ]
        if missing:
            raise KeyError(f"Selection manifest samples missing from feature cache: {missing[:10]}")

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.selection_rows:
            grouped.setdefault(run_key(row), []).append(row)

        self.examples: list[HistoryExample] = []
        for rows in grouped.values():
            rows.sort(key=lambda row: int(row["annotation_row_index"]))
            for current_position, current_row in enumerate(rows):
                actual_history = list(rows[:current_position])
                if history_order == "graph_valid":
                    actual_history = randomized_graph_valid_history(
                        actual_history,
                        graph=self.graph,
                        seed=stable_sample_seed(self.shuffle_seed, str(current_row["sample_name"])),
                    )
                self.examples.append(
                    HistoryExample(
                        current_cache_index=cache_lookup[str(current_row["sample_name"])],
                        history_cache_indices=tuple(
                            cache_lookup[str(row["sample_name"])] for row in actual_history
                        ),
                        current_row=current_row,
                        history_rows=tuple(actual_history),
                    )
                )
        self.examples.sort(
            key=lambda example: (
                str(example.current_row["participant"]),
                str(example.current_row["run"]),
                int(example.current_row["annotation_row_index"]),
            )
        )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if example.history_cache_indices:
            history_indices = torch.tensor(example.history_cache_indices, dtype=torch.long)
            history_features = self.features.index_select(0, history_indices)
        else:
            history_features = self.features.new_zeros((0, self.feature_dim))
        length = len(example.history_rows)
        # Position IDs encode distance in the presented sequence; 1 means most recent.
        position_ids = torch.arange(length, 0, -1, dtype=torch.long)
        history_node_classes = torch.tensor(
            [int(row["node_idx"]) - 1 for row in example.history_rows], dtype=torch.long
        )
        row = example.current_row
        return {
            "current_feature": self.features[example.current_cache_index],
            "history_features": history_features,
            "history_position_ids": position_ids,
            "history_node_classes": history_node_classes,
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
            "history_sample_names": [str(hist["sample_name"]) for hist in example.history_rows],
        }


def collate_history_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    feature_dim = int(batch[0]["current_feature"].shape[0])
    max_history = max(int(item["history_features"].shape[0]) for item in batch)
    history_features = torch.zeros((batch_size, max_history, feature_dim), dtype=torch.float32)
    history_positions = torch.zeros((batch_size, max_history), dtype=torch.long)
    history_nodes = torch.full((batch_size, max_history), -1, dtype=torch.long)
    history_mask = torch.ones((batch_size, max_history), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = int(item["history_features"].shape[0])
        if length:
            history_features[row_index, :length] = item["history_features"]
            history_positions[row_index, :length] = item["history_position_ids"]
            history_nodes[row_index, :length] = item["history_node_classes"]
            history_mask[row_index, :length] = False
    return {
        "current_feature": torch.stack([item["current_feature"] for item in batch]),
        "history_features": history_features,
        "history_position_ids": history_positions,
        "history_node_classes": history_nodes,
        "history_padding_mask": history_mask,
        "node_target": torch.tensor([item["node_target"] for item in batch], dtype=torch.long),
        "tier3_target": torch.tensor([item["tier3_target"] for item in batch], dtype=torch.long),
        "stage_id": torch.tensor([item["stage_id"] for item in batch], dtype=torch.long),
        "sample_name": [item["sample_name"] for item in batch],
        "participant": [item["participant"] for item in batch],
        "run": [item["run"] for item in batch],
        "annotation_row_index": [item["annotation_row_index"] for item in batch],
        "history_sample_names": [item["history_sample_names"] for item in batch],
    }

