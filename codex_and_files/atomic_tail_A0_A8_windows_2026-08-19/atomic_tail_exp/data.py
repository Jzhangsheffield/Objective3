from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .augmentation import (
    TransitionModel,
    augment_history,
    corrupt_atomic_tail,
    stable_seed,
)
from .graph import TaskGraph


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
    return rows


def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    cache = safe_torch_load(path)
    required = {"features", "tier3_logits", "records", "metadata"}
    if not isinstance(cache, dict) or not required.issubset(cache):
        raise ValueError(f"Feature cache {path} lacks required keys {sorted(required)}")
    if len(cache["records"]) != int(cache["features"].shape[0]):
        raise ValueError(f"Feature/record length mismatch in {path}")
    return cache


@dataclass(frozen=True)
class HistoryExample:
    current_cache_index: int
    current_row: dict[str, Any]
    history_rows: tuple[dict[str, Any], ...]


class MultiViewHistoryDataset(Dataset):
    """Returns actual, atomic-tail and optional corrupted-tail histories."""

    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        graph: TaskGraph,
        experiment: dict[str, Any],
        augmentation_config: dict[str, Any],
        seed: int,
        training: bool,
    ) -> None:
        self.cache = load_feature_cache(feature_cache_path)
        self.features = self.cache["features"].float()
        self.rows = read_jsonl(selection_manifest)
        self.graph = graph
        self.experiment = experiment
        self.augmentation_config = augmentation_config
        self.base_seed = int(seed)
        self.training = bool(training)
        self.epoch = 1
        self.lookup = {str(row["sample_name"]): index for index, row in enumerate(self.cache["records"])}
        missing = [str(row["sample_name"]) for row in self.rows if str(row["sample_name"]) not in self.lookup]
        if missing:
            raise KeyError(f"Manifest samples absent from cache: {missing[:10]}")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            grouped[(str(row["participant"]), str(row["run"]))].append(row)
        examples = []
        for run_rows in grouped.values():
            run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
            for index, current in enumerate(run_rows):
                examples.append(HistoryExample(self.lookup[str(current["sample_name"])], current, tuple(run_rows[:index])))
        self.examples = sorted(
            examples,
            key=lambda item: (
                str(item.current_row["participant"]),
                str(item.current_row["run"]),
                int(item.current_row["annotation_row_index"]),
            ),
        )
        self.transition_model = (
            TransitionModel.fit(
                self.rows,
                graph.num_nodes,
                float(augmentation_config["transition_laplace"]),
            )
            if self.training and experiment["sampling"] == "plausibility_weighted"
            else None
        )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _refresh_round(self) -> int:
        interval = self.augmentation_config.get("refresh_interval", "once")
        if str(interval).lower() == "once":
            return 0
        interval = max(1, int(interval))
        return (self.epoch - 1) // interval

    def _view(self, rows: list[dict[str, Any]], actual_rows: list[dict[str, Any]], position_mode: str) -> dict[str, Any]:
        if rows:
            indices = torch.tensor([self.lookup[str(row["sample_name"])] for row in rows], dtype=torch.long)
            features = self.features.index_select(0, indices)
        else:
            features = self.features.new_zeros((0, self.feature_dim))
        if position_mode == "actual_recency":
            actual_recency = {
                str(row["sample_name"]): len(actual_rows) - index
                for index, row in enumerate(actual_rows)
            }
            positions = torch.tensor([actual_recency[str(row["sample_name"])] for row in rows], dtype=torch.long)
        else:
            positions = torch.arange(len(rows), 0, -1, dtype=torch.long)
        return {
            "features": features,
            "position_ids": positions,
            "node_classes": torch.tensor([int(row["node_idx"]) - 1 for row in rows], dtype=torch.long),
            "sample_names": [str(row["sample_name"]) for row in rows],
        }

    def _make_item(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        actual_rows = list(example.history_rows)
        experiment = self.experiment
        if self.training and experiment["train_view"] != "actual":
            result = augment_history(
                actual_rows,
                self.graph,
                stable_seed(self.base_seed, self._refresh_round(), str(example.current_row["sample_name"])),
                bool(experiment["active_tail_only"]),
                str(experiment["sampling"]),
                self.transition_model,
                int(self.augmentation_config["candidate_count"]),
                float(self.augmentation_config["sampling_temperature"]),
                float(self.augmentation_config["max_normalized_kendall_distance"]),
                int(self.augmentation_config["min_changed_positions"]),
                int(self.augmentation_config["preserve_latest_non_tail"]),
            )
            augmented_rows = list(result.rows)
            decision = result.decision
            changed = result.changed
            distance = result.normalized_kendall_distance
        else:
            from .augmentation import select_active_tail

            augmented_rows = list(actual_rows)
            decision = select_active_tail(actual_rows, self.graph)
            changed = False
            distance = 0.0
        corrupted_rows, corruption_valid = corrupt_atomic_tail(augmented_rows, decision)
        row = example.current_row
        position_mode = str(experiment["position_mode"])
        return {
            "current_feature": self.features[example.current_cache_index],
            "actual": self._view(actual_rows, actual_rows, position_mode),
            "augmented": self._view(augmented_rows, actual_rows, position_mode),
            "corrupted": self._view(corrupted_rows, actual_rows, position_mode),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]),
            "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]),
            "participant": str(row["participant"]),
            "run": str(row["run"]),
            "annotation_row_index": int(row["annotation_row_index"]),
            "tail_reason": decision.reason,
            "tail_length": len(decision.node_ids),
            "augmentation_changed": changed,
            "augmentation_distance": distance,
            "corruption_valid": corruption_valid,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._make_item(index)

    def audit(self) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        tail_lengths: Counter[int] = Counter()
        changed = 0
        distances = []
        corruptible = 0
        for index in range(len(self)):
            item = self._make_item(index)
            reason_counts[item["tail_reason"]] += 1
            tail_lengths[item["tail_length"]] += 1
            changed += int(item["augmentation_changed"])
            distances.append(float(item["augmentation_distance"]))
            corruptible += int(item["corruption_valid"])
        total = max(1, len(self))
        return {
            "samples": len(self),
            "tail_reason_counts": dict(sorted(reason_counts.items())),
            "tail_length_counts": {str(key): value for key, value in sorted(tail_lengths.items())},
            "augmentation_changed": changed,
            "augmentation_changed_fraction": changed / total,
            "mean_normalized_kendall_distance": sum(distances) / total,
            "tail_aux_eligible": corruptible,
            "tail_aux_eligible_fraction": corruptible / total,
            "uses_current_target_for_reordering": False,
        }


def _collate_view(batch: list[dict[str, Any]], name: str) -> dict[str, torch.Tensor]:
    feature_dim = int(batch[0]["current_feature"].shape[0])
    max_length = max(int(item[name]["features"].shape[0]) for item in batch)
    features = torch.zeros((len(batch), max_length, feature_dim), dtype=torch.float32)
    positions = torch.zeros((len(batch), max_length), dtype=torch.long)
    nodes = torch.full((len(batch), max_length), -1, dtype=torch.long)
    padding = torch.ones((len(batch), max_length), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = int(item[name]["features"].shape[0])
        if length:
            features[row_index, :length] = item[name]["features"]
            positions[row_index, :length] = item[name]["position_ids"]
            nodes[row_index, :length] = item[name]["node_classes"]
            padding[row_index, :length] = False
    return {
        f"{name}_history_features": features,
        f"{name}_history_position_ids": positions,
        f"{name}_history_node_classes": nodes,
        f"{name}_history_padding_mask": padding,
    }


def collate_multiview_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "current_feature": torch.stack([item["current_feature"] for item in batch]),
        "node_target": torch.tensor([item["node_target"] for item in batch], dtype=torch.long),
        "tier3_target": torch.tensor([item["tier3_target"] for item in batch], dtype=torch.long),
        "stage_id": torch.tensor([item["stage_id"] for item in batch], dtype=torch.long),
        "corruption_valid": torch.tensor([item["corruption_valid"] for item in batch], dtype=torch.bool),
        "augmentation_changed": torch.tensor([item["augmentation_changed"] for item in batch], dtype=torch.bool),
        "augmentation_distance": torch.tensor([item["augmentation_distance"] for item in batch], dtype=torch.float32),
    }
    for view in ("actual", "augmented", "corrupted"):
        result.update(_collate_view(batch, view))
    for key in ("sample_name", "participant", "run", "annotation_row_index", "tail_reason", "tail_length"):
        result[key] = [item[key] for item in batch]
    return result
