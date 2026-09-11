from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .data import FeatureHistoryDataset
from .graph import TaskGraphSpec, randomized_graph_valid_history


def stable_epoch_sample_seed(base_seed: int, epoch: int, sample_name: str) -> int:
    """Derive a reproducible, sample-specific graph shuffle seed for one epoch."""
    if int(epoch) < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    digest = hashlib.sha256(
        f"{int(base_seed)}:{int(epoch)}:{sample_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class EpochGraphValidHistoryDataset(FeatureHistoryDataset):
    """Causal history dataset that resamples one graph-valid order per training epoch.

    The parent dataset is deliberately constructed with ``actual`` order, so the
    existing static ``FeatureHistoryDataset`` behavior remains untouched.  The
    epoch-dependent reorder happens only in this new class.
    """

    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        graph: TaskGraphSpec,
        shuffle_seed: int = 1,
    ) -> None:
        super().__init__(
            feature_cache_path=feature_cache_path,
            selection_manifest=selection_manifest,
            history_order="actual",
            graph=graph,
            shuffle_seed=shuffle_seed,
        )
        self.graph = graph
        self.base_shuffle_seed = int(shuffle_seed)
        self.epoch = 1
        self.cache_lookup = {
            str(row["sample_name"]): index
            for index, row in enumerate(self.cache["records"])
        }

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 1:
            raise ValueError(f"Epoch must be >= 1, got {epoch}")
        self.epoch = int(epoch)

    def ordered_history_rows(self, index: int, epoch: int | None = None) -> list[dict[str, Any]]:
        example = self.examples[index]
        selected_epoch = self.epoch if epoch is None else int(epoch)
        return randomized_graph_valid_history(
            list(example.history_rows),
            graph=self.graph,
            seed=stable_epoch_sample_seed(
                self.base_shuffle_seed,
                selected_epoch,
                str(example.current_row["sample_name"]),
            ),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        history_rows = self.ordered_history_rows(index)
        if history_rows:
            history_indices = torch.tensor(
                [self.cache_lookup[str(row["sample_name"])] for row in history_rows],
                dtype=torch.long,
            )
            history_features = self.features.index_select(0, history_indices)
        else:
            history_features = self.features.new_zeros((0, self.feature_dim))

        length = len(history_rows)
        position_ids = torch.arange(length, 0, -1, dtype=torch.long)
        history_node_classes = torch.tensor(
            [int(row["node_idx"]) - 1 for row in history_rows], dtype=torch.long
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
            "history_sample_names": [str(hist["sample_name"]) for hist in history_rows],
        }

    def audit_epochs(self, epochs: int) -> dict[str, Any]:
        """Summarize how often epoch-wise resampling can actually change history."""
        if int(epochs) < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        total = len(self.examples)
        short_history = 0
        repeated_node_fallback = 0
        samples_with_multiple_orders = 0
        samples_ever_different_from_actual = 0
        per_epoch_different_from_actual = [0 for _ in range(int(epochs))]
        per_epoch_changed_from_previous = [0 for _ in range(int(epochs))]

        for index, example in enumerate(self.examples):
            actual_rows = list(example.history_rows)
            actual_names = tuple(str(row["sample_name"]) for row in actual_rows)
            if len(actual_rows) <= 1:
                short_history += 1
            node_indices = [int(row["node_idx"]) for row in actual_rows]
            if len(node_indices) > 1 and len(set(node_indices)) != len(node_indices):
                repeated_node_fallback += 1

            sampled_orders: list[tuple[str, ...]] = []
            previous: tuple[str, ...] | None = None
            for epoch in range(1, int(epochs) + 1):
                current = tuple(
                    str(row["sample_name"])
                    for row in self.ordered_history_rows(index, epoch=epoch)
                )
                sampled_orders.append(current)
                if current != actual_names:
                    per_epoch_different_from_actual[epoch - 1] += 1
                if previous is not None and current != previous:
                    per_epoch_changed_from_previous[epoch - 1] += 1
                previous = current

            unique_orders = set(sampled_orders)
            if len(unique_orders) > 1:
                samples_with_multiple_orders += 1
            if any(order != actual_names for order in unique_orders):
                samples_ever_different_from_actual += 1

        denominator = max(1, total)
        return {
            "policy": "graph_valid_epoch_shuffle",
            "seed_formula": "sha256(base_seed:epoch:sample_name)",
            "base_seed": self.base_shuffle_seed,
            "epochs_audited": int(epochs),
            "total_examples": total,
            "history_length_le_one": short_history,
            "history_length_le_one_fraction": short_history / denominator,
            "repeated_node_fallback": repeated_node_fallback,
            "repeated_node_fallback_fraction": repeated_node_fallback / denominator,
            "samples_with_multiple_orders": samples_with_multiple_orders,
            "samples_with_multiple_orders_fraction": (
                samples_with_multiple_orders / denominator
            ),
            "samples_ever_different_from_actual": samples_ever_different_from_actual,
            "samples_ever_different_from_actual_fraction": (
                samples_ever_different_from_actual / denominator
            ),
            "per_epoch_different_from_actual": per_epoch_different_from_actual,
            "per_epoch_changed_from_previous": per_epoch_changed_from_previous,
        }
