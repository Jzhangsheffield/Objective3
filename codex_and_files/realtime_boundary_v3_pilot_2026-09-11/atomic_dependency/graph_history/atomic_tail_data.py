from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_data import EpochGraphValidHistoryDataset
from .graph import TaskGraphSpec, randomized_graph_valid_history


@dataclass(frozen=True)
class AtomicTailDecision:
    node_indices: tuple[int, ...]
    reason: str

    @property
    def applied(self) -> bool:
        return bool(self.node_indices)


def normalize_refresh_interval(value: str | int) -> int | None:
    """Return a positive epoch interval, or ``None`` for one order per run."""
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"Refresh interval must be >= 1, got {value}")
        return value
    normalized = str(value).strip().lower()
    if normalized == "once":
        return None
    try:
        interval = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Refresh interval must be a positive integer or 'once', got {value!r}"
        ) from exc
    if interval < 1:
        raise ValueError(f"Refresh interval must be >= 1, got {interval}")
    return interval


def refresh_policy_label(value: str | int | None) -> str:
    interval = None if value is None else normalize_refresh_interval(value)
    return "refresh_once" if interval is None else f"refresh_every_{interval}"


def refresh_round_for_epoch(epoch: int, interval: int | None) -> int:
    if int(epoch) < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    return 0 if interval is None else (int(epoch) - 1) // interval


def stable_atomic_tail_seed(
    base_seed: int,
    refresh_round: int,
    sample_name: str,
) -> int:
    digest = hashlib.sha256(
        (
            f"atomic_tail:{int(base_seed)}:{int(refresh_round)}:"
            f"{sample_name}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def select_atomic_tail(
    history_rows: list[dict[str, Any]],
    graph: TaskGraphSpec,
) -> AtomicTailDecision:
    """Select the active incomplete atomic prefix using history alone.

    The current target is deliberately not an input.  A tail is eligible only when
    the latest real history node completes a proper prefix of one atomic sequence.
    Completed sequences and non-prefix observations are not moved to the end.
    """
    if not history_rows:
        return AtomicTailDecision((), "empty_history")

    actual_nodes = [int(row["node_idx"]) for row in history_rows]
    if len(set(actual_nodes)) != len(actual_nodes):
        return AtomicTailDecision((), "repeated_node_fallback")

    latest_node = actual_nodes[-1]
    candidate_sequences = [
        sequence for sequence in graph.atomic_sequences if latest_node in sequence
    ]
    if not candidate_sequences:
        return AtomicTailDecision((), "latest_node_not_atomic")

    observed_set = set(actual_nodes)
    reasons: list[str] = []
    for sequence in candidate_sequences:
        latest_position = sequence.index(latest_node)
        expected_prefix = tuple(sequence[: latest_position + 1])
        observed_in_sequence = tuple(
            node_idx for node_idx in actual_nodes if node_idx in set(sequence)
        )
        if observed_in_sequence != expected_prefix:
            reasons.append("observed_atomic_nodes_not_prefix")
            continue
        if len(expected_prefix) == len(sequence):
            reasons.append("atomic_sequence_complete")
            continue

        tail_set = set(expected_prefix)
        remaining_nodes = observed_set - tail_set
        dependency_conflict = any(
            tail_set.intersection(graph.all_must_previous[node_idx])
            for node_idx in remaining_nodes
        )
        if dependency_conflict:
            reasons.append("tail_dependency_conflict")
            continue
        return AtomicTailDecision(expected_prefix, "active_incomplete_atomic_prefix")

    reason_priority = (
        "atomic_sequence_complete",
        "observed_atomic_nodes_not_prefix",
        "tail_dependency_conflict",
    )
    for reason in reason_priority:
        if reason in reasons:
            return AtomicTailDecision((), reason)
    return AtomicTailDecision((), "no_eligible_atomic_tail")


def atomic_tail_graph_valid_history(
    history_rows: list[dict[str, Any]],
    graph: TaskGraphSpec,
    seed: int,
) -> tuple[list[dict[str, Any]], AtomicTailDecision]:
    """Randomize legal history while anchoring an active atomic prefix at the end."""
    decision = select_atomic_tail(history_rows, graph)
    if decision.reason == "repeated_node_fallback":
        return list(history_rows), decision
    if not decision.applied:
        return (
            randomized_graph_valid_history(
                list(history_rows),
                graph=graph,
                seed=seed,
            ),
            decision,
        )

    row_by_node = {int(row["node_idx"]): row for row in history_rows}
    tail_set = set(decision.node_indices)
    remaining_rows = [
        row for row in history_rows if int(row["node_idx"]) not in tail_set
    ]
    reordered_rows = randomized_graph_valid_history(
        remaining_rows,
        graph=graph,
        seed=seed,
    )
    tail_rows = [row_by_node[node_idx] for node_idx in decision.node_indices]
    return reordered_rows + tail_rows, decision


class AtomicTailGraphValidHistoryDataset(EpochGraphValidHistoryDataset):
    """Graph-valid history with an atomic tail and configurable refresh cadence."""

    def __init__(
        self,
        feature_cache_path: str | Path,
        selection_manifest: str | Path,
        graph: TaskGraphSpec,
        shuffle_seed: int = 1,
        refresh_interval: str | int = 1,
    ) -> None:
        super().__init__(
            feature_cache_path=feature_cache_path,
            selection_manifest=selection_manifest,
            graph=graph,
            shuffle_seed=shuffle_seed,
        )
        self.refresh_interval = normalize_refresh_interval(refresh_interval)
        self.refresh_policy = refresh_policy_label(self.refresh_interval)

    def refresh_round(self, epoch: int | None = None) -> int:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        return refresh_round_for_epoch(selected_epoch, self.refresh_interval)

    def ordered_history_rows(
        self,
        index: int,
        epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        example = self.examples[index]
        selected_round = self.refresh_round(epoch)
        rows, _ = atomic_tail_graph_valid_history(
            list(example.history_rows),
            graph=self.graph,
            seed=stable_atomic_tail_seed(
                self.base_shuffle_seed,
                selected_round,
                str(example.current_row["sample_name"]),
            ),
        )
        return rows

    def audit_epochs(self, epochs: int) -> dict[str, Any]:
        if int(epochs) < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        epoch_rounds = [
            self.refresh_round(epoch) for epoch in range(1, int(epochs) + 1)
        ]
        unique_rounds = list(dict.fromkeys(epoch_rounds))
        reason_counts: Counter[str] = Counter()
        tail_length_counts: Counter[int] = Counter()
        samples_with_multiple_orders = 0
        samples_ever_different_from_actual = 0
        atomic_tail_violations = 0
        per_refresh_different_from_actual = [0 for _ in unique_rounds]
        per_refresh_changed_from_previous = [0 for _ in unique_rounds]

        for index, example in enumerate(self.examples):
            actual_rows = list(example.history_rows)
            actual_names = tuple(str(row["sample_name"]) for row in actual_rows)
            decision = select_atomic_tail(actual_rows, self.graph)
            reason_counts[decision.reason] += 1
            tail_length_counts[len(decision.node_indices)] += 1

            sampled_orders: list[tuple[str, ...]] = []
            previous: tuple[str, ...] | None = None
            tail_names = tuple(
                str(row["sample_name"])
                for row in actual_rows
                if int(row["node_idx"]) in set(decision.node_indices)
            )
            for round_index, refresh_round in enumerate(unique_rounds):
                rows, _ = atomic_tail_graph_valid_history(
                    actual_rows,
                    graph=self.graph,
                    seed=stable_atomic_tail_seed(
                        self.base_shuffle_seed,
                        refresh_round,
                        str(example.current_row["sample_name"]),
                    ),
                )
                current = tuple(str(row["sample_name"]) for row in rows)
                sampled_orders.append(current)
                if current != actual_names:
                    per_refresh_different_from_actual[round_index] += 1
                if previous is not None and current != previous:
                    per_refresh_changed_from_previous[round_index] += 1
                if tail_names and current[-len(tail_names) :] != tail_names:
                    atomic_tail_violations += 1
                previous = current

            if len(set(sampled_orders)) > 1:
                samples_with_multiple_orders += 1
            if any(order != actual_names for order in sampled_orders):
                samples_ever_different_from_actual += 1

        total = len(self.examples)
        denominator = max(1, total)
        applied = reason_counts["active_incomplete_atomic_prefix"]
        return {
            "policy": "atomic_tail_graph_valid",
            "refresh_policy": self.refresh_policy,
            "refresh_interval_epochs": self.refresh_interval,
            "seed_formula": (
                "sha256(atomic_tail:base_seed:refresh_round:sample_name)"
            ),
            "base_seed": self.base_shuffle_seed,
            "epochs_audited": int(epochs),
            "epoch_to_refresh_round": epoch_rounds,
            "unique_refresh_rounds": unique_rounds,
            "total_examples": total,
            "atomic_tail_applied": applied,
            "atomic_tail_applied_fraction": applied / denominator,
            "decision_reason_counts": dict(sorted(reason_counts.items())),
            "tail_length_counts": {
                str(length): count
                for length, count in sorted(tail_length_counts.items())
            },
            "atomic_tail_violations": atomic_tail_violations,
            "samples_with_multiple_orders": samples_with_multiple_orders,
            "samples_with_multiple_orders_fraction": (
                samples_with_multiple_orders / denominator
            ),
            "samples_ever_different_from_actual": samples_ever_different_from_actual,
            "samples_ever_different_from_actual_fraction": (
                samples_ever_different_from_actual / denominator
            ),
            "per_refresh_different_from_actual": (
                per_refresh_different_from_actual
            ),
            "per_refresh_changed_from_previous": (
                per_refresh_changed_from_previous
            ),
        }
