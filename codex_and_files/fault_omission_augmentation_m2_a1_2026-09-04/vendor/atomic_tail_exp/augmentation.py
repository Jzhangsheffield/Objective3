from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .graph import TaskGraph, is_graph_valid, randomized_graph_valid_history


@dataclass(frozen=True)
class TailDecision:
    node_ids: tuple[int, ...]
    reason: str

    @property
    def applied(self) -> bool:
        return bool(self.node_ids)


@dataclass(frozen=True)
class AugmentationResult:
    rows: tuple[dict[str, Any], ...]
    decision: TailDecision
    changed: bool
    normalized_kendall_distance: float


def stable_seed(base_seed: int, refresh_round: int, sample_name: str, stream: str = "main") -> int:
    digest = hashlib.sha256(
        f"atomic-tail-a0-a8:{base_seed}:{refresh_round}:{sample_name}:{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def select_active_tail(history_rows: list[dict[str, Any]], graph: TaskGraph) -> TailDecision:
    """Detect an incomplete atomic prefix without observing the current target."""
    if not history_rows:
        return TailDecision((), "empty_history")
    actual_nodes = [int(row["node_idx"]) for row in history_rows]
    if len(set(actual_nodes)) != len(actual_nodes):
        return TailDecision((), "repeated_node_fallback")
    latest = actual_nodes[-1]
    candidates = [sequence for sequence in graph.atomic_sequences if latest in sequence]
    if not candidates:
        return TailDecision((), "latest_node_not_atomic")
    reasons: list[str] = []
    observed = set(actual_nodes)
    for sequence in candidates:
        expected = tuple(sequence[: sequence.index(latest) + 1])
        sequence_set = set(sequence)
        seen = tuple(node for node in actual_nodes if node in sequence_set)
        if seen != expected:
            reasons.append("observed_atomic_nodes_not_prefix")
            continue
        if len(expected) == len(sequence):
            reasons.append("atomic_sequence_complete")
            continue
        tail_set = set(expected)
        remaining = observed - tail_set
        conflict = any(tail_set.intersection(graph.all_must_previous[node]) for node in remaining)
        if conflict:
            reasons.append("tail_dependency_conflict")
            continue
        return TailDecision(expected, "active_incomplete_atomic_prefix")
    for reason in ("atomic_sequence_complete", "observed_atomic_nodes_not_prefix", "tail_dependency_conflict"):
        if reason in reasons:
            return TailDecision((), reason)
    return TailDecision((), "no_eligible_atomic_tail")


def normalized_kendall_distance(actual_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> float:
    if len(actual_rows) <= 1:
        return 0.0
    actual_position = {str(row["sample_name"]): index for index, row in enumerate(actual_rows)}
    permutation = [actual_position[str(row["sample_name"])] for row in candidate_rows]
    inversions = sum(
        1
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
        if permutation[left] > permutation[right]
    )
    maximum = len(permutation) * (len(permutation) - 1) / 2
    return float(inversions / maximum) if maximum else 0.0


def changed_positions(actual_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> int:
    return sum(
        str(left["sample_name"]) != str(right["sample_name"])
        for left, right in zip(actual_rows, candidate_rows)
    )


class TransitionModel:
    """First-order transition prior estimated only from the outer training fold."""

    def __init__(self, counts: dict[int, Counter[int]], num_nodes: int, laplace: float = 0.5) -> None:
        self.counts = counts
        self.num_nodes = int(num_nodes)
        self.laplace = float(laplace)

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], num_nodes: int, laplace: float = 0.5) -> "TransitionModel":
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["participant"]), str(row["run"]))].append(row)
        counts: dict[int, Counter[int]] = defaultdict(Counter)
        for run_rows in grouped.values():
            run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
            nodes = [int(row["node_idx"]) for row in run_rows]
            for previous, current in zip(nodes, nodes[1:]):
                counts[previous][current] += 1
        return cls(dict(counts), num_nodes, laplace)

    def log_score(self, rows: list[dict[str, Any]]) -> float:
        nodes = [int(row["node_idx"]) for row in rows]
        score = 0.0
        for previous, current in zip(nodes, nodes[1:]):
            outgoing = self.counts.get(previous, Counter())
            denominator = sum(outgoing.values()) + self.laplace * self.num_nodes
            probability = (outgoing[current] + self.laplace) / denominator
            score += math.log(max(probability, 1e-12))
        return score


def _weighted_choice(candidates: list[list[dict[str, Any]]], scores: list[float], temperature: float, rng: random.Random) -> list[dict[str, Any]]:
    scaled = [score / max(float(temperature), 1e-6) for score in scores]
    maximum = max(scaled)
    weights = [math.exp(score - maximum) for score in scaled]
    return rng.choices(candidates, weights=weights, k=1)[0]


def augment_history(
    history_rows: list[dict[str, Any]],
    graph: TaskGraph,
    seed: int,
    active_tail_only: bool,
    sampling: str,
    transition_model: TransitionModel | None,
    candidate_count: int,
    temperature: float,
    max_distance: float,
    min_changed: int,
    preserve_latest_non_tail: int,
) -> AugmentationResult:
    actual = list(history_rows)
    decision = select_active_tail(actual, graph)
    rng = random.Random(seed)
    if decision.reason == "repeated_node_fallback":
        return AugmentationResult(tuple(actual), decision, False, 0.0)
    if not decision.applied and active_tail_only:
        return AugmentationResult(tuple(actual), decision, False, 0.0)

    tail_set = set(decision.node_ids)
    remaining = [row for row in actual if int(row["node_idx"]) not in tail_set]
    row_by_node = {int(row["node_idx"]): row for row in actual}
    tail = [row_by_node[node] for node in decision.node_ids]

    if sampling in {"none"}:
        selected_remaining = remaining
    elif sampling == "uniform":
        selected_remaining = randomized_graph_valid_history(remaining, graph, rng)
    elif sampling == "plausibility_weighted":
        if transition_model is None:
            raise ValueError("plausibility_weighted sampling requires a TransitionModel")
        candidates: list[list[dict[str, Any]]] = []
        scores: list[float] = []
        seen: set[tuple[str, ...]] = set()
        for candidate_index in range(max(1, int(candidate_count))):
            candidate_rng = random.Random(rng.getrandbits(64) + candidate_index)
            shuffled = randomized_graph_valid_history(remaining, graph, candidate_rng)
            combined = shuffled + tail
            key = tuple(str(row["sample_name"]) for row in combined)
            if key in seen:
                continue
            seen.add(key)
            if preserve_latest_non_tail > 0 and len(remaining) >= preserve_latest_non_tail:
                if [str(row["sample_name"]) for row in shuffled[-preserve_latest_non_tail:]] != [
                    str(row["sample_name"]) for row in remaining[-preserve_latest_non_tail:]
                ]:
                    continue
            distance = normalized_kendall_distance(actual, combined)
            if distance > float(max_distance):
                continue
            if changed_positions(actual, combined) < int(min_changed):
                continue
            candidates.append(shuffled)
            scores.append(transition_model.log_score(combined))
        selected_remaining = (
            _weighted_choice(candidates, scores, temperature, rng) if candidates else remaining
        )
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling}")

    combined = selected_remaining + tail
    if not is_graph_valid(combined, graph):
        raise RuntimeError("Augmenter produced an invalid graph order")
    distance = normalized_kendall_distance(actual, combined)
    return AugmentationResult(tuple(combined), decision, combined != actual, distance)


def corrupt_atomic_tail(valid_rows: list[dict[str, Any]], decision: TailDecision) -> tuple[list[dict[str, Any]], bool]:
    """Create a hard negative by swapping two final atomic-prefix elements."""
    if len(decision.node_ids) < 2:
        return list(valid_rows), False
    corrupted = list(valid_rows)
    corrupted[-1], corrupted[-2] = corrupted[-2], corrupted[-1]
    return corrupted, True
