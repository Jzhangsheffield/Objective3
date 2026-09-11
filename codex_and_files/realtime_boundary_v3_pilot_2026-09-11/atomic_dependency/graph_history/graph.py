from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .constants import NUM_GRAPH_NODES, RELATION_TO_ID
from .utils import read_json


@dataclass(frozen=True)
class TaskGraphSpec:
    task_graph_path: Path
    relation_matrix_path: Path
    graph_json: dict[str, Any]
    relation_json: dict[str, Any]
    relation_ids: torch.Tensor
    node_to_tier3: torch.Tensor
    node_to_stage: torch.Tensor
    all_must_previous: dict[int, tuple[int, ...]]
    immediate_previous: dict[int, int | None]
    atomic_sequences: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, task_graph_path: str | Path, relation_matrix_path: str | Path) -> "TaskGraphSpec":
        task_graph_path = Path(task_graph_path)
        relation_matrix_path = Path(relation_matrix_path)
        graph_json = read_json(task_graph_path)
        relation_json = read_json(relation_matrix_path)

        nodes = {int(node["node_idx"]): node for node in graph_json["nodes"]}
        expected = set(range(1, NUM_GRAPH_NODES + 1))
        if not expected.issubset(nodes):
            missing = sorted(expected - set(nodes))
            raise ValueError(f"Task graph is missing action nodes: {missing}")

        node_to_tier3 = torch.tensor(
            [int(nodes[idx]["action_id_tier3"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )
        node_to_stage = torch.tensor(
            [int(nodes[idx]["stage_id"]) for idx in range(1, NUM_GRAPH_NODES + 1)],
            dtype=torch.long,
        )

        columns = [int(value) for value in relation_json["column_node_idx"]]
        column_lookup = {node_idx: column for column, node_idx in enumerate(columns)}
        rows = {int(row["current_node_idx"]): row["values"] for row in relation_json["rows"]}
        relation_ids = torch.empty((NUM_GRAPH_NODES, NUM_GRAPH_NODES), dtype=torch.long)
        for current_node in range(1, NUM_GRAPH_NODES + 1):
            for previous_node in range(1, NUM_GRAPH_NODES + 1):
                code = rows[current_node][column_lookup[previous_node]]
                normalized = "X" if code == "." else str(code)
                if normalized not in RELATION_TO_ID:
                    raise ValueError(
                        f"Unsupported relation code {code!r} for ({current_node}, {previous_node})"
                    )
                relation_ids[current_node - 1, previous_node - 1] = RELATION_TO_ID[normalized]

        all_must_previous: dict[int, tuple[int, ...]] = {}
        immediate_previous: dict[int, int | None] = {}
        for node_idx in range(1, NUM_GRAPH_NODES + 1):
            node = nodes[node_idx]
            history = node["feature_history_constraints"]["all_must_previous_nodes"]
            all_must_previous[node_idx] = tuple(int(value) for value in history if 1 <= int(value) <= 35)
            immediate = node["execution_constraints"].get("must_immediately_previous_node")
            immediate_previous[node_idx] = int(immediate) if immediate is not None else None

        atomic_sequences = tuple(
            tuple(int(value) for value in item["nodes"] if 1 <= int(value) <= 35)
            for item in graph_json.get("atomic_sequences", [])
        )

        return cls(
            task_graph_path=task_graph_path,
            relation_matrix_path=relation_matrix_path,
            graph_json=graph_json,
            relation_json=relation_json,
            relation_ids=relation_ids,
            node_to_tier3=node_to_tier3,
            node_to_stage=node_to_stage,
            all_must_previous=all_must_previous,
            immediate_previous=immediate_previous,
            atomic_sequences=atomic_sequences,
        )


def stable_sample_seed(base_seed: int, sample_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def randomized_graph_valid_history(
    history_rows: list[dict[str, Any]],
    graph: TaskGraphSpec,
    seed: int,
) -> list[dict[str, Any]]:
    """Return one deterministic randomized topological order of observed history.

    Only relations among observed history nodes are used.  The current target node is
    deliberately not an input, preventing current-label leakage.  Runs containing a
    repeated graph node fall back to actual order; the primary M3 protocol uses normal
    runs, where nodes are unique.
    """
    if len(history_rows) <= 1:
        return list(history_rows)

    node_indices = [int(row["node_idx"]) for row in history_rows]
    if len(set(node_indices)) != len(node_indices):
        return list(history_rows)

    row_by_node = {int(row["node_idx"]): row for row in history_rows}
    observed = set(row_by_node)
    assigned: set[int] = set()
    blocks: list[list[int]] = []

    for sequence in graph.atomic_sequences:
        block = [node for node in sequence if node in observed]
        if block:
            blocks.append(block)
            assigned.update(block)

    for node_idx in node_indices:
        if node_idx not in assigned:
            blocks.append([node_idx])
            assigned.add(node_idx)

    node_to_block: dict[int, int] = {}
    for block_idx, block in enumerate(blocks):
        for node_idx in block:
            node_to_block[node_idx] = block_idx

    successors: dict[int, set[int]] = {idx: set() for idx in range(len(blocks))}
    indegree = {idx: 0 for idx in range(len(blocks))}
    for current_node in observed:
        current_block = node_to_block[current_node]
        for previous_node in graph.all_must_previous[current_node]:
            if previous_node not in observed:
                continue
            previous_block = node_to_block[previous_node]
            if previous_block == current_block or current_block in successors[previous_block]:
                continue
            successors[previous_block].add(current_block)
            indegree[current_block] += 1

    rng = random.Random(seed)
    available = [idx for idx, degree in indegree.items() if degree == 0]
    ordered_blocks: list[int] = []
    while available:
        selected = rng.choice(available)
        available.remove(selected)
        ordered_blocks.append(selected)
        for successor in sorted(successors[selected]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.append(successor)

    if len(ordered_blocks) != len(blocks):
        raise RuntimeError("Observed task-graph history unexpectedly contains a cycle")

    ordered_nodes = [node for block_idx in ordered_blocks for node in blocks[block_idx]]
    return [row_by_node[node] for node in ordered_nodes]

