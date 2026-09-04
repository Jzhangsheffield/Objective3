from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskGraph:
    num_nodes: int
    node_to_tier3: tuple[int, ...]
    node_to_stage: tuple[int, ...]
    all_must_previous: dict[int, tuple[int, ...]]
    atomic_sequences: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, task_graph_path: str | Path, relation_matrix_path: str | Path | None = None) -> "TaskGraph":
        with Path(task_graph_path).open("r", encoding="utf-8") as handle:
            graph_json = json.load(handle)
        all_nodes = {int(item["node_idx"]): item for item in graph_json["nodes"]}
        nodes = {
            node_id: item
            for node_id, item in all_nodes.items()
            if int(item.get("action_id_tier3", -1)) >= 0
        }
        num_nodes = max(nodes)
        expected = set(range(1, num_nodes + 1))
        if not expected.issubset(nodes):
            raise ValueError(f"Task graph is missing action nodes: {sorted(expected - set(nodes))}")
        if relation_matrix_path is not None and not Path(relation_matrix_path).is_file():
            raise FileNotFoundError(relation_matrix_path)
        dependencies = {}
        for node_id, node in nodes.items():
            history = node["feature_history_constraints"]["all_must_previous_nodes"]
            dependencies[node_id] = tuple(int(value) for value in history if int(value) in expected)
        atomic = tuple(
            tuple(int(value) for value in item["nodes"] if int(value) in expected)
            for item in graph_json.get("atomic_sequences", [])
        )
        return cls(
            num_nodes=num_nodes,
            node_to_tier3=tuple(int(nodes[index]["action_id_tier3"]) for index in range(1, num_nodes + 1)),
            node_to_stage=tuple(int(nodes[index]["stage_id"]) for index in range(1, num_nodes + 1)),
            all_must_previous=dependencies,
            atomic_sequences=atomic,
        )


def randomized_graph_valid_history(rows: list[dict[str, Any]], graph: TaskGraph, rng: random.Random) -> list[dict[str, Any]]:
    """Randomized topological sort using history only; repeated nodes keep actual order."""
    if len(rows) <= 1:
        return list(rows)
    node_ids = [int(row["node_idx"]) for row in rows]
    if len(set(node_ids)) != len(node_ids):
        return list(rows)
    row_by_node = {int(row["node_idx"]): row for row in rows}
    observed = set(row_by_node)
    blocks: list[list[int]] = []
    assigned: set[int] = set()
    for sequence in graph.atomic_sequences:
        block = [node for node in sequence if node in observed]
        if block:
            blocks.append(block)
            assigned.update(block)
    for node in node_ids:
        if node not in assigned:
            blocks.append([node])
            assigned.add(node)
    node_to_block = {node: block_index for block_index, block in enumerate(blocks) for node in block}
    successors = {index: set() for index in range(len(blocks))}
    indegree = {index: 0 for index in range(len(blocks))}
    for current in observed:
        current_block = node_to_block[current]
        for previous in graph.all_must_previous[current]:
            if previous not in observed:
                continue
            previous_block = node_to_block[previous]
            if previous_block == current_block or current_block in successors[previous_block]:
                continue
            successors[previous_block].add(current_block)
            indegree[current_block] += 1
    available = [index for index, degree in indegree.items() if degree == 0]
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
        raise RuntimeError("Observed task graph contains a cycle")
    return [row_by_node[node] for block_index in ordered_blocks for node in blocks[block_index]]


def is_graph_valid(rows: list[dict[str, Any]], graph: TaskGraph) -> bool:
    positions = {int(row["node_idx"]): index for index, row in enumerate(rows)}
    if len(positions) != len(rows):
        return True
    for current, current_position in positions.items():
        for previous in graph.all_must_previous[current]:
            if previous in positions and positions[previous] > current_position:
                return False
    return True
