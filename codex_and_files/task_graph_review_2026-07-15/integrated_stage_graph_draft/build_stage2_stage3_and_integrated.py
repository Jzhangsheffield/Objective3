from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "task_graph" / "task_graph.json"
STAGE2_DIR = HERE.parent / "stage2_feature_history_draft"
STAGE3_DIR = HERE.parent / "stage3_feature_history_draft"


DIRECT_MUST = {
    0: [],
    1: [0], 2: [1], 3: [0], 4: [2, 3], 5: [4],
    6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [10],
    12: [5, 6, 7, 8, 9, 11],
    13: [12], 14: [13], 15: [14], 16: [15], 17: [16],
    18: [17], 19: [18], 20: [19], 21: [20], 22: [21],
    23: [22], 24: [23], 25: [24],
    26: [25], 27: [26], 28: [25], 29: [25], 30: [25],
    31: [25], 32: [25], 33: [32], 34: [32], 35: [34],
    36: [27, 28, 29, 30, 31, 33, 35],
}

IMMEDIATE = {
    2: 1,
    11: 10,
    **{idx: idx - 1 for idx in range(13, 26)},
    27: 26,
    35: 34,
}

ATOMIC_SEQUENCES = [
    {"sequence_id": "unlock_and_place_lock", "nodes": [1, 2]},
    {"sequence_id": "remove_and_place_cover_stage1", "nodes": [10, 11]},
    {"sequence_id": "stage2_linear_sequence", "nodes": list(range(12, 26))},
    {"sequence_id": "replace_protection_cover", "nodes": [26, 27]},
    {"sequence_id": "take_lock_and_lock_crimper", "nodes": [34, 35]},
]


def compact_numeric_arrays(text: str) -> str:
    pattern = re.compile(r"\[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)?\]")

    def compact(match: re.Match[str]) -> str:
        values = re.findall(r"-?\d+", match.group(0))
        return "[" + ", ".join(values) + "]"

    return pattern.sub(compact, text)


def transitive_ancestors(indices: list[int]) -> dict[int, set[int]]:
    memo: dict[int, set[int]] = {}

    def visit(idx: int, active: set[int]) -> set[int]:
        if idx in memo:
            return memo[idx]
        if idx in active:
            raise ValueError(f"Mandatory dependency cycle at node {idx}")
        active = active | {idx}
        result: set[int] = set()
        for parent in DIRECT_MUST[idx]:
            result.add(parent)
            result.update(visit(parent, active))
        memo[idx] = result
        return result

    for idx in indices:
        visit(idx, set())
    return memo


def build_atomic_blocks(indices: list[int]) -> tuple[list[list[int]], dict[int, int], dict[int, int]]:
    assigned = set()
    blocks = []
    for sequence in ATOMIC_SEQUENCES:
        block = list(sequence["nodes"])
        blocks.append(block)
        assigned.update(block)
    for idx in indices:
        if idx not in assigned:
            blocks.append([idx])
    block_of = {idx: block_no for block_no, block in enumerate(blocks) for idx in block}
    position = {idx: pos for block in blocks for pos, idx in enumerate(block)}
    return blocks, block_of, position


def block_reachability(blocks: list[list[int]], block_of: dict[int, int]) -> dict[int, set[int]]:
    edges = {block_no: set() for block_no in range(len(blocks))}
    for target, parents in DIRECT_MUST.items():
        for parent in parents:
            a, b = block_of[parent], block_of[target]
            if a != b:
                edges[a].add(b)
    indegree = {block: 0 for block in edges}
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(block for block, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        block = queue.popleft()
        visited += 1
        for target in edges[block]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(edges):
        raise ValueError("Atomic-block dependency graph contains a cycle")

    memo: dict[int, set[int]] = {}

    def reachable(block: int) -> set[int]:
        if block in memo:
            return memo[block]
        result: set[int] = set()
        for target in edges[block]:
            result.add(target)
            result.update(reachable(target))
        memo[block] = result
        return result

    for block in edges:
        reachable(block)
    return memo


def build_nodes() -> list[dict]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    metadata = {int(node["node_idx"]): node for node in source["nodes"]}
    # Correct the obvious draft typo only in generated drafts.
    metadata[28] = dict(metadata[28])
    metadata[28]["node_id"] = "node_28_turn_off_extractor_fan"

    indices = list(range(37))
    ancestors = transitive_ancestors(indices)
    blocks, block_of, position = build_atomic_blocks(indices)
    block_reach = block_reachability(blocks, block_of)

    output = []
    for idx in indices:
        possible = []
        for candidate in indices:
            if candidate == idx:
                continue
            candidate_block = block_of[candidate]
            current_block = block_of[idx]
            if candidate_block == current_block:
                can_precede = position[candidate] < position[idx]
            else:
                can_precede = candidate_block not in block_reach[current_block]
            if can_precede:
                possible.append(candidate)

        all_must = sorted(ancestors[idx])
        optional = sorted(set(possible) - set(all_must))
        base = {
            key: value
            for key, value in metadata[idx].items()
            if key not in {"must_previous_nodes", "optional_previous_nodes"}
        }
        base["execution_constraints"] = {
            "direct_must_previous_nodes": DIRECT_MUST[idx],
            "must_immediately_previous_node": IMMEDIATE.get(idx),
        }
        base["feature_history_constraints"] = {
            "all_must_previous_nodes": all_must,
            "possible_previous_nodes": possible,
            "optional_previous_nodes": optional,
        }
        output.append(base)
    return output


def document(nodes: list[dict], scope: str) -> dict:
    if scope == "stage2":
        selected = [node for node in nodes if 12 <= node["node_idx"] <= 25]
        note = "Stage 2 nodes only; history lists use global node indices and include mandatory Stage 1 history."
        sequences = [sequence for sequence in ATOMIC_SEQUENCES if sequence["sequence_id"] == "stage2_linear_sequence"]
    elif scope == "stage3":
        selected = [node for node in nodes if 26 <= node["node_idx"] <= 36]
        note = "Stage 3 nodes plus end; history lists use global node indices and include mandatory Stage 1 and Stage 2 history."
        sequences = [sequence for sequence in ATOMIC_SEQUENCES if sequence["sequence_id"] in {"replace_protection_cover", "take_lock_and_lock_crimper"}]
    else:
        selected = nodes
        note = "Integrated Stage 1–3 draft; the original task_graph.json is unchanged."
        sequences = ATOMIC_SEQUENCES
    return {
        "schema_version": "feature-history-v1",
        "scope": scope,
        "source_note": note,
        "semantics": {
            "execution_constraints": "Used for scheduling and legality checks; must remain acyclic.",
            "feature_history_constraints": "Used as a legal historical-feature candidate mask; mutual possible relationships are allowed.",
            "possible_previous_nodes": "A node is included when at least one valid integrated execution can place it before the current node.",
            "optional_previous_nodes": "Derived as possible_previous_nodes minus all_must_previous_nodes.",
        },
        "atomic_sequences": sequences,
        "nodes": selected,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    path.write_text(compact_numeric_arrays(formatted) + "\n", encoding="utf-8")


def validate(nodes: list[dict]) -> dict:
    by_idx = {node["node_idx"]: node for node in nodes}
    assert sorted(by_idx) == list(range(37))
    for idx, node in by_idx.items():
        execution = node["execution_constraints"]
        history = node["feature_history_constraints"]
        immediate = execution["must_immediately_previous_node"]
        if immediate is not None:
            assert immediate in execution["direct_must_previous_nodes"]
        assert set(history["all_must_previous_nodes"]).issubset(history["possible_previous_nodes"])
        assert set(history["optional_previous_nodes"]) == set(history["possible_previous_nodes"]) - set(history["all_must_previous_nodes"])
    for idx in range(13, 26):
        assert by_idx[idx]["execution_constraints"]["must_immediately_previous_node"] == idx - 1
    assert by_idx[27]["execution_constraints"]["must_immediately_previous_node"] == 26
    assert by_idx[33]["execution_constraints"]["direct_must_previous_nodes"] == [32]
    assert by_idx[34]["execution_constraints"]["direct_must_previous_nodes"] == [32]
    assert by_idx[35]["execution_constraints"]["must_immediately_previous_node"] == 34
    assert by_idx[36]["feature_history_constraints"]["all_must_previous_nodes"] == list(range(36))
    return {
        "node_count": len(nodes),
        "mandatory_graph_acyclic": True,
        "stage2_immediate_edges": 13,
        "stage3_immediate_pairs": [[26, 27], [34, 35]],
        "end_has_all_nodes_as_mandatory_history": True,
    }


def main() -> None:
    nodes = build_nodes()
    validation = validate(nodes)
    write_json(STAGE2_DIR / "stage2_task_graph_latest.json", document(nodes, "stage2"))
    write_json(STAGE3_DIR / "stage3_task_graph_latest.json", document(nodes, "stage3"))
    write_json(HERE / "integrated_task_graph_latest.json", document(nodes, "integrated"))
    write_json(HERE / "validation_summary.json", validation)


if __name__ == "__main__":
    main()
