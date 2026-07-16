from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "stage1_immediate_draft" / "stage1_task_graph.json"
OUTPUT = HERE / "stage1_task_graph_latest.json"


def compact_numeric_arrays(text: str) -> str:
    """Keep simple numeric arrays on one line while preserving object indentation."""
    pattern = re.compile(r"\[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)?\]")

    def compact(match: re.Match[str]) -> str:
        values = re.findall(r"-?\d+", match.group(0))
        return "[" + ", ".join(values) + "]"

    return pattern.sub(compact, text)


def transitive_ancestors(nodes: dict[int, dict]) -> dict[int, set[int]]:
    memo: dict[int, set[int]] = {}

    def visit(idx: int) -> set[int]:
        if idx in memo:
            return memo[idx]
        result: set[int] = set()
        for parent in nodes[idx].get("must_previous_nodes", []):
            parent = int(parent)
            result.add(parent)
            result.update(visit(parent))
        memo[idx] = result
        return result

    for idx in nodes:
        visit(idx)
    return memo


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_nodes = source["nodes"]
    by_idx = {int(node["node_idx"]): node for node in source_nodes}
    ancestors = transitive_ancestors(by_idx)

    # Immediate pairs are contracted into atomic scheduling blocks when deciding
    # whether one node can legally occur before another.
    blocks = [[0], [1, 2], [3], [6], [7], [8], [9], [10, 11], [4], [5], [12]]
    block_of = {idx: block_no for block_no, block in enumerate(blocks) for idx in block}
    position_in_block = {idx: pos for block in blocks for pos, idx in enumerate(block)}
    block_edges: dict[int, set[int]] = {i: set() for i in range(len(blocks))}
    for node in source_nodes:
        target = int(node["node_idx"])
        for parent in node.get("must_previous_nodes", []):
            a, b = block_of[int(parent)], block_of[target]
            if a != b:
                block_edges[a].add(b)

    block_reach: dict[int, set[int]] = {}

    def reachable(block: int) -> set[int]:
        if block in block_reach:
            return block_reach[block]
        result: set[int] = set()
        for nxt in block_edges[block]:
            result.add(nxt)
            result.update(reachable(nxt))
        block_reach[block] = result
        return result

    for block in block_edges:
        reachable(block)

    output_nodes = []
    all_indices = sorted(by_idx)
    for idx in all_indices:
        node = by_idx[idx]
        possible: list[int] = []
        for candidate in all_indices:
            if candidate == idx:
                continue
            candidate_block = block_of[candidate]
            current_block = block_of[idx]
            if candidate_block == current_block:
                can_precede = position_in_block[candidate] < position_in_block[idx]
            else:
                # candidate can be placed before current unless the hard block DAG
                # requires the current block to occur first.
                can_precede = candidate_block not in block_reach[current_block]
            if can_precede:
                possible.append(candidate)

        all_must = sorted(ancestors[idx])
        optional = sorted(set(possible) - set(all_must))
        output_node = {
            key: value
            for key, value in node.items()
            if key not in {"must_previous_nodes", "must_immediately_previous_node", "optional_previous_nodes"}
        }
        output_node["execution_constraints"] = {
            "direct_must_previous_nodes": node.get("must_previous_nodes", []),
            "must_immediately_previous_node": node.get("must_immediately_previous_node"),
        }
        output_node["feature_history_constraints"] = {
            "all_must_previous_nodes": all_must,
            "possible_previous_nodes": possible,
            "optional_previous_nodes": optional,
        }
        output_nodes.append(output_node)

    result = {
        "schema_version": "stage1-feature-history-v1",
        "source_note": "Standalone Stage 1 draft; the original task_graph.json is unchanged.",
        "semantics": {
            "execution_constraints": "Used for scheduling and legality checks; must remain acyclic.",
            "feature_history_constraints": "Used only as the model's legal historical-feature candidate mask; mutual possible relationships are allowed.",
            "possible_previous_nodes": "A node is included when at least one valid Stage 1 execution can place it before the current node.",
            "optional_previous_nodes": "Derived as possible_previous_nodes minus all_must_previous_nodes.",
        },
        "atomic_sequences": [
            {"sequence_id": "unlock_and_place_lock", "nodes": [1, 2]},
            {"sequence_id": "remove_and_place_cover", "nodes": [10, 11]},
        ],
        "nodes": output_nodes,
    }
    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    OUTPUT.write_text(compact_numeric_arrays(formatted) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
