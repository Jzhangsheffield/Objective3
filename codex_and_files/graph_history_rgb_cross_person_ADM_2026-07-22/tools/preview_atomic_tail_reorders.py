from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from graph_history.atomic_tail_data import (
    atomic_tail_graph_valid_history,
    stable_atomic_tail_seed,
)
from graph_history.graph import TaskGraphSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preview atomic-tail graph-valid orders for canonical histories "
            "[1, ..., current_node - 1]."
        )
    )
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-round", type=int, default=0)
    parser.add_argument("--first-current-node", type=int, default=1)
    parser.add_argument("--last-current-node", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.first_current_node <= args.last_current_node <= 35:
        parser.error("Require 1 <= first current node <= last current node <= 35")
    if args.refresh_round < 0:
        parser.error("--refresh-round must be >= 0")

    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    output = []
    for current_node in range(
        args.first_current_node,
        args.last_current_node + 1,
    ):
        history_rows = [
            {
                "node_idx": node_idx,
                "sample_name": f"canonical_node_{node_idx}",
            }
            for node_idx in range(1, current_node)
        ]
        reordered, decision = atomic_tail_graph_valid_history(
            history_rows,
            graph=graph,
            seed=stable_atomic_tail_seed(
                args.seed,
                args.refresh_round,
                f"canonical_current_node_{current_node}",
            ),
        )
        row = {
            "current_node": current_node,
            "canonical_history": list(range(1, current_node)),
            "reordered_history": [
                int(item["node_idx"]) for item in reordered
            ],
            "atomic_tail": list(decision.node_indices),
            "decision_reason": decision.reason,
        }
        output.append(row)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    for row in output:
        print(
            f"current={row['current_node']:02d} "
            f"history={row['reordered_history']} "
            f"atomic_tail={row['atomic_tail']} "
            f"reason={row['decision_reason']}"
        )


if __name__ == "__main__":
    main()
