# Latest Stage 1 task graph draft

This version separates task-execution constraints from model feature-history constraints. The original `task_graph/task_graph.json` has not been modified.

## Files

- `stage1_task_graph_latest.json`: latest Stage 1 graph and per-node feature-history candidate sets.
- `build_latest_stage1.py`: reproducible derivation of mandatory ancestors, possible predecessors, and optional predecessors.
- `stage1_feature_history_standalone.html`: standalone view of the execution graph and feature-history matrix.
- `stage1-feature-history-graph.html`: editable visualization fragment.

## Matrix symbols

- `M`: the column node is a mandatory direct or transitive predecessor.
- `I`: the column node must immediately precede the row node.
- `O`: the column node can occur earlier in at least one valid execution, but is not mandatory.
- `·`: the column node cannot occur before the row node.
- `—`: the row and column refer to the same node.

The synthetic `node_12_stage_1_complete` joins nodes `[5, 6, 7, 8, 9, 11]` and is only used to express Stage 1 completion in this standalone draft.
