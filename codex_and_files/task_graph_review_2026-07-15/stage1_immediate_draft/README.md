# Stage 1 immediate-predecessor draft

This draft is separate from the original `task_graph/task_graph.json`; the original file has not been modified.

## Files

- `stage1_task_graph.json`: proposed Stage 1 data model.
- `stage1_task_graph_standalone.html`: standalone graph for opening in a browser.
- `stage1-task-graph.html`: editable visualization fragment.

## Constraint encoding

- Nodes 2 and 11 use `must_immediately_previous_node` to prevent another task from being inserted after nodes 1 and 10 respectively.
- Node 4 uses `must_previous_nodes: [2, 3]`.
- Node 5 uses `must_previous_nodes: [4]`.
- Freely reorderable tasks have no dependencies between one another; `optional_previous_nodes` is empty.
- Synthetic node 12 represents Stage 1 completion and joins nodes `[5, 6, 7, 8, 9, 11]`. It is not intended to replace the original Stage 2 node 12 when merging the draft.
