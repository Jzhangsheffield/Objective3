# Task graph draft validation report

Source: `D:\Junxi_data\Objective3_thermal_crimp\task_graph\task_graph.json`

## Structural checks

- Nodes: **37**, with continuous `node_idx` 0–36: **yes**.
- Duplicate `node_idx`: **none**.
- Duplicate `node_id`: **none**.
- References to missing nodes: **none**.
- Self references: **none**.
- Cycle using only mandatory edges: **none**.
- Stage counts: start/end=2, stage 1=11, stage 2=14, stage 3=10.

## Findings and suggested corrections

### 1. Definite data error: duplicated predecessor in node 36

`node_36_end.optional_previous_nodes` contains `23` twice:

```json
"optional_previous_nodes": [23, 22, 23, 24, 25, 26, 28, 30]
```

At minimum, remove the duplicate. More importantly, review the intended end condition as described below.

### 2. Likely typo in node ID

`node_28_tunr_off_extractor_fan` probably should be `node_28_turn_off_extractor_fan`. The action labels are already spelled correctly. If other files refer to the old ID, update them together.

### 3. Start node is disconnected

Node 0 has no outgoing dependency path because every stage-1 entry action has an empty mandatory predecessor list. If node 0 is intended as the formal graph root, add `0` as a mandatory predecessor to every action that is genuinely allowed to start immediately (for example nodes 1, 3, 6, 7, 8, 9, and 10). If node 0 is only a display sentinel, document that convention instead.

### 4. Stage 1 does not currently guarantee completion before stage 2

Node 12 requires only node 5. Therefore the mandatory graph allows crimping to begin without completing nodes 6, 7, 8, 9, 10, and 11. If all setup actions must finish first, use a stage-1 join condition. The simplest representation is:

```json
"node_12_take_plier_from_table": {
  "must_previous_nodes": [5, 6, 7, 8, 9, 11]
}
```

This assumes node 11 represents completion of the 10→11 cover-removal branch. Confirm the real-world safety logic before applying it.

### 5. Optional predecessors create many reverse and two-way relations

There are **27** optional references where a higher `node_idx` is listed as a predecessor of a lower one, and **15** immediate two-way optional pairs. Examples include 1↔3, 6↔7, and 7↔8. This is not necessarily wrong if `optional_previous_nodes` means “may occur in either order,” but it is cyclic if downstream code treats every listed predecessor as a normal directed dependency.

Recommended schema choice:

- If optional edges are real prerequisites, require the combined graph to be acyclic and keep only the intended direction.
- If these actions are freely orderable, represent them as a parallel/commutative group instead of two directed optional edges.
- If the field means an observation/history feature rather than a scheduling constraint, rename or document it so graph validators do not interpret it as dependency structure.

### 6. End node does not represent completion of shutdown

Node 36 has no mandatory predecessors and its optional list omits several terminal shutdown branches, including nodes 27, 29, 31, 33, and 35. If shutdown actions are all required, node 36 should be a mandatory join over the terminal actions of each branch. Based on the current graph, a plausible candidate is:

```json
"must_previous_nodes": [27, 28, 29, 30, 31, 33, 35],
"optional_previous_nodes": []
```

This is a semantic suggestion, not an automatic fix: verify whether turning off the main switch (33) must wait for other equipment shutdowns and whether cover replacement (27) must precede locking (35).

### 7. Shutdown ordering may need safety constraints

Nodes 26–32 all become available immediately after node 25. Current mandatory edges allow cover replacement, pedal movement, utility shutdown, and crimper shutdown in any order. If the operating procedure requires a safe order, add those constraints explicitly. In particular, review whether node 33 should depend on all powered-device shutdown nodes (28, 29, 30, 32), and whether locking (35) should depend on both cover replacement (27) and main-switch shutdown (33).

## Edge interpretation used in the graph

- Solid arrow: `must_previous_nodes` predecessor → current node.
- Dashed purple arrow: `optional_previous_nodes` predecessor → current node.
- Node fill color: `stage_id`; stage −1 is used for start/end.

The visualization intentionally preserves the draft data rather than silently correcting it.
