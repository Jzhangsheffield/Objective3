# Integrated Stage 1–3 task graph review

## Result

The generated mandatory dependency graph contains 37 nodes (`node_idx` 0–36), is acyclic, and gives node 36 all nodes 0–35 as mandatory transitive history. The original `task_graph/task_graph.json` was not modified.

## Stage boundaries used

- Stage 2 node 12 starts only after Stage 1 terminal nodes `[5, 6, 7, 8, 9, 11]` are complete.
- Stage 2 is encoded as one immediate sequence from node 12 through node 25.
- Stage 3 entry branches `[26, 28, 29, 30, 31, 32]` require node 25.
- Node 36 joins Stage 3 terminal branches `[27, 28, 29, 30, 31, 33, 35]`.

## Stage 3 interpretation

- `26 → 27` is immediate: no task may be inserted between taking and replacing the protection cover.
- `32 → 33` is a normal mandatory relationship: other tasks may occur after turning off the crimper and before turning off the main switch.
- `32 → 34` is also normal mandatory: other tasks may occur before taking the lock.
- `34 → 35` is immediate: no task may be inserted between taking the lock and locking the crimper.
- Nodes 28, 29, 30, and 31 otherwise remain freely reorderable.

## Changes relative to the original draft

1. The synthetic Stage 1 completion node previously numbered 12 is not included in the integrated graph because node 12 is the real Stage 2 `take plier` action. Stage 1 completion is represented by node 12's join prerequisites instead.
2. The original typo `node_28_tunr_off_extractor_fan` is corrected to `node_28_turn_off_extractor_fan` in generated drafts only.
3. The original disconnected start/end sentinels are connected through Stage 1 entry prerequisites and the Stage 3 terminal join.
4. The original optional dependency lists are replaced by derived feature-history sets and are not interpreted as scheduling edges.

## Questions worth confirming before replacing the production graph

### Main-switch safety condition

The current interpretation follows the stated requirement exactly: node 33 requires only node 32. It therefore permits the main switch to be turned off while the water pump, air compressor, or extractor fan is still on. If that is not physically valid, change node 33 to:

```json
"direct_must_previous_nodes": [28, 29, 30, 32]
```

### Cover replacement versus locking

The current graph allows `34 → 35` to finish before `26 → 27`, because no relationship between the cover and lock branches was specified. If the protection cover must be installed before locking, add node 27 as a prerequisite of node 34 or node 35. Adding it to node 34 preserves the immediate pair:

```json
"node_34_take_lock_from_table": {
  "direct_must_previous_nodes": [27, 32]
}
```

### Meaning of “Stage 2 linear”

Stage 2 is currently encoded as strictly immediate: every node 13–25 has the preceding node as `must_immediately_previous_node`. If “linear” only means fixed relative order but permits non-task observations or auxiliary events between actions, keep the mandatory chain but clear the immediate field.

## Feature-history matrix scope

The Stage 2 and Stage 3 visual matrices show local stage candidates plus the stage-entry condition for readability. Their JSON files contain global history lists, so Stage 2 nodes include Stage 1 mandatory history and Stage 3 nodes include mandatory history from both earlier stages.
