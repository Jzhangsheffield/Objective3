# Manifest graph-node enrichment

## Output

- `3_camera_mindrove_manifest_with_graph_nodes.jsonl`: enriched copy of the source manifest.
- `enrichment_report.json`: matching, sequence-alignment, validation, and node-frequency report.
- `enrich_manifest_with_graph_nodes.py`: reproducible enrichment program.

The source manifest at `C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\3_camera_mindrove_manifest.jsonl` was not modified.

## Added fields

Each JSONL record contains three new fields immediately after `tier3`:

```json
"node_id": "node_3_turn_on_main_switch",
"node_idx": 3,
"stage_id": 1
```

## Duplicate-label handling

Unique tier3 labels are matched directly. Repeated Stage 2 labels are aligned using `participant`, `run`, `annotation_row_index`, the Stage 2 sequence, and neighboring node transitions.

The data contains six observed `node 22 -> node 16` repeat transitions, representing extra crimp/reverse actions in intentionally erroneous runs. The enrichment reuses nodes 16–22 to identify those observed actions, but this transition must **not** be added to the task graph: the integrated graph intentionally represents only the standard normal run. Skipped and repeated actions remain deviations that can be detected against that standard.

## Validation

- Source and output: 1,895 records each.
- Unmatched labels: 0.
- Unresolved duplicate labels: 0.
- Every output record has `node_id`, `node_idx`, and `stage_id` matching the integrated graph.
- Removing the three new fields reproduces every original JSON object exactly.

## Subset manifests

The eight requested dataset directories contained 20 manifest files with 15,614 total rows. Every `sample_name` was found in the enriched master manifest, and every original record matched its master record before modification.

- `subset_manifests_backup_2026-07-15/`: byte-for-byte backups of all 20 original subset manifests, preserving their dataset-folder names.
- `subset_manifests_enriched_staged/`: validated enriched copies used for the update.
- `subset_manifest_enrichment_report.json`: file paths, row counts, SHA-256 hashes, matching results, and post-update verification.
- `enrich_subset_manifests.py`: reproducible backup, staging, update, and verification program.

All 20 target manifests were updated and their final hashes match the validated staged copies.
