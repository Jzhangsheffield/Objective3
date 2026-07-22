# Frozen graph assets for the A/D/M cross-person package

`integrated_task_graph_latest.json` and `integrated_feature_history_matrix.json`
are copied snapshots of the reviewed task graph. The experiment code treats them as
read-only inputs. Rows of the relation matrix are current candidates; columns are
historical nodes.

These are immutable copies of the same reviewed assets used by the J-as-test pilot
package. Keeping copies in both packages makes their runs independent while preserving
an identical graph definition.
