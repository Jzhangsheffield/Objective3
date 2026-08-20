from __future__ import annotations

import random
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.augmentation import TransitionModel, augment_history, select_active_tail
from atomic_tail_exp.config import load_config, normalize_experiment_ids
from atomic_tail_exp.graph import TaskGraph, is_graph_valid


def row(sample: str, node: int, position: int) -> dict:
    return {"sample_name": sample, "node_idx": node, "participant": "P", "run": "r1", "annotation_row_index": position}


def main() -> int:
    config = load_config(PACKAGE_ROOT / "config" / "experiment_config.json")
    assert normalize_experiment_ids(None, config) == ["A0", "A3-DualPos", "A4-DualPos"]
    assert normalize_experiment_ids("A5", config) == ["A0", "A5"]
    assert normalize_experiment_ids("a3-FULL-shuffle,A4", config) == ["A0", "A3-full-shuffle", "A4"]
    full_shuffle = config["experiments"]["A3-full-shuffle"]
    assert full_shuffle["active_tail_only"] is False
    assert full_shuffle["position_mode"] == "actual_recency"
    assert full_shuffle["status"] == "deferred"
    a4 = config["experiments"]["A4"]
    assert a4["status"] == "deferred"
    a3_dual = config["experiments"]["A3-DualPos"]
    assert a3_dual["position_mode"] == "true_plus_shift"
    assert a3_dual["schedule"] == "scratch"
    a4_dual = config["experiments"]["A4-DualPos"]
    assert a4_dual["position_mode"] == "true_plus_shift"
    assert a4_dual["schedule"] == "dualpos_finetune_calibrate"
    assert a4_dual["shift_warmup_epochs"] == 2
    assert a4_dual["mixed_finetune_epochs"] == 8
    assert a4_dual["actual_calibration_epochs"] == 3
    assert a4_dual["actual_ce_weight"] == 0.6
    assert a4_dual["refresh_interval"] == 2
    graph = TaskGraph(
        num_nodes=6,
        node_to_tier3=(0, 1, 2, 3, 4, 5),
        node_to_stage=(1, 1, 1, 1, 1, 1),
        all_must_previous={1: (), 2: (), 3: (), 4: (), 5: (), 6: ()},
        atomic_sequences=((2, 3, 4),),
    )
    active = [row("a", 1, 1), row("b", 2, 2), row("c", 3, 3)]
    decision = select_active_tail(active, graph)
    assert decision.applied and decision.node_ids == (2, 3)
    inactive = [row("a", 1, 1), row("f", 6, 2)]
    result = augment_history(
        inactive, graph, 1, True, "uniform", None, 8, 1.0, 1.0, 0, 0
    )
    assert list(result.rows) == inactive and not result.changed
    training_rows = [row("a", 1, 1), row("b", 2, 2), row("c", 3, 3), row("d", 4, 4)]
    transition = TransitionModel.fit(training_rows, 6)
    result = augment_history(
        active, graph, 2, True, "plausibility_weighted", transition, 8, 0.75, 1.0, 0, 0
    )
    assert [int(item["node_idx"]) for item in result.rows[-2:]] == [2, 3]
    assert is_graph_valid(list(result.rows), graph)
    shuffled = list(active)
    random.Random(1).shuffle(shuffled)
    assert 0.0 <= result.normalized_kendall_distance <= 1.0
    print("Smoke tests passed: DualPos selection/config, deferred flags, gating, sampling, graph validity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
