from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch

from graph_history.atomic_tail_data import (
    AtomicTailGraphValidHistoryDataset,
    atomic_tail_graph_valid_history,
    normalize_refresh_interval,
    refresh_policy_label,
    select_atomic_tail,
)
from graph_history.graph import TaskGraphSpec
from graph_history.utils import write_jsonl


def canonical_history(last_node: int) -> list[dict[str, object]]:
    return [
        {"node_idx": node_idx, "sample_name": f"node_{node_idx}"}
        for node_idx in range(1, last_node + 1)
    ]


def build_fixture(root: Path, current_node: int) -> tuple[Path, Path]:
    records = []
    rows = []
    for node_idx in range(1, current_node + 1):
        sample_name = f"sample_{node_idx}"
        records.append({"sample_name": sample_name})
        rows.append(
            {
                "sample_name": sample_name,
                "participant": "P",
                "run": "run_1",
                "annotation_row_index": node_idx - 1,
                "node_idx": node_idx,
                "tier3_id": (node_idx - 1) % 31,
                "stage_id": 1 if node_idx <= 11 else 2,
            }
        )
    cache_path = root / "features.pt"
    torch.save(
        {
            "features": torch.randn(current_node, 512),
            "tier3_logits": torch.randn(current_node, 31),
            "records": records,
            "metadata": {"fixture": True},
        },
        cache_path,
    )
    manifest_path = root / "train.jsonl"
    write_jsonl(manifest_path, rows)
    return cache_path, manifest_path


def assert_atomic_tail_rules(graph: TaskGraphSpec) -> None:
    node_11_history = canonical_history(10)
    reordered, decision = atomic_tail_graph_valid_history(
        node_11_history,
        graph,
        seed=1,
    )
    assert decision.node_indices == (10,)
    assert int(reordered[-1]["node_idx"]) == 10

    node_15_history = canonical_history(14)
    reordered, decision = atomic_tail_graph_valid_history(
        node_15_history,
        graph,
        seed=2,
    )
    assert decision.node_indices == (12, 13, 14)
    assert [int(row["node_idx"]) for row in reordered[-3:]] == [12, 13, 14]

    completed = select_atomic_tail(canonical_history(11), graph)
    assert not completed.applied
    assert completed.reason == "atomic_sequence_complete"

    non_prefix_rows = canonical_history(9) + [
        {"node_idx": 11, "sample_name": "node_11"}
    ]
    non_prefix = select_atomic_tail(non_prefix_rows, graph)
    assert not non_prefix.applied
    assert non_prefix.reason == "observed_atomic_nodes_not_prefix"

    repeated_rows = node_11_history + [
        {"node_idx": 10, "sample_name": "node_10_repeat"}
    ]
    reordered, repeated = atomic_tail_graph_valid_history(
        repeated_rows,
        graph,
        seed=3,
    )
    assert repeated.reason == "repeated_node_fallback"
    assert reordered == repeated_rows
    print("atomic_tail_rules=ok node10_tail=ok stage2_prefix_tail=ok")


def assert_refresh_schedules(graph: TaskGraphSpec) -> None:
    assert normalize_refresh_interval("once") is None
    assert normalize_refresh_interval("1") == 1
    assert normalize_refresh_interval(10) == 10
    assert refresh_policy_label(None) == "refresh_once"
    assert refresh_policy_label(10) == "refresh_every_10"

    with tempfile.TemporaryDirectory(prefix="atomic_tail_refresh_") as temp:
        cache_path, manifest_path = build_fixture(Path(temp), current_node=15)
        dataset = AtomicTailGraphValidHistoryDataset(
            cache_path,
            manifest_path,
            graph=graph,
            shuffle_seed=42,
            refresh_interval=10,
        )
        last_index = len(dataset) - 1
        order_epoch_1 = tuple(
            row["sample_name"]
            for row in dataset.ordered_history_rows(last_index, epoch=1)
        )
        order_epoch_10 = tuple(
            row["sample_name"]
            for row in dataset.ordered_history_rows(last_index, epoch=10)
        )
        order_epoch_11 = tuple(
            row["sample_name"]
            for row in dataset.ordered_history_rows(last_index, epoch=11)
        )
        assert order_epoch_1 == order_epoch_10
        assert dataset.refresh_round(1) == 0
        assert dataset.refresh_round(10) == 0
        assert dataset.refresh_round(11) == 1
        assert order_epoch_1[-3:] == (
            "sample_12",
            "sample_13",
            "sample_14",
        )
        assert order_epoch_11[-3:] == (
            "sample_12",
            "sample_13",
            "sample_14",
        )

        once_dataset = AtomicTailGraphValidHistoryDataset(
            cache_path,
            manifest_path,
            graph=graph,
            shuffle_seed=42,
            refresh_interval="once",
        )
        assert tuple(
            row["sample_name"]
            for row in once_dataset.ordered_history_rows(last_index, epoch=1)
        ) == tuple(
            row["sample_name"]
            for row in once_dataset.ordered_history_rows(last_index, epoch=50)
        )
        audit = dataset.audit_epochs(25)
        assert audit["epoch_to_refresh_round"] == (
            [0] * 10 + [1] * 10 + [2] * 5
        )
        assert audit["atomic_tail_violations"] == 0
        print(
            "refresh_schedule=ok",
            f"epoch11_changed={order_epoch_1 != order_epoch_11}",
            "tail_violations=0",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for atomic-tail graph-valid reordering"
    )
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    args = parser.parse_args()
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    assert_atomic_tail_rules(graph)
    assert_refresh_schedules(graph)
    print("Atomic-tail graph-valid smoke test passed.")


if __name__ == "__main__":
    main()
