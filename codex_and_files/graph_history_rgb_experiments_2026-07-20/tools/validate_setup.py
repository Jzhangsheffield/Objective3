from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from graph_history.constants import DEFAULT_CAMERA_ID
from graph_history.graph import TaskGraphSpec
from graph_history.protocols import find_fault_manifest
from graph_history.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paths and the J-as-test experiment inputs")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    train_rows = read_jsonl(dataset_root / "J_as_test" / "train_manifest.jsonl")
    test_rows = read_jsonl(dataset_root / "J_as_test" / "test_manifest.jsonl")
    fault_path = find_fault_manifest(dataset_root, "M")
    field = f"{args.camera_id}_rgb"
    missing = [row["sample_name"] for row in train_rows + test_rows if field not in row]
    if missing:
        raise ValueError(f"Camera field {field} missing for {missing[:10]}")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"train_samples={len(train_rows)} test_samples={len(test_rows)}")
    print(f"relation_matrix_shape={tuple(graph.relation_ids.shape)}")
    print(f"M fault manifest={fault_path.name}")
    print("Setup validation passed.")


if __name__ == "__main__":
    main()

