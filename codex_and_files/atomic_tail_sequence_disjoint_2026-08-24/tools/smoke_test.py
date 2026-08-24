from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sequence_disjoint_exp.common import load_package_config, read_json, resolve_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one dataset/model batch without training")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participant", default="A")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    package_config = load_package_config(args.config)
    paths = resolve_paths(package_config, args.participant, args.seed)
    legacy_root = Path(paths["legacy_atomic_package_root"])
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    from atomic_tail_exp.data import MultiViewHistoryDataset, collate_multiview_batch
    from atomic_tail_exp.graph import TaskGraph
    from atomic_tail_exp.model import build_model
    from atomic_tail_exp.training import forward_view

    legacy_config = read_json(paths["legacy_atomic_config"])
    experiment = copy.deepcopy(legacy_config["experiments"]["A3-DualPos"])
    experiment.update({
        "experiment_id": "A3-DualPos-Every10",
        "refresh_interval": 10,
    })
    graph = TaskGraph.load(paths["task_graph"], paths["relation_matrix"])
    scope_root = Path(paths["protocol_root"]) / str(package_config["grid"]["train_scope"])
    train_dataset = MultiViewHistoryDataset(
        paths["train_cache"], scope_root / "train.jsonl", graph, experiment,
        legacy_config["augmentation"], args.seed, training=True,
    )
    if len(train_dataset) < 2:
        raise RuntimeError("Filtered training dataset is unexpectedly small")
    candidates = [train_dataset[index] for index in range(min(64, len(train_dataset)))]
    nonempty = [item for item in candidates if item["actual"]["features"].shape[0] > 0]
    if len(nonempty) < 2:
        raise RuntimeError("Could not find two non-empty histories for smoke test")
    batch = collate_multiview_batch(nonempty[:2])
    model = build_model(legacy_config["model"])
    actual_logits, _ = forward_view(model, batch, "actual")
    augmented_logits, _ = forward_view(model, batch, "augmented")
    expected_shape = (2, int(legacy_config["model"]["num_nodes"]))
    if tuple(actual_logits.shape) != expected_shape or tuple(augmented_logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected logits: actual={tuple(actual_logits.shape)}, augmented={tuple(augmented_logits.shape)}"
        )
    test_dataset = MultiViewHistoryDataset(
        paths["test_cache"], scope_root / "test_all.jsonl", graph, experiment,
        legacy_config["augmentation"], args.seed, training=False,
    )
    print(
        f"Smoke test passed: participant={args.participant}, seed={args.seed}, "
        f"train_samples={len(train_dataset)}, test_samples={len(test_dataset)}, logits={expected_shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

