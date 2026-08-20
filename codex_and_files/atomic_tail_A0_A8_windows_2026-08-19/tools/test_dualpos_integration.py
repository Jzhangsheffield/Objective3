from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.config import load_config, run_spec
from atomic_tail_exp.data import MultiViewHistoryDataset, collate_multiview_batch
from atomic_tail_exp.graph import TaskGraph
from atomic_tail_exp.model import build_model
from atomic_tail_exp.training import (
    build_phase_optimizer,
    configure_trainable_parameters,
    forward_view,
    load_checkpoint_compatible,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-batch DualPos integration test using shared features/A0 weights.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "experiment_config.json"))
    parser.add_argument("--participant", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scope", default="all_runs")
    args = parser.parse_args()

    config = load_config(args.config)
    spec = run_spec(config, "A4-DualPos", args.participant, args.seed, args.scope)
    graph = TaskGraph.load(spec["paths"]["task_graph"], spec["paths"]["relation_matrix"])
    manifest = Path(spec["paths"]["protocol_root"]) / args.scope / "train.jsonl"
    dataset = MultiViewHistoryDataset(
        spec["paths"]["train_cache"], manifest, graph, spec,
        config["augmentation"], args.seed, training=True,
    )
    dataset.set_epoch(1)
    selected = []
    changed = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if item["augmentation_changed"] or len(selected) < 32:
            selected.append(item)
            changed += int(item["augmentation_changed"])
        if len(selected) >= 64 and changed >= 8:
            break
    if not selected or changed == 0:
        raise AssertionError("Integration batch contains no changed DualPos samples")
    batch = collate_multiview_batch(selected)
    assert int((batch["actual_history_shift_ids"] != 0).sum()) == 0
    assert int((batch["augmented_history_shift_ids"] != 0).sum()) > 0

    model = build_model(config["model"])
    load_report = load_checkpoint_compatible(model, spec["warm_start_checkpoint"])
    model.eval()
    with torch.no_grad():
        actual_logits, actual_extra = forward_view(model, batch, "actual")
        logits_without_shift, extra_without_shift = model(
            batch["current_feature"],
            batch["actual_history_features"],
            batch["actual_history_position_ids"],
            batch["actual_history_padding_mask"],
        )
    assert torch.equal(actual_logits, logits_without_shift)
    assert torch.equal(actual_extra["history_context"], extra_without_shift["history_context"])

    configure_trainable_parameters(model, "shift_only")
    model.train()
    actual_logits, _ = forward_view(model, batch, "actual")
    augmented_logits, _ = forward_view(model, batch, "augmented")
    targets = batch["node_target"]
    loss = 0.6 * F.cross_entropy(actual_logits, targets) + 0.4 * F.cross_entropy(augmented_logits, targets)
    loss.backward()
    shift_gradient = model.shift_embedding.weight.grad
    assert shift_gradient is not None and float(shift_gradient.abs().sum()) > 0.0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("shift_embedding.")
    )
    configure_trainable_parameters(model, "all")
    joint_optimizer = build_phase_optimizer(model, 1e-4, 1e-4, shift_learning_rate=5e-4)
    assert sorted(group["lr"] for group in joint_optimizer.param_groups) == [1e-4, 5e-4]
    configure_trainable_parameters(model, "base_only")
    assert not model.shift_embedding.weight.requires_grad
    print({
        "status": "passed",
        "samples": len(selected),
        "changed_samples": changed,
        "nonzero_shift_tokens": int((batch["augmented_history_shift_ids"] != 0).sum()),
        "shift_gradient_l1": float(shift_gradient.abs().sum()),
        "loaded_tensor_count": load_report["loaded_tensor_count"],
        "missing_keys": load_report["missing_keys"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
