from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from graph_history.data import FeatureHistoryDataset, collate_history_batch
from graph_history.engine import evaluate_feature_model, train_feature_model
from graph_history.graph import TaskGraphSpec
from graph_history.models import build_direct_context_model
from graph_history.utils import (
    ensure_dir,
    ensure_new_output_dir,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)


DIRECT_MODEL_NAMES = {
    "m1_direct": "direct_history_no_position",
    "m2_direct": "direct_actual_history",
    "m3_direct": "direct_graph_valid_shuffle",
}


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_history_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train isolated direct-head M1-M3 variants on frozen Tier-3 feature caches "
            "without an M0 checkpoint or logit delta"
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(DIRECT_MODEL_NAMES))
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=35)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    model_dir = ensure_new_output_dir(
        Path(args.output_root) / args.train_scope / args.model,
        overwrite=args.overwrite,
    )
    train_manifest = Path(args.protocol_root) / args.train_scope / "train.jsonl"
    history_order = "graph_valid" if args.model == "m3_direct" else "actual"
    train_dataset = FeatureHistoryDataset(
        args.train_cache,
        train_manifest,
        history_order=history_order,
        graph=graph,
        shuffle_seed=args.seed,
    )
    train_loader = build_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, device=device
    )

    model = build_direct_context_model(
        model_name=args.model,
        feature_dim=train_dataset.feature_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_history=args.max_history,
        dropout=args.dropout,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_log = train_feature_model(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        node_to_tier3=graph.node_to_tier3,
        epochs=args.epochs,
        action_loss_weight=args.action_loss_weight,
        amp=args.amp,
    )

    checkpoint_path = model_dir / "last.pth"
    feature_metadata = dict(train_dataset.cache.get("metadata", {}))
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        vars(args),
        extra={
            "experiment_family": "direct_head_fusion",
            "model_long_name": DIRECT_MODEL_NAMES[args.model],
            "visual_backbone_frozen_via_feature_cache": True,
            "feature_cache_metadata": feature_metadata,
            "train_log": train_log,
        },
    )
    write_json(model_dir / "train_log.json", train_log)
    write_json(
        model_dir / "experiment_config.json",
        {
            **vars(args),
            "experiment_family": "direct_head_fusion",
            "model_long_name": DIRECT_MODEL_NAMES[args.model],
            "history_order": history_order,
            "visual_backbone_frozen_via_feature_cache": True,
            "feature_cache_metadata": feature_metadata,
            "uses_m0_checkpoint": False,
            "uses_logit_delta": False,
            "node_head_initialization": "random",
        },
    )
    write_json(
        model_dir / "learned_parameters.json",
        {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "fusion_parameters": sum(parameter.numel() for parameter in model.fusion.parameters()),
            "node_classifier_parameters": sum(
                parameter.numel() for parameter in model.node_classifier.parameters()
            ),
            "checkpoint": str(checkpoint_path),
        },
    )
    print(f"Saved final epoch checkpoint: {checkpoint_path}")

    # Test manifests/caches are consumed only after the final checkpoint is saved.
    test_result_root = ensure_dir(model_dir / "test_results")
    for split_name in ("test_normal", "test_fault", "test_all"):
        selection_manifest = Path(args.protocol_root) / args.train_scope / f"{split_name}.jsonl"
        test_dataset = FeatureHistoryDataset(
            args.test_cache,
            selection_manifest,
            history_order=history_order,
            graph=graph,
            shuffle_seed=args.seed,
        )
        test_loader = build_loader(
            test_dataset, args.batch_size, args.num_workers, shuffle=False, device=device
        )
        metrics = evaluate_feature_model(
            model, test_loader, device, graph.node_to_tier3, test_result_root, split_name
        )
        print(
            f"{split_name}: node_acc={metrics['node']['accuracy']:.4f} "
            f"tier3_acc={metrics['tier3']['accuracy']:.4f} "
            f"tier3_macro_f1={metrics['tier3']['macro_f1']:.4f}",
            flush=True,
        )
    write_json(
        model_dir / "completed.json",
        {
            "experiment_family": "direct_head_fusion",
            "model": args.model,
            "checkpoint": str(checkpoint_path),
            "train_scope": args.train_scope,
            "tested_splits": ["test_normal", "test_fault", "test_all"],
            "uses_m0_checkpoint": False,
            "uses_logit_delta": False,
        },
    )


if __name__ == "__main__":
    main()
