from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graph_history.constants import MODEL_NAMES
from graph_history.data import FeatureHistoryDataset, collate_history_batch
from graph_history.engine import evaluate_feature_model, train_feature_model
from graph_history.graph import TaskGraphSpec
from graph_history.models import FeatureNodeClassifier, build_context_model
from graph_history.utils import (
    ensure_dir,
    load_compatible_state,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)


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
    parser = argparse.ArgumentParser(description="Train M0-M6 on cached RGB features without validation")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--m0-checkpoint", default=None)
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
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    model_dir = ensure_dir(Path(args.output_root) / args.train_scope / args.model)
    train_manifest = Path(args.protocol_root) / args.train_scope / "train.jsonl"
    history_order = "graph_valid" if args.model == "m3" else "actual"
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

    if args.model == "m0":
        model = FeatureNodeClassifier(feature_dim=train_dataset.feature_dim)
        baseline_report = None
    else:
        m0_checkpoint = Path(args.m0_checkpoint) if args.m0_checkpoint else (
            Path(args.output_root) / args.train_scope / "m0" / "last.pth"
        )
        if not m0_checkpoint.is_file():
            raise FileNotFoundError(
                f"M0 checkpoint required before {args.model}: {m0_checkpoint}"
            )
        baseline = FeatureNodeClassifier(feature_dim=train_dataset.feature_dim)
        baseline_report = load_compatible_state(baseline, m0_checkpoint)
        model = build_context_model(
            model_name=args.model,
            baseline=baseline,
            relation_ids=graph.relation_ids,
            feature_dim=train_dataset.feature_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            max_history=args.max_history,
            dropout=args.dropout,
        )
    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
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
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        vars(args),
        extra={
            "model_long_name": MODEL_NAMES[args.model],
            "baseline_load_report": baseline_report,
            "train_log": train_log,
        },
    )
    write_json(model_dir / "train_log.json", train_log)
    parameter_summary = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "checkpoint": str(checkpoint_path),
    }
    if getattr(model, "graph_source", "none") != "none":
        parameter_summary["relation_bias"] = model.relation_bias.detach().cpu().tolist()
        parameter_summary["immediate_not_last_bias"] = (
            model.immediate_not_last_bias.detach().cpu().tolist()
        )
    write_json(model_dir / "learned_parameters.json", parameter_summary)
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


if __name__ == "__main__":
    main()
