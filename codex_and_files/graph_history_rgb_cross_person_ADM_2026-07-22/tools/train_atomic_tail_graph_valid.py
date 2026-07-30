from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from graph_history.atomic_tail_data import (
    AtomicTailGraphValidHistoryDataset,
    normalize_refresh_interval,
    refresh_policy_label,
)
from graph_history.data import collate_history_batch
from graph_history.dynamic_engine import train_epoch_shuffled_feature_model
from graph_history.dynamic_models import build_joint_head_delta_model
from graph_history.engine import evaluate_feature_model
from graph_history.graph import TaskGraphSpec
from graph_history.models import (
    FeatureNodeClassifier,
    build_context_model,
    build_direct_context_model,
)
from graph_history.utils import (
    ensure_dir,
    ensure_new_output_dir,
    load_compatible_state,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)


ATOMIC_TAIL_MODEL_NAMES = {
    "m3_atomic_tail_frozen_m0_delta": "atomic_tail_frozen_m0_delta",
    "m3_atomic_tail_joint_head_delta": "atomic_tail_joint_head_delta",
    "m3_atomic_tail_direct_fusion": "atomic_tail_direct_fusion",
}


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    persistent_workers: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_history_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train isolated atomic-tail graph-valid models.  The active incomplete "
            "atomic prefix is anchored at the end, while the remaining history is "
            "legally randomized at a configurable epoch interval."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(ATOMIC_TAIL_MODEL_NAMES),
    )
    parser.add_argument(
        "--train-scope",
        default="normal_only",
        choices=["normal_only", "all_runs"],
    )
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--m0-checkpoint", default=None)
    parser.add_argument(
        "--shuffle-refresh-interval",
        default="1",
        help="Positive epoch count (for example 1 or 10), or 'once'.",
    )
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

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    try:
        refresh_interval = normalize_refresh_interval(
            args.shuffle_refresh_interval
        )
    except ValueError as exc:
        parser.error(str(exc))
    refresh_policy = refresh_policy_label(refresh_interval)

    frozen_model_name = "m3_atomic_tail_frozen_m0_delta"
    if args.model == frozen_model_name and not args.m0_checkpoint:
        parser.error(f"--model {frozen_model_name} requires --m0-checkpoint")
    if args.model != frozen_model_name and args.m0_checkpoint:
        parser.error(
            "--m0-checkpoint is only valid for the frozen-M0 model; "
            "joint-head and direct models must not load M0"
        )

    seed_everything(args.seed)
    device = select_device(args.device)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    model_dir = ensure_new_output_dir(
        (
            Path(args.output_root)
            / args.train_scope
            / refresh_policy
            / args.model
        ),
        overwrite=args.overwrite,
    )
    train_manifest = Path(args.protocol_root) / args.train_scope / "train.jsonl"
    train_dataset = AtomicTailGraphValidHistoryDataset(
        feature_cache_path=args.train_cache,
        selection_manifest=train_manifest,
        graph=graph,
        shuffle_seed=args.seed,
        refresh_interval=args.shuffle_refresh_interval,
    )
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        device=device,
        persistent_workers=False,
    )

    baseline_report = None
    if args.model == frozen_model_name:
        baseline = FeatureNodeClassifier(feature_dim=train_dataset.feature_dim)
        baseline_report = load_compatible_state(baseline, args.m0_checkpoint)
        model = build_context_model(
            model_name="m3",
            baseline=baseline,
            relation_ids=graph.relation_ids,
            feature_dim=train_dataset.feature_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            max_history=args.max_history,
            dropout=args.dropout,
        )
        node_head_initialization = "m0_checkpoint_frozen"
        uses_m0_checkpoint = True
        uses_logit_delta = True
    elif args.model == "m3_atomic_tail_joint_head_delta":
        model = build_joint_head_delta_model(
            feature_dim=train_dataset.feature_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            max_history=args.max_history,
            dropout=args.dropout,
        )
        node_head_initialization = "random_trainable"
        uses_m0_checkpoint = False
        uses_logit_delta = True
    else:
        model = build_direct_context_model(
            model_name="m3_direct",
            feature_dim=train_dataset.feature_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            max_history=args.max_history,
            dropout=args.dropout,
        )
        node_head_initialization = "random_trainable"
        uses_m0_checkpoint = False
        uses_logit_delta = False

    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    shuffle_audit = train_dataset.audit_epochs(args.epochs)
    write_json(model_dir / "shuffle_audit.json", shuffle_audit)
    train_log = train_epoch_shuffled_feature_model(
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
    experiment_metadata = {
        "experiment_family": "atomic_tail_graph_valid",
        "model_long_name": ATOMIC_TAIL_MODEL_NAMES[args.model],
        "visual_backbone_frozen_via_feature_cache": True,
        "feature_cache_metadata": feature_metadata,
        "training_history_order": "atomic_tail_graph_valid",
        "evaluation_history_order": "atomic_tail_graph_valid_static_seeded",
        "shuffle_refresh_policy": refresh_policy,
        "shuffle_refresh_interval_epochs": refresh_interval,
        "shuffle_seed_formula": (
            "sha256(atomic_tail:base_seed:refresh_round:sample_name)"
        ),
        "uses_current_target_for_reordering": False,
        "uses_m0_checkpoint": uses_m0_checkpoint,
        "uses_logit_delta": uses_logit_delta,
        "node_head_initialization": node_head_initialization,
        "baseline_load_report": baseline_report,
        "shuffle_audit": shuffle_audit,
    }
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        vars(args),
        extra={**experiment_metadata, "train_log": train_log},
    )
    write_json(model_dir / "train_log.json", train_log)
    write_json(
        model_dir / "experiment_config.json",
        {**vars(args), **experiment_metadata},
    )
    parameter_summary = {
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "checkpoint": str(checkpoint_path),
        "uses_m0_checkpoint": uses_m0_checkpoint,
        "uses_logit_delta": uses_logit_delta,
        "node_head_initialization": node_head_initialization,
    }
    for module_name in ("baseline", "node_classifier", "delta_head", "fusion"):
        module = getattr(model, module_name, None)
        if module is not None:
            parameter_summary[f"{module_name}_parameters"] = sum(
                parameter.numel() for parameter in module.parameters()
            )
            parameter_summary[f"{module_name}_trainable_parameters"] = sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
    write_json(model_dir / "learned_parameters.json", parameter_summary)
    print(f"Saved final epoch checkpoint: {checkpoint_path}")

    test_result_root = ensure_dir(model_dir / "test_results")
    for split_name in ("test_normal", "test_fault", "test_all"):
        selection_manifest = (
            Path(args.protocol_root)
            / args.train_scope
            / f"{split_name}.jsonl"
        )
        test_dataset = AtomicTailGraphValidHistoryDataset(
            feature_cache_path=args.test_cache,
            selection_manifest=selection_manifest,
            graph=graph,
            shuffle_seed=args.seed,
            refresh_interval="once",
        )
        test_loader = build_loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            device=device,
            persistent_workers=True,
        )
        metrics = evaluate_feature_model(
            model,
            test_loader,
            device,
            graph.node_to_tier3,
            test_result_root,
            split_name,
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
            "experiment_family": "atomic_tail_graph_valid",
            "model": args.model,
            "checkpoint": str(checkpoint_path),
            "train_scope": args.train_scope,
            "refresh_policy": refresh_policy,
            "tested_splits": ["test_normal", "test_fault", "test_all"],
            "training_history_order": "atomic_tail_graph_valid",
            "evaluation_history_order": (
                "atomic_tail_graph_valid_static_seeded"
            ),
            "uses_current_target_for_reordering": False,
            "uses_m0_checkpoint": uses_m0_checkpoint,
            "uses_logit_delta": uses_logit_delta,
            "node_head_initialization": node_head_initialization,
        },
    )


if __name__ == "__main__":
    main()
