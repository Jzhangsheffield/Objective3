from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from graph_history.data import FeatureHistoryDataset, collate_history_batch
from graph_history.engine import evaluate_feature_model
from graph_history.graph import TaskGraphSpec
from graph_history.models import build_direct_context_model
from graph_history.utils import (
    ensure_new_output_dir,
    load_compatible_state,
    seed_everything,
    select_device,
    write_json,
)


EXPECTED_MODEL = "m3_atomic_tail_direct_fusion"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_loader(
    dataset: FeatureHistoryDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_history_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing Atomic-tail Direct Fusion checkpoint with the "
            "actual chronological test history.  Training outputs and the original "
            "atomic-tail test_results directory are never modified."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument(
        "--train-scope",
        required=True,
        choices=["normal_only", "all_runs"],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_normal", "test_fault", "test_all"],
        choices=["test_normal", "test_fault", "test_all"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / "last.pth"
    config_path = model_dir / "experiment_config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Atomic-tail checkpoint: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Atomic-tail config: {config_path}")

    config = read_json(config_path)
    if config.get("model") != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, found {config.get('model')!r} in {config_path}"
        )
    if config.get("train_scope") != args.train_scope:
        raise ValueError(
            "Requested train scope does not match checkpoint config: "
            f"{args.train_scope!r} != {config.get('train_scope')!r}"
        )

    seed = int(config.get("seed", 1))
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    num_workers = int(
        config.get("num_workers", 4)
        if args.num_workers is None
        else args.num_workers
    )
    seed_everything(seed)
    device = select_device(args.device)
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    protocol_root = Path(args.protocol_root).resolve()
    test_cache = Path(args.test_cache).resolve()
    output_dir = ensure_new_output_dir(
        args.output_dir or model_dir / "test_results_actual_order",
        overwrite=args.overwrite,
    )

    first_manifest = (
        protocol_root / args.train_scope / f"{args.splits[0]}.jsonl"
    )
    first_dataset = FeatureHistoryDataset(
        feature_cache_path=test_cache,
        selection_manifest=first_manifest,
        history_order="actual",
        graph=graph,
        shuffle_seed=seed,
    )
    model = build_direct_context_model(
        model_name="m3_direct",
        feature_dim=first_dataset.feature_dim,
        d_model=int(config.get("d_model", 256)),
        num_heads=int(config.get("num_heads", 4)),
        max_history=int(config.get("max_history", 35)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    load_report = load_compatible_state(model, checkpoint)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(
            "Atomic-tail Direct checkpoint is not fully compatible: "
            f"{load_report}"
        )

    write_json(
        output_dir / "evaluation_config.json",
        {
            **vars(args),
            "model": EXPECTED_MODEL,
            "mode": "evaluation_only_existing_checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_training_history_order": config.get(
                "training_history_order"
            ),
            "original_evaluation_history_order": config.get(
                "evaluation_history_order"
            ),
            "evaluation_history_order": "actual_chronological",
            "uses_current_target_for_reordering": False,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "checkpoint_load_report": load_report,
        },
    )

    for split_name in args.splits:
        if split_name == args.splits[0]:
            dataset = first_dataset
        else:
            manifest = (
                protocol_root / args.train_scope / f"{split_name}.jsonl"
            )
            dataset = FeatureHistoryDataset(
                feature_cache_path=test_cache,
                selection_manifest=manifest,
                history_order="actual",
                graph=graph,
                shuffle_seed=seed,
            )
        loader = build_loader(dataset, batch_size, num_workers, device)
        metrics = evaluate_feature_model(
            model,
            loader,
            device,
            graph.node_to_tier3,
            output_dir,
            split_name,
        )
        print(
            f"{split_name}: node_acc={metrics['node']['accuracy']:.4f} "
            f"tier3_acc={metrics['tier3']['accuracy']:.4f} "
            f"tier3_macro_f1={metrics['tier3']['macro_f1']:.4f}",
            flush=True,
        )

    write_json(
        output_dir / "completed.json",
        {
            "experiment_family": "atomic_tail_actual_order_evaluation",
            "model": EXPECTED_MODEL,
            "checkpoint": str(checkpoint),
            "train_scope": args.train_scope,
            "tested_splits": args.splits,
            "training_history_order": "atomic_tail_graph_valid",
            "evaluation_history_order": "actual_chronological",
            "uses_current_target_for_reordering": False,
        },
    )
    print(f"Saved actual-order evaluation: {output_dir}")


if __name__ == "__main__":
    main()
