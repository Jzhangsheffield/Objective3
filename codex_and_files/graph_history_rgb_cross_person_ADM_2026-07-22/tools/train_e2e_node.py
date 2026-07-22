from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from graph_history.backbone import generate_model
from graph_history.constants import DEFAULT_CAMERA_ID, NUM_GRAPH_NODES
from graph_history.data import RGBClipDataset
from graph_history.graph import TaskGraphSpec
from graph_history.utils import (
    ensure_new_output_dir,
    load_compatible_state,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)
from graph_history.video_evaluation import evaluate_node_video_model


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_epoch(model, loader, optimizer, scaler, device: torch.device, amp: bool):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["node_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(video)
            loss = F.cross_entropy(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = int(target.shape[0])
        loss_sum += float(loss.detach()) * count
        correct += int((logits.argmax(dim=-1) == target).sum())
        total += count
    return loss_sum / max(1, total), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an end-to-end RGB ResNet3D-18 to predict 35 graph nodes"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init", required=True, choices=["scratch", "tier3"])
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Debug only")
    parser.add_argument("--max-test-samples", type=int, default=-1, help="Debug only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.init == "tier3" and not args.init_checkpoint:
        parser.error("--init tier3 requires --init-checkpoint")
    seed_everything(args.seed)
    device = select_device(args.device)
    protocol_root = Path(args.protocol_root).resolve()
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    output_dir = ensure_new_output_dir(args.output_dir, overwrite=args.overwrite)
    train_manifest = protocol_root / args.train_scope / "train.jsonl"
    train_dataset = RGBClipDataset(
        args.dataset_root, train_manifest, args.camera_id, train=True
    )
    if args.max_train_samples > 0:
        train_dataset.rows = train_dataset.rows[: args.max_train_samples]
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, device=device
    )

    model = generate_model(18, num_classes=NUM_GRAPH_NODES).to(device)
    init_report = None
    model_name = "e2e_node_scratch"
    if args.init == "tier3":
        model_name = "e2e_node_from_tier3"
        init_report = load_compatible_state(model, args.init_checkpoint)
        if set(init_report["missing_keys"]) != {"fc.weight", "fc.bias"}:
            raise RuntimeError(
                "Tier-3 initialization should load the entire backbone and skip only fc: "
                f"{init_report}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train_log: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss, accuracy = train_epoch(
            model, train_loader, optimizer, scaler, device, args.amp
        )
        row = {
            "epoch": epoch,
            "loss": loss,
            "node_accuracy": accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        train_log.append(row)
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} loss={loss:.6f} "
            f"node_acc={accuracy:.4f} seconds={row['seconds']:.1f}",
            flush=True,
        )
        scheduler.step()

    checkpoint_path = output_dir / "last.pth"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        args.epochs,
        vars(args),
        {"model": model_name, "init_report": init_report, "train_log": train_log},
    )
    write_json(output_dir / "train_log.json", train_log)
    write_json(
        output_dir / "experiment_config.json",
        {**vars(args), "model": model_name, "init_report": init_report},
    )
    final_load_report = load_compatible_state(model, checkpoint_path)
    if final_load_report["missing_keys"] or final_load_report["unexpected_keys"]:
        raise RuntimeError(f"Saved final node checkpoint failed reload: {final_load_report}")
    print(f"Saved and reloaded final epoch checkpoint: {checkpoint_path}")

    # Test manifests are loaded only after the final checkpoint has been written and reloaded.
    test_root = output_dir / "test_results"
    for split_name in ("test_normal", "test_fault", "test_all"):
        manifest = protocol_root / args.train_scope / f"{split_name}.jsonl"
        dataset = RGBClipDataset(args.dataset_root, manifest, args.camera_id, train=False)
        if args.max_test_samples > 0:
            dataset.rows = dataset.rows[: args.max_test_samples]
        loader = make_loader(
            dataset, args.batch_size, args.num_workers, shuffle=False, device=device
        )
        metrics = evaluate_node_video_model(
            model,
            loader,
            device,
            graph.node_to_tier3,
            test_root,
            split_name,
            model_name,
            checkpoint_path,
            amp=args.amp,
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
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "tested_splits": ["test_normal", "test_fault", "test_all"],
        },
    )


if __name__ == "__main__":
    main()
