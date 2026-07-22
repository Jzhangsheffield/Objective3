from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph_history.backbone import generate_model
from graph_history.constants import DEFAULT_CAMERA_ID, NUM_TIER3_CLASSES
from graph_history.data import RGBClipDataset
from graph_history.metrics import classification_metrics
from graph_history.utils import ensure_dir, save_checkpoint, seed_everything, select_device, write_json


def train_epoch(model, loader, optimizer, scaler, device, amp: bool):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["tier3_target"].to(device, non_blocking=True)
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


@torch.no_grad()
def evaluate(model, loader, device, output_dir: Path, split_name: str):
    model.eval()
    truth: list[int] = []
    pred: list[int] = []
    rows: list[dict] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["tier3_target"].to(device, non_blocking=True)
        probabilities = F.softmax(model(video), dim=-1)
        predicted = probabilities.argmax(dim=-1)
        truth.extend(target.cpu().tolist())
        pred.extend(predicted.cpu().tolist())
        for index in range(target.shape[0]):
            rows.append(
                {
                    "sample_name": batch["sample_name"][index],
                    "participant": batch["participant"][index],
                    "run": batch["run"][index],
                    "annotation_row_index": int(batch["annotation_row_index"][index]),
                    "true_tier3_id": int(target[index]),
                    "pred_tier3_id": int(predicted[index]),
                    "confidence": float(probabilities[index, predicted[index]]),
                }
            )
    metrics = classification_metrics(truth, pred, NUM_TIER3_CLASSES)
    metrics["split"] = split_name
    metrics["samples"] = len(rows)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    with (output_dir / f"{split_name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RGB ResNet3D-18 from scratch without validation")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    train_manifest = Path(args.protocol_root) / args.train_scope / "train.jsonl"
    train_dataset = RGBClipDataset(
        args.dataset_root, train_manifest, args.camera_id, train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = generate_model(18, num_classes=NUM_TIER3_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train_log: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_epoch(model, train_loader, optimizer, scaler, device, args.amp)
        train_log.append({"epoch": epoch, "loss": loss, "accuracy": accuracy, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch:03d}/{args.epochs:03d} loss={loss:.6f} tier3_acc={accuracy:.4f}", flush=True)
        scheduler.step()

    checkpoint_path = output_dir / "last.pth"
    save_checkpoint(checkpoint_path, model, optimizer, args.epochs, vars(args), {"train_log": train_log})
    write_json(output_dir / "train_log.json", train_log)
    print(f"Saved final epoch checkpoint: {checkpoint_path}")

    # Test data is loaded only after final-epoch training has completed and last.pth exists.
    eval_root = ensure_dir(output_dir / "test_results")
    for split_name in ("test_normal", "test_fault", "test_all"):
        manifest = Path(args.protocol_root) / args.train_scope / f"{split_name}.jsonl"
        dataset = RGBClipDataset(args.dataset_root, manifest, args.camera_id, train=False)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        metrics = evaluate(model, loader, device, eval_root, split_name)
        print(f"{split_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
