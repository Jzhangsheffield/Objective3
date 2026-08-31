from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase_b.config import load_config
from phase_b.data import IMUDataset
from phase_b.evaluation import write_probability_evaluation
from phase_b.io import read_jsonl, seed_everything, write_json
from phase_b.metrics import derive_node_to_tier3
from phase_b.models import IMUResNet10
from phase_b.training import select_device


def collate_imu(batch: list[dict]) -> dict:
    return {
        "imu": torch.stack([row["imu"] for row in batch]),
        "node_target": torch.tensor([row["node_target"] for row in batch]),
        "rows": [row["row"] for row in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device, mapping, output, split) -> dict:
    model.eval()
    probabilities, rows = [], []
    for batch in loader:
        probabilities.append(F.softmax(model(batch["imu"].to(device)), dim=-1).cpu())
        rows.extend(batch["rows"])
    return write_probability_evaluation(rows, torch.cat(probabilities), mapping, output, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one direct-node right-hand IMU expert")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    settings = config["imu"]
    protocol = Path(args.protocol_dir)
    cache = Path(args.cache_dir)
    mapping = derive_node_to_tier3(read_jsonl(protocol / "train.jsonl"), int(config["num_nodes"]))
    train_dataset = IMUDataset(
        cache / "train_imu.pt", protocol / "train.jsonl", training=True,
        shift_probability=float(settings["time_shift_probability"]),
        shift_max_fraction=float(settings["time_shift_max_fraction"]),
    )
    device = select_device(args.device)
    loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_imu,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = IMUResNet10(
        in_channels=int(settings["channels"]), base_channels=int(settings["base_channels"]),
        num_nodes=int(config["num_nodes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(config["crossfit"]["imu_expert_epochs"])
    accumulation = max(1, int(settings["effective_batch_size"]) // int(settings["batch_size"]))
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        count = 0
        for step, batch in enumerate(loader, 1):
            target = batch["node_target"].to(device)
            loss = F.cross_entropy(model(batch["imu"].to(device)), target)
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * target.numel()
            count += target.numel()
        log.append({"epoch": epoch, "node_cross_entropy": loss_sum / max(1, count)})
        print(f"epoch={epoch}/{epochs} loss={log[-1]['node_cross_entropy']:.6f}", flush=True)
    checkpoint = output / "last.pth"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs,
        "seed": args.seed, "model_name": "IMUResNet10", "config": config,
        "node_to_tier3": mapping, "train_log": log,
    }, checkpoint)
    write_json(output / "train_log.json", log)
    results = output / "test_results"
    for split in config["evaluation"]["splits"]:
        dataset = IMUDataset(cache / "test_imu.pt", protocol / f"{split}.jsonl")
        test_loader = DataLoader(
            dataset, batch_size=int(settings["batch_size"]), shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_imu,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate(model, test_loader, device, mapping, results, split)
    write_json(output / "completed.json", {
        "checkpoint": str(checkpoint), "seed": args.seed,
        "training_target": "direct_node", "tested_splits": config["evaluation"]["splits"],
    })


if __name__ == "__main__":
    main()
