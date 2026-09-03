from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, add_phase_b_to_path, inner_feature_root, inner_m2_root,
    inner_protocol, load_config, outer_imu_token_cache, outer_m2_root,
    outer_protocol, safe_torch_load, seed_everything, select_device, sha256,
    write_json,
)
from phase_b1_imu_m2.data import IMUFeatureHistoryDataset, collate_history
from phase_b1_imu_m2.model import IMUM2Direct


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, mapping, output, split, writer) -> None:
    model.eval()
    probabilities, rows = [], []
    for batch in loader:
        logits, _ = model(move(batch, device))
        probabilities.append(F.softmax(logits, dim=-1).cpu())
        rows.extend(batch["rows"])
    writer(rows, torch.cat(probabilities), mapping, output, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an actual-history M2 head over frozen Phase B IMU features")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scope", required=True, choices=["inner", "outer"])
    parser.add_argument("--outer", required=True, choices=list("ADJM"))
    parser.add_argument("--inner", choices=list("ADJM"), default=None)
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.scope == "inner" and (args.inner is None or args.inner == args.outer):
        raise ValueError("inner scope requires --inner different from --outer")
    if args.scope == "outer" and args.inner is not None:
        raise ValueError("outer scope must not specify --inner")

    config = load_config(args.config)
    add_phase_b_to_path(config)
    from phase_b.evaluation import write_probability_evaluation
    from phase_b.io import read_jsonl
    from phase_b.metrics import derive_node_to_tier3

    if args.scope == "inner":
        feature_root = inner_feature_root(config, args.outer, str(args.inner), args.seed)
        train_cache, test_cache = feature_root / "train.pt", feature_root / "test.pt"
        protocol = inner_protocol(config, args.outer, str(args.inner))
        output = inner_m2_root(config, args.outer, str(args.inner), args.seed)
    else:
        train_cache = outer_imu_token_cache(config, args.outer, args.seed, "train")
        test_cache = outer_imu_token_cache(config, args.outer, args.seed, "test")
        protocol = outer_protocol(config, args.outer)
        output = outer_m2_root(config, args.outer, args.seed)
    required = [train_cache, test_cache, protocol / "train.jsonl", protocol / "test_all.jsonl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing M2 prerequisites:\n" + "\n".join(missing))
    completed = output / "completed.json"
    if completed.is_file() and not args.overwrite:
        print(f"SKIP completed: {completed}")
        return
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite partial output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    settings = config["imu_m2"]
    device = select_device(args.device)
    train_dataset = IMUFeatureHistoryDataset(train_cache, protocol / "train.jsonl")
    train_loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_history,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = IMUM2Direct(
        feature_dim=int(settings["feature_dim"]), d_model=int(settings["d_model"]),
        num_heads=int(settings["num_heads"]), max_history=int(settings["max_history"]),
        dropout=float(settings["dropout"]), num_nodes=int(config["num_nodes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    log = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        started = time.time()
        total_loss = correct = samples = 0
        for batch in train_loader:
            moved = move(batch, device)
            logits, _ = model(moved)
            loss = F.cross_entropy(logits, moved["node_target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            count = int(moved["node_target"].numel())
            total_loss += float(loss.detach()) * count
            correct += int((logits.argmax(dim=-1) == moved["node_target"]).sum())
            samples += count
        row = {
            "epoch": epoch, "train_node_cross_entropy": total_loss / samples,
            "train_node_accuracy": correct / samples, "seconds": time.time() - started,
        }
        log.append(row)
        print(
            f"epoch={epoch:03d}/{settings['epochs']} loss={row['train_node_cross_entropy']:.6f} "
            f"node_acc={row['train_node_accuracy']:.4f}", flush=True,
        )

    checkpoint = output / "last.pth"
    cache_payload = safe_torch_load(train_cache)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": int(settings["epochs"]), "condition": "IMU_M2",
        "scope": args.scope, "outer": args.outer, "inner": args.inner,
        "seed": args.seed, "settings": settings,
        "source_feature_cache": str(train_cache.resolve()),
        "source_feature_sha256": sha256(train_cache),
        "source_feature_metadata": cache_payload.get("metadata", {}),
    }, checkpoint)
    write_json(output / "train_log.json", log)

    mapping = derive_node_to_tier3(read_jsonl(protocol / "train.jsonl"), int(config["num_nodes"]))
    for split in config["evaluation"]["splits"]:
        dataset = IMUFeatureHistoryDataset(test_cache, protocol / f"{split}.jsonl")
        loader = DataLoader(
            dataset, batch_size=int(settings["batch_size"]), shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_history,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate(model, loader, device, mapping, output / "test_results", split, write_probability_evaluation)
    write_json(completed, {
        "condition": "IMU_M2", "scope": args.scope, "outer": args.outer,
        "inner": args.inner, "seed": args.seed, "checkpoint": str(checkpoint.resolve()),
        "splits": config["evaluation"]["splits"], "encoder_frozen_via_cache": True,
        "history": "actual_same_run_past_only",
    })
    print(f"Saved IMU M2 expert: {output}")


if __name__ == "__main__":
    main()
