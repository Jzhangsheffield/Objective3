from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.engine import evaluate, train_model
from phase_a.io import file_sha256, read_jsonl, seed_everything, write_json
from phase_a.metrics import derive_node_to_tier3
from phase_a.sensor_data import SignalFeatureHistoryDataset, collate_feature_history
from phase_a.supplementary import (
    base_protocol_dir,
    experiment_spec,
    load_supplementary_config,
    supplementary_feature_cache,
    supplementary_model_dir,
    validate_supplementary_condition,
)
from phase_a.supplementary_models import SensorM2Direct


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one sensor-only M2-Direct from frozen Tier3 signal features")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_supplementary_config(args.config)
    condition = validate_supplementary_condition(config, args.condition)
    spec = experiment_spec(config, condition)
    if spec["task"] != "m2_node":
        raise ValueError(f"{condition} is {spec['task']}; train_sensor_m2 accepts only S1-S4")
    seed_everything(args.seed)
    output = supplementary_model_dir(config, condition, args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    upstream = str(spec["upstream"])
    train_features = supplementary_feature_cache(config, upstream, args.participant, args.seed, "train")
    test_features = supplementary_feature_cache(config, upstream, args.participant, args.seed, "test")
    for path in (train_features, test_features):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen Tier3 feature cache is missing: {path}")

    protocols = base_protocol_dir(config, args.participant)
    node_to_tier3 = derive_node_to_tier3(read_jsonl(protocols / "train.jsonl"))
    datasets = {}
    for split, manifest in (
        ("train", "train.jsonl"),
        ("test_all", "test_all.jsonl"),
        ("test_normal", "test_normal.jsonl"),
        ("test_fault", "test_fault.jsonl"),
    ):
        datasets[split] = SignalFeatureHistoryDataset(
            train_features if split == "train" else test_features, protocols / manifest
        )
    base = config["base"]
    training = config["direct_training"]
    device = select_device(args.device)
    train_loader = DataLoader(
        datasets["train"], batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_feature_history,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = SensorM2Direct(
        feature_dim=int(config["signal_feature_dim"]), d_model=int(base["d_model"]),
        num_heads=int(base["num_heads"]), max_history=int(base["max_history"]),
        dropout=float(base["dropout"]), num_nodes=35,
    ).to(device)
    accumulation = max(1, int(training["effective_batch_size"]) // int(training["batch_size"]))
    log, optimizer = train_model(
        model, train_loader, device, int(training["epochs"]), float(training["learning_rate"]),
        float(training["weight_decay"]), 0.0, node_to_tier3, accumulation,
    )
    checkpoint_path = output / "last.pth"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(training["epochs"]),
        "condition": condition,
        "participant": args.participant,
        "seed": args.seed,
        "spec": spec,
        "supplementary_config": config,
        "node_to_tier3": node_to_tier3,
        "initialization": "scratch_m2_and_node_head",
        "signal_encoder_training": "upstream_direct_tier3_then_frozen_feature_cache",
        "upstream": upstream,
        "train_feature_cache": str(train_features),
        "test_feature_cache": str(test_features),
        "train_feature_sha256": file_sha256(train_features),
        "test_feature_sha256": file_sha256(test_features),
    }, checkpoint_path)
    write_json(output / "train_log.json", log)
    for split in ("test_all", "test_normal", "test_fault"):
        loader = DataLoader(
            datasets[split], batch_size=int(training["batch_size"]), shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_feature_history,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate(model, loader, device, node_to_tier3, output / "test_results", split)
    write_json(output / "completed.json", {
        "condition": condition,
        "participant": args.participant,
        "seed": args.seed,
        "task": "m2_node",
        "modality": spec["modality"],
        "encoder": spec["encoder"],
        "upstream": upstream,
        "initialization": "scratch_m2_and_node_head",
        "signal_encoder": "frozen_after_direct_tier3_training",
        "checkpoint": str(checkpoint_path),
        "splits": ["test_all", "test_normal", "test_fault"],
    })


if __name__ == "__main__":
    main()
