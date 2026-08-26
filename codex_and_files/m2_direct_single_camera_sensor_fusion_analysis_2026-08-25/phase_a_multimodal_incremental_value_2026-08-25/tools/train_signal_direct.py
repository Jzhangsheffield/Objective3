from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.io import read_jsonl, seed_everything, write_json
from phase_a.metrics import derive_node_to_tier3
from phase_a.sensor_data import SignalClipDataset, collate_signal
from phase_a.supplementary import (
    base_protocol_dir,
    base_signal_cache,
    direct_num_classes,
    experiment_spec,
    load_supplementary_config,
    signal_channels,
    supplementary_model_dir,
    validate_supplementary_condition,
)
from phase_a.supplementary_engine import evaluate_direct, train_direct
from phase_a.supplementary_models import SignalDirectClassifier


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one independent right-hand signal Direct Node/Tier3 model")
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
    if spec["task"] not in {"direct_node", "direct_tier3"}:
        raise ValueError(f"{condition} is {spec['task']}; train_signal_direct accepts only direct tasks")
    seed_everything(args.seed)
    output = supplementary_model_dir(config, condition, args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocols = base_protocol_dir(config, args.participant)
    node_to_tier3 = derive_node_to_tier3(read_jsonl(protocols / "train.jsonl"))
    training = config["direct_training"]
    datasets = {}
    for split, manifest in (
        ("train", "train.jsonl"),
        ("test_all", "test_all.jsonl"),
        ("test_normal", "test_normal.jsonl"),
        ("test_fault", "test_fault.jsonl"),
    ):
        datasets[split] = SignalClipDataset(
            base_signal_cache(config, args.participant, "train" if split == "train" else "test"),
            protocols / manifest,
            spec["modality"],
            training=split == "train",
            time_shift_probability=float(training["time_shift_augmentation_probability"]),
            time_shift_max_fraction=float(training["time_shift_augmentation_max_fraction"]),
        )
    device = select_device(args.device)
    train_loader = DataLoader(
        datasets["train"], batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_signal,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = SignalDirectClassifier(
        config, spec["encoder"], signal_channels(spec["modality"]), direct_num_classes(spec["task"])
    ).to(device)
    accumulation = max(1, int(training["effective_batch_size"]) // int(training["batch_size"]))
    target_key = "node_target" if spec["task"] == "direct_node" else "tier3_target"
    log, optimizer = train_direct(
        model, train_loader, device, target_key, int(training["epochs"]),
        float(training["learning_rate"]), float(training["weight_decay"]), accumulation,
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
        "initialization": "scratch",
        "training_target": target_key,
    }, checkpoint_path)
    write_json(output / "train_log.json", log)
    for split in ("test_all", "test_normal", "test_fault"):
        loader = DataLoader(
            datasets[split], batch_size=int(training["batch_size"]), shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_signal,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate_direct(model, loader, device, spec["task"], node_to_tier3, output / "test_results", split)
    write_json(output / "completed.json", {
        "condition": condition,
        "participant": args.participant,
        "seed": args.seed,
        "task": spec["task"],
        "modality": spec["modality"],
        "encoder": spec["encoder"],
        "initialization": "scratch",
        "checkpoint": str(checkpoint_path),
        "splits": ["test_all", "test_normal", "test_fault"],
    })


if __name__ == "__main__":
    main()
