from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from sensor_boundary.config import format_path, load_config
from sensor_boundary.engine import build_model, compute_normalization, make_loader, run_epoch, save_checkpoint
from sensor_boundary.protocols import load_protocol_runs
from sensor_boundary.utils import set_seed, resolve_device, write_json


def validation_split(runs: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ranked = sorted(runs, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    count = max(1, round(len(ranked) * fraction))
    return ranked[count:], ranked[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one causal IMU/EMG boundary model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(cfg["training"]["device"])
    protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / "train.jsonl"
    runs = load_protocol_runs(protocol, int(cfg["data"].get("limit_train_runs", 0)))
    train_runs, validation_runs = validation_split(runs, float(cfg["training"]["validation_run_fraction"]), args.seed)
    cache_root = Path(cfg["paths"]["cache_root"])
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite intentionally: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.jsonl"
    if args.overwrite and log_path.is_file():
        log_path.unlink()
    mean, std, channel_names = compute_normalization(cache_root, train_runs)
    if len(channel_names) != int(cfg["model"]["input_channels"]):
        raise ValueError(f"Config expects {cfg['model']['input_channels']} channels, cache has {len(channel_names)}")
    write_json(output / "resolved_config.json", {
        **cfg,
        "condition": vars(args),
        "train_runs": train_runs,
        "validation_runs": validation_runs,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "channel_names": channel_names,
    })
    model = build_model(cfg["model"], mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loader = make_loader(cache_root, train_runs, cfg["training"], True)
    validation_loader = make_loader(cache_root, validation_runs, cfg["training"], False)
    best = float("inf")
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg["training"]["loss"], optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, cfg["training"]["loss"])
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        save_checkpoint(output / "last.pth", model, optimizer, epoch, cfg, row, channel_names)
        if validation_metrics["loss"] < best:
            best = validation_metrics["loss"]
            save_checkpoint(output / "best.pth", model, optimizer, epoch, cfg, row, channel_names)
        print(f"epoch={epoch} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f}")


if __name__ == "__main__":
    main()
