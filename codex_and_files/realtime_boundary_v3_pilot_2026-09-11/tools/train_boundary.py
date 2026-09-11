from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from boundary_experiment.config import format_path, load_config
from boundary_experiment.engine import build_model, make_loader, run_epoch, save_boundary_checkpoint
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import resolve_device, set_seed, write_json


def _validation_split(runs: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ranked = sorted(runs, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    count = max(1, round(len(ranked) * fraction))
    return ranked[count:], ranked[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one causal boundary detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(cfg["training"]["device"])
    cache_root = format_path(cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed, scope=args.scope, stride=cfg["features"]["stride_frames"])
    protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / "train.jsonl"
    train_runs, validation_runs = _validation_split(load_protocol_runs(protocol), float(cfg["training"]["validation_run_fraction"]), args.seed)
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite only intentionally: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", {**cfg, "condition": vars(args), "train_runs": train_runs, "validation_runs": validation_runs})
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loader = make_loader(cache_root, train_runs, cfg["training"], True)
    validation_loader = make_loader(cache_root, validation_runs, cfg["training"], False)
    best = float("inf")
    log_path = output / "training_log.jsonl"
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg["training"]["loss"], optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, cfg["training"]["loss"], None)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        save_boundary_checkpoint(output / "last.pth", model, optimizer, epoch, cfg, row)
        if validation_metrics["loss"] < best:
            best = validation_metrics["loss"]
            save_boundary_checkpoint(output / "best.pth", model, optimizer, epoch, cfg, row)
        print(f"epoch={epoch} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f}")


if __name__ == "__main__":
    main()
