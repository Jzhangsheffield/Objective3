from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from boundary_experiment.config import format_path, load_config
from boundary_experiment.engine import evaluate_caches, load_boundary_checkpoint
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one boundary checkpoint on normal/fault/all")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--checkpoint-name", default="best.pth")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    model, _ = load_boundary_checkpoint(output / args.checkpoint_name, device)
    cache_root = format_path(cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed, scope=args.scope, stride=cfg["features"]["stride_frames"])
    for split in ("test_normal", "test_fault", "test_all"):
        protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / f"{split}.jsonl"
        runs = load_protocol_runs(protocol)
        result = evaluate_caches(model, cache_root, runs, device, cfg["online"], cfg["evaluation"], output / "evaluation" / split)
        print(split, result["macro"])


if __name__ == "__main__":
    main()
