from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from sensor_boundary.config import format_path, load_config
from sensor_boundary.engine import evaluate_caches, load_checkpoint
from sensor_boundary.protocols import load_protocol_runs
from sensor_boundary.utils import read_json, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normal/fault/all test splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--no-calibrated-online", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    output = format_path(cfg["paths"]["output_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    model, _ = load_checkpoint(output / args.checkpoint_name, device)
    online_cfg = cfg["online"]
    calibrated = output / "calibrated_online.json"
    if calibrated.is_file() and not args.no_calibrated_online:
        online_cfg = read_json(calibrated)["selected_online"]
        print(f"Using validation-calibrated online settings from {calibrated}")
    protocol_root = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope
    limit = int(cfg["data"].get("limit_test_runs", 0))
    for split in cfg["data"]["test_splits"]:
        runs = load_protocol_runs(protocol_root / f"{split}.jsonl", limit)
        result = evaluate_caches(
            model, cfg["paths"]["cache_root"], runs, device, online_cfg, cfg["evaluation"],
            output / "evaluation" / split,
        )
        print(split, result["macro"], f"RTF={result['overall_real_time_factor']:.4f}")


if __name__ == "__main__":
    main()
