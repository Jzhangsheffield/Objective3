from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_run_index
from boundary_experiment.config import format_path, load_config
from boundary_experiment.features import build_frozen_backbone, extract_run_features
from boundary_experiment.protocols import load_protocol_runs
from boundary_experiment.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract causal RGB window features for one LOSO condition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--splits", nargs="+", default=["train", "test_all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    checkpoint = format_path(cfg["paths"]["backbone_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    model, load_report = build_frozen_backbone(cfg["paths"]["atomic_project_root"], checkpoint, device)
    cache_root = format_path(
        cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed,
        scope=args.scope, stride=cfg["features"]["stride_frames"],
    )
    run_index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    names: set[str] = set()
    for split in args.splits:
        protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / f"{split}.jsonl"
        names.update(load_protocol_runs(protocol))
    records = {}
    for position, name in enumerate(sorted(names), 1):
        output = cache_root / f"{name}.pt"
        if output.is_file() and not args.overwrite:
            print(f"[{position}/{len(names)}] skip {name}")
            continue
        print(f"[{position}/{len(names)}] extract {name}")
        records[name] = extract_run_features(run_index[name], model, output, device, cfg["features"], checkpoint)
    write_json(cache_root / "extraction_manifest.json", {"load_report": load_report, "runs": records})


if __name__ == "__main__":
    main()
