from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from sensor_boundary.annotations import load_run_index
from sensor_boundary.config import load_config
from sensor_boundary.preprocessing import extract_run
from sensor_boundary.protocols import load_protocol_runs
from sensor_boundary.utils import read_json, write_json


def _tensorize(cache: dict) -> dict:
    return {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in cache.items()
    }


def _limit(cfg: dict, split: str) -> int:
    return int(cfg["data"].get("limit_train_runs", 0) if split == "train" else cfg["data"].get("limit_test_runs", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create past-only IMU or EMG window caches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--splits", nargs="+", default=["train", "test_all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"])
    protocol_root = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope
    names: set[str] = set()
    for split in args.splits:
        names.update(load_protocol_runs(protocol_root / f"{split}.jsonl", _limit(cfg, split)))
    cache_root = Path(cfg["paths"]["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "extraction_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {"modality": cfg["sensor"]["modality"], "runs": {}}
    for position, name in enumerate(sorted(names), 1):
        output, sidecar = cache_root / f"{name}.pt", cache_root / f"{name}.json"
        if output.is_file() and sidecar.is_file() and not args.overwrite:
            print(f"[{position}/{len(names)}] skip {name}")
            continue
        print(f"[{position}/{len(names)}] extract {name}")
        cache = extract_run(index[name], cfg["sensor"])
        torch.save(_tensorize(cache), output)
        write_json(sidecar, cache["metadata"])
        manifest["runs"][name] = cache["metadata"]
        write_json(manifest_path, manifest)
    print(f"Cache ready: {cache_root}")


if __name__ == "__main__":
    main()
