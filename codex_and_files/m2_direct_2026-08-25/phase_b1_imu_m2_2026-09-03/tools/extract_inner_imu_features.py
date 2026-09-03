from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_b1_imu_m2.common import (
    DEFAULT_CONFIG, add_phase_b_to_path, inner_feature_root, inner_phase_b_root,
    inner_protocol, load_config, read_jsonl, safe_torch_load, select_device,
    sha256, write_json,
)


@torch.no_grad()
def extract(model, signal_cache: Path, manifest: Path, device: torch.device, batch_size: int) -> dict:
    cache = safe_torch_load(signal_cache)
    signals = torch.as_tensor(cache["imu"]).float()
    records = cache["records"]
    lookup = {str(row["sample_name"]): index for index, row in enumerate(records)}
    rows = read_jsonl(manifest)
    indices = []
    for row in rows:
        name = str(row["sample_name"])
        if name not in lookup:
            raise KeyError(f"IMU signal cache misses {name}")
        indices.append(lookup[name])
    features = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        selected = torch.tensor(indices[start:start + batch_size], dtype=torch.long)
        value = model.forward_features(signals[selected].to(device, non_blocking=True))
        features.append(value.float().cpu())
    result = torch.cat(features)
    if result.shape != (len(rows), 512):
        raise RuntimeError(f"Unexpected IMU feature shape: {tuple(result.shape)}")
    return {"features": result, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen inner-LOSO IMU features for the new M2 expert")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--outer", required=True, choices=list("ADJM"))
    parser.add_argument("--inner", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.inner == args.outer:
        raise ValueError("inner participant must differ from outer participant")
    config = load_config(args.config)
    add_phase_b_to_path(config)
    from phase_b.models import IMUResNet10

    source = inner_phase_b_root(config, args.outer, args.inner, args.seed)
    protocol = inner_protocol(config, args.outer, args.inner)
    checkpoint = source / "imu_direct_node" / "last.pth"
    signal_paths = {"train": source / "imu_cache" / "train_imu.pt", "test": source / "imu_cache" / "test_imu.pt"}
    manifest_paths = {"train": protocol / "train.jsonl", "test": protocol / "test_all.jsonl"}
    required = [checkpoint, *signal_paths.values(), *manifest_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Phase B prerequisites:\n" + "\n".join(missing))

    output = inner_feature_root(config, args.outer, args.inner, args.seed)
    completed = output / "completed.json"
    if completed.is_file() and not args.overwrite:
        print(f"SKIP completed: {completed}")
        return
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite partial output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    phase_b_config = config["_phase_b_config"]
    model = IMUResNet10(
        in_channels=int(phase_b_config["imu"]["channels"]),
        base_channels=int(phase_b_config["imu"]["base_channels"]),
        num_nodes=int(config["num_nodes"]),
    ).to(select_device(args.device))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    device = next(model.parameters()).device

    for split in ("train", "test"):
        payload = extract(model, signal_paths[split], manifest_paths[split], device, args.batch_size)
        payload["metadata"] = {
            "split": split,
            "outer": args.outer,
            "inner": args.inner,
            "seed": args.seed,
            "source_encoder": str(checkpoint.resolve()),
            "source_encoder_sha256": sha256(checkpoint),
            "source_signal_cache": str(signal_paths[split].resolve()),
            "source_manifest": str(manifest_paths[split].resolve()),
            "feature_dim": 512,
            "encoder_frozen": True,
        }
        torch.save(payload, output / f"{split}.pt")
    write_json(completed, {
        "outer": args.outer, "inner": args.inner, "seed": args.seed,
        "train_features": str((output / "train.pt").resolve()),
        "test_features": str((output / "test.pt").resolve()),
        "source_encoder": str(checkpoint.resolve()),
    })
    print(f"Saved inner IMU features: {output}")


if __name__ == "__main__":
    main()
