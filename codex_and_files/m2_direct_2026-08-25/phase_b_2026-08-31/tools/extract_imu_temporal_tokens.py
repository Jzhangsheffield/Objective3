from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_b.config import load_config
from phase_b.data import IMUDataset
from phase_b.io import write_json
from phase_b.models import IMUResNet10
from phase_b.training import select_device


def collate(batch: list[dict]) -> dict:
    return {"imu": torch.stack([row["imu"] for row in batch]), "rows": [row["row"] for row in batch]}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IMU temporal tokens plus global features")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = load_config(args.config)
    device = select_device(args.device)
    dataset = IMUDataset(args.cache, args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    model = IMUResNet10(
        int(config["imu"]["channels"]), int(config["imu"]["base_channels"]),
        int(config["num_nodes"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    tokens, globals_, rows = [], [], []
    for batch in loader:
        value = model.forward_tokens(batch["imu"].to(device))
        tokens.append(value.float().cpu())
        globals_.append(value.mean(dim=1).float().cpu())
        rows.extend(batch["rows"])
    payload = {
        "global_features": torch.cat(globals_), "temporal_tokens": torch.cat(tokens), "records": rows,
        "metadata": {
            "manifest": str(Path(args.manifest).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
            "temporal_channels": 512, "expected_temporal_tokens": 8, "global_dim": 512,
        },
    }
    if payload["temporal_tokens"].shape[1:] != (8, 512):
        raise RuntimeError(f"Unexpected IMU token shape: {tuple(payload['temporal_tokens'].shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    write_json(output.with_suffix(".metadata.json"), payload["metadata"])


if __name__ == "__main__":
    main()
