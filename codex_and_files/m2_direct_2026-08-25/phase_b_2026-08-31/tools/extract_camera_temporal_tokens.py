from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_b.config import load_config
from phase_b.io import seed_everything, write_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB layer2 temporal tokens plus global features")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = load_config(args.config)
    project = Path(config["m2_project_root"])
    sys.path.insert(0, str(project))
    from graph_history.backbone import generate_model
    from graph_history.constants import NUM_TIER3_CLASSES
    from graph_history.data import RGBClipDataset
    from graph_history.utils import load_compatible_state, select_device

    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = RGBClipDataset(config["dataset_root"], args.manifest, args.camera_id, train=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = generate_model(18, num_classes=NUM_TIER3_CLASSES).to(device)
    report = load_compatible_state(model, args.checkpoint)
    model.eval()
    globals_, temporal, records = [], [], []
    cursor = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            value = model.maxpool(model.relu(model.bn1(model.conv1(video))))
            value = model.layer1(value)
            layer2 = model.layer2(value)
            temporal_value = layer2.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
            value = model.layer4(model.layer3(layer2))
            global_value = model.avgpool(value).flatten(1)
        temporal.append(temporal_value.float().cpu())
        globals_.append(global_value.float().cpu())
        count = video.shape[0]
        records.extend(dataset.rows[cursor:cursor + count])
        cursor += count
        print(f"extracted={cursor}/{len(dataset)}", flush=True)
    payload = {
        "global_features": torch.cat(globals_), "temporal_tokens": torch.cat(temporal),
        "records": records,
        "metadata": {
            "manifest": str(Path(args.manifest).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
            "camera_id": args.camera_id, "temporal_layer": "layer2", "temporal_channels": 128,
            "expected_temporal_tokens": 4, "global_dim": 512, "load_report": report,
        },
    }
    if payload["temporal_tokens"].shape[1:] != (4, 128):
        raise RuntimeError(f"Unexpected camera token shape: {tuple(payload['temporal_tokens'].shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    write_json(output.with_suffix(".metadata.json"), payload["metadata"])


if __name__ == "__main__":
    main()
