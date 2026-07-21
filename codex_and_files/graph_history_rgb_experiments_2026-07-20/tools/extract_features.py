from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graph_history.backbone import generate_model
from graph_history.constants import DEFAULT_CAMERA_ID, NUM_TIER3_CLASSES
from graph_history.data import RGBClipDataset
from graph_history.utils import ensure_dir, load_compatible_state, seed_everything, select_device, write_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic 512-D RGB features")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1, help="Debug only; -1 extracts all")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = RGBClipDataset(args.dataset_root, args.manifest, args.camera_id, train=False)
    if args.max_samples > 0:
        dataset.rows = dataset.rows[: args.max_samples]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = generate_model(18, num_classes=NUM_TIER3_CLASSES).to(device)
    report = load_compatible_state(model, args.checkpoint)
    if "fc.weight" in report["missing_keys"] or "fc.bias" in report["missing_keys"]:
        raise RuntimeError(f"Tier-3 checkpoint classifier was not loaded: {report}")
    model.eval()
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    records: list[dict] = []
    row_cursor = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            batch_features = model.forward_features(video)
            batch_logits = model.forward_head(batch_features)
        features.append(batch_features.float().cpu())
        logits.append(batch_logits.float().cpu())
        count = int(video.shape[0])
        records.extend(dataset.rows[row_cursor:row_cursor + count])
        row_cursor += count
        print(f"extracted={row_cursor}/{len(dataset)}", flush=True)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    metadata = {
        "dataset_root": str(args.dataset_root),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "camera_id": args.camera_id,
        "feature_dim": 512,
        "n_frames": 16,
        "rgb_size": 224,
        "load_report": report,
    }
    torch.save(
        {
            "features": torch.cat(features, dim=0),
            "tier3_logits": torch.cat(logits, dim=0),
            "records": records,
            "metadata": metadata,
        },
        output_path,
    )
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(f"Saved feature cache: {output_path}")


if __name__ == "__main__":
    main()
