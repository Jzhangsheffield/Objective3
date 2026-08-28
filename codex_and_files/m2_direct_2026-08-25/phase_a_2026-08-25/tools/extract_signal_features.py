from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.io import file_sha256, write_json
from phase_a.sensor_data import SignalClipDataset, collate_signal
from phase_a.supplementary import (
    base_protocol_dir,
    direct_num_classes,
    evaluation_protocols,
    experiment_spec,
    load_supplementary_config,
    signal_channels,
    supplementary_feature_cache,
    supplementary_feature_dir,
    supplementary_model_dir,
    training_signal_cache,
    validate_supplementary_condition,
)
from phase_a.supplementary_models import SignalDirectClassifier


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def extract(model, loader, device: torch.device) -> tuple[torch.Tensor, list[dict]]:
    model.eval()
    features = []
    records = []
    for raw_batch in loader:
        signal = raw_batch["signal"].to(device, non_blocking=True)
        features.append(model.forward_features(signal).cpu())
        for index, sample_name in enumerate(raw_batch["sample_name"]):
            records.append({
                "sample_name": sample_name,
                "participant": raw_batch["participant"][index],
                "run": raw_batch["run"][index],
                "annotation_row_index": raw_batch["annotation_row_index"][index],
                "node_idx": int(raw_batch["node_target"][index]) + 1,
                "tier3_id": int(raw_batch["tier3_target"][index]),
                "stage_id": int(raw_batch["stage_id"][index]),
            })
    return torch.cat(features) if features else torch.empty((0, model.feature_dim)), records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen 512-D signal features from S9-S12 Tier3 checkpoints")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_supplementary_config(args.config)
    condition = validate_supplementary_condition(config, args.condition)
    spec = experiment_spec(config, condition)
    if spec["task"] != "direct_tier3":
        raise ValueError(f"Feature extraction requires S9-S12 direct_tier3, received {condition}")
    source = supplementary_model_dir(config, condition, args.participant, args.seed) / "last.pth"
    if not source.is_file():
        raise FileNotFoundError(f"Tier3 checkpoint is missing: {source}")
    output = supplementary_feature_dir(config, condition, args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty feature output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = SignalDirectClassifier(
        config, spec["encoder"], signal_channels(spec["modality"], config), direct_num_classes(spec["task"])
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    protocols = base_protocol_dir(config, args.participant)
    batch_size = int(config["direct_training"]["batch_size"])
    written = {}
    extraction_sets = [{
        "split": "train", "cache": training_signal_cache(config, args.participant),
        "manifest": protocols / "train.jsonl",
    }]
    extraction_sets.extend({
        "split": protocol["feature_split"], "cache": protocol["cache"],
        "manifest": protocol["manifest_dir"] / "test_all.jsonl",
    } for protocol in evaluation_protocols(config, args.participant))
    for item in extraction_sets:
        split = item["split"]
        dataset = SignalClipDataset(
            item["cache"], item["manifest"],
            spec["modality"], training=False,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_signal, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        features, records = extract(model, loader, device)
        target = supplementary_feature_cache(config, condition, args.participant, args.seed, split)
        torch.save({
            "features": features,
            "records": records,
            "modality": spec["modality"],
            "encoder": spec["encoder"],
            "feature_dim": int(model.feature_dim),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": file_sha256(source),
            "augmentation": "none",
        }, target)
        written[split] = str(target)
    write_json(output / "completed.json", {
        "condition": condition,
        "participant": args.participant,
        "seed": args.seed,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "feature_dim": int(model.feature_dim),
        "outputs": written,
    })


if __name__ == "__main__":
    main()
