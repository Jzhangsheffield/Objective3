from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.engine import evaluate
from phase_a.io import write_json
from phase_a.sensor_data import SignalClipDataset, SignalFeatureHistoryDataset, collate_feature_history, collate_signal
from phase_a.supplementary import (
    base_protocol_dir,
    base_signal_cache,
    direct_num_classes,
    experiment_spec,
    load_supplementary_config,
    signal_channels,
    supplementary_model_dir,
    validate_supplementary_condition,
)
from phase_a.supplementary_engine import evaluate_direct
from phase_a.supplementary_models import SensorM2Direct, SignalDirectClassifier


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def materialize_features(model, loader, device: torch.device, target: Path, metadata: dict) -> None:
    model.eval()
    features = []
    records = []
    for raw_batch in loader:
        features.append(model.forward_features(raw_batch["signal"].to(device)).cpu())
        for index, name in enumerate(raw_batch["sample_name"]):
            records.append({
                "sample_name": name,
                "participant": raw_batch["participant"][index],
                "run": raw_batch["run"][index],
                "annotation_row_index": raw_batch["annotation_row_index"][index],
            })
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.cat(features), "records": records, **metadata}, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-offset and zero-signal stress tests for S1-S12")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_supplementary_config(args.config)
    condition = validate_supplementary_condition(config, args.condition)
    spec = experiment_spec(config, condition)
    device = select_device(args.device)
    output = supplementary_model_dir(config, condition, args.participant, args.seed)
    checkpoint = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
    protocols = base_protocol_dir(config, args.participant)
    batch_size = int(config["direct_training"]["batch_size"])
    scenarios = [("clean", 0.0, False), ("zero_signal", 0.0, True)]
    scenarios.extend((f"offset_{fraction:+.2f}", float(fraction), False)
                     for fraction in config["base"]["stress_time_offsets_fraction"])

    if spec["task"] in {"direct_node", "direct_tier3"}:
        model = SignalDirectClassifier(
            config, spec["encoder"], signal_channels(spec["modality"]), direct_num_classes(spec["task"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        for scenario, offset, zero in scenarios:
            for split, manifest in (("test_all", "test_all.jsonl"), ("test_fault", "test_fault.jsonl")):
                dataset = SignalClipDataset(
                    base_signal_cache(config, args.participant, "test"), protocols / manifest,
                    spec["modality"], fixed_offset_fraction=offset, zero_signal=zero,
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_signal)
                evaluate_direct(model, loader, device, spec["task"], checkpoint["node_to_tier3"],
                                output / "stress_results" / scenario, split)
            print(f"completed {condition} stress scenario {scenario}", flush=True)
    else:
        upstream = str(spec["upstream"])
        upstream_spec = experiment_spec(config, upstream)
        upstream_checkpoint_path = supplementary_model_dir(
            config, upstream, args.participant, args.seed
        ) / "last.pth"
        upstream_checkpoint = torch.load(upstream_checkpoint_path, map_location="cpu", weights_only=False)
        encoder = SignalDirectClassifier(
            config, upstream_spec["encoder"], signal_channels(upstream_spec["modality"]), 31
        ).to(device)
        encoder.load_state_dict(upstream_checkpoint["model"], strict=True)
        model = SensorM2Direct(
            feature_dim=int(config["signal_feature_dim"]), d_model=int(config["base"]["d_model"]),
            num_heads=int(config["base"]["num_heads"]), max_history=int(config["base"]["max_history"]),
            dropout=float(config["base"]["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        for scenario, offset, zero in scenarios:
            feature_path = output / "stress_feature_caches" / f"{scenario}.pt"
            signal_dataset = SignalClipDataset(
                base_signal_cache(config, args.participant, "test"), protocols / "test_all.jsonl",
                upstream_spec["modality"], fixed_offset_fraction=offset, zero_signal=zero,
            )
            signal_loader = DataLoader(signal_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=args.num_workers, collate_fn=collate_signal)
            materialize_features(encoder, signal_loader, device, feature_path, {
                "scenario": scenario, "offset_fraction": offset, "zero_signal": zero,
                "upstream": upstream,
            })
            for split, manifest in (("test_all", "test_all.jsonl"), ("test_fault", "test_fault.jsonl")):
                dataset = SignalFeatureHistoryDataset(feature_path, protocols / manifest)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_feature_history)
                evaluate(model, loader, device, checkpoint["node_to_tier3"],
                         output / "stress_results" / scenario, split)
            print(f"completed {condition} stress scenario {scenario}", flush=True)
    write_json(output / "stress_completed.json", {
        "condition": condition,
        "participant": args.participant,
        "seed": args.seed,
        "scenarios": [name for name, _, _ in scenarios],
        "interpretation": "zero_signal is degradation-only; sensor-only models have no A0 fallback path",
    })


if __name__ == "__main__":
    main()
