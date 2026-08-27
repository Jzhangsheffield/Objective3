from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_a.sensor_data import SignalClipDataset, SignalFeatureHistoryDataset, collate_feature_history, collate_signal
from phase_a.supplementary import (
    direct_num_classes,
    evaluation_protocols,
    experiment_spec,
    load_supplementary_config,
    signal_channels,
    supplementary_feature_cache,
    supplementary_model_dir,
    validate_supplementary_condition,
)
from phase_a.supplementary_models import SensorM2Direct, SignalDirectClassifier


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end signal encoder + direct/M2 head latency")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "supplementary_experiments.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", default="A", choices=list("ADJM"))
    parser.add_argument("--seed", default=1, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_supplementary_config(args.config)
    condition = validate_supplementary_condition(config, args.condition)
    spec = experiment_spec(config, condition)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    output = supplementary_model_dir(config, condition, args.participant, args.seed)
    checkpoint = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
    protocol = evaluation_protocols(config, args.participant)[0]
    signal_dataset = SignalClipDataset(
        protocol["cache"], protocol["manifest_dir"] / "test_all.jsonl", spec["modality"]
    )
    raw = collate_signal([signal_dataset[0]])
    signal = raw["signal"].to(device)
    signal_scope = "bilateral" if config.get("signal_data", {}).get("scope") == "bilateral" else "right-hand"
    if spec["task"] in {"direct_node", "direct_tier3"}:
        encoder = SignalDirectClassifier(
            config, spec["encoder"], signal_channels(spec["modality"], config), direct_num_classes(spec["task"])
        ).to(device).eval()
        encoder.load_state_dict(checkpoint["model"], strict=True)

        def forward():
            return encoder(signal)

        scope = f"{signal_scope} signal encoder + direct classification head"
    else:
        upstream = str(spec["upstream"])
        upstream_spec = experiment_spec(config, upstream)
        upstream_checkpoint = torch.load(
            supplementary_model_dir(config, upstream, args.participant, args.seed) / "last.pth",
            map_location="cpu", weights_only=False,
        )
        encoder = SignalDirectClassifier(
            config, upstream_spec["encoder"], signal_channels(upstream_spec["modality"], config), 31
        ).to(device).eval()
        encoder.load_state_dict(upstream_checkpoint["model"], strict=True)
        m2 = SensorM2Direct(
            feature_dim=int(config["signal_feature_dim"]), d_model=int(config["base"]["d_model"]),
            num_heads=int(config["base"]["num_heads"]), max_history=int(config["base"]["max_history"]),
            dropout=float(config["base"]["dropout"]),
        ).to(device).eval()
        m2.load_state_dict(checkpoint["model"], strict=True)
        feature_dataset = SignalFeatureHistoryDataset(
            supplementary_feature_cache(config, upstream, args.participant, args.seed, protocol["feature_split"]),
            protocol["manifest_dir"] / "test_all.jsonl",
        )
        feature_batch = collate_feature_history([feature_dataset[0]])
        feature_batch = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in feature_batch.items()}

        def forward():
            current_feature = encoder.forward_features(signal)
            batch = dict(feature_batch)
            batch["current_feature"] = current_feature
            return m2(batch)

        scope = f"current {signal_scope} signal encoder + cached prior-history features + M2 + node head"
    latency = config["base"]["latency"]
    with torch.inference_mode():
        for _ in range(int(latency["warmup_iterations"])):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(int(latency["measured_iterations"])):
            started = time.perf_counter()
            forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
    result = {
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "evaluation_protocol_for_example": protocol["name"],
        "scope": scope, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(), "iterations": len(samples),
        "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95), "p99_ms": percentile(samples, 0.99),
        "clips_per_second": 1000.0 / statistics.mean(samples),
        "target_p95_ms": latency["target_p95_ms"],
        "target_min_clips_per_second": latency["target_min_clips_per_second"],
        "acceptance_status": "UNSET" if latency["target_p95_ms"] is None or latency["target_min_clips_per_second"] is None else "CHECK_REQUIRED",
    }
    from phase_a.io import write_json
    write_json(output / "latency_end_to_end_signal_scope.json", result)
    print(result)


if __name__ == "__main__":
    main()
