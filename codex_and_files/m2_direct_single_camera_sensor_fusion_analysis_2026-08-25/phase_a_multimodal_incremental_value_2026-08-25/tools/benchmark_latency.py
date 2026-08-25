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
from torch.utils.data import DataLoader

from phase_a.config import load_config, validate_condition
from phase_a.data import MultimodalHistoryDataset, collate_multimodal
from phase_a.io import write_json
from phase_a.models import PhaseAM2Direct
from phase_a.paths import model_dir, primary_feature_cache, protocol_dir, secondary_feature_cache, signal_cache


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cached-feature fusion/head latency (not RGB decoding/backbones)")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", default="A", choices=list("ADJM"))
    parser.add_argument("--seed", default=1, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    condition = validate_condition(args.condition)
    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(model_dir(config, condition, args.participant, args.seed) / "last.pth",
                            map_location="cpu", weights_only=False)
    model = PhaseAM2Direct(
        condition, config["feature_dim"], config["d_model"], config["num_heads"],
        config["max_history"], config["dropout"], config["modality_dropout"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    protocols = protocol_dir(config, args.participant)
    secondary_path = (secondary_feature_cache(config, args.participant, args.seed, "test")
                      if condition in {"A1", "A3", "A7"}
                      else primary_feature_cache(config, args.participant, args.seed, "test"))
    dataset = MultimodalHistoryDataset(
        primary_feature_cache(config, args.participant, args.seed, "test"),
        secondary_path,
        signal_cache(config, args.participant, "test"), protocols / "test_all.jsonl",
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_multimodal)))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    latency_config = config["latency"]
    with torch.inference_mode():
        for _ in range(latency_config["warmup_iterations"]):
            model(batch)
        if device.type == "cuda": torch.cuda.synchronize()
        samples = []
        for _ in range(latency_config["measured_iterations"]):
            started = time.perf_counter()
            model(batch)
            if device.type == "cuda": torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
    result = {
        "scope": "cached RGB features + right signal encoders + fusion + M2 history + node head",
        "excluded": ["RGB tensor loading", "RGB temporal/spatial preprocessing", "one or two ResNet3D backbones"],
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(), "iterations": len(samples),
        "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95), "p99_ms": percentile(samples, 0.99),
        "clips_per_second": 1000.0 / statistics.mean(samples),
        "target_p95_ms": latency_config["target_p95_ms"],
        "target_min_clips_per_second": latency_config["target_min_clips_per_second"],
        "acceptance_status": "UNSET" if latency_config["target_p95_ms"] is None or latency_config["target_min_clips_per_second"] is None else "CHECK_REQUIRED",
    }
    output = model_dir(config, condition, args.participant, args.seed) / "latency_cached_feature_scope.json"
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
