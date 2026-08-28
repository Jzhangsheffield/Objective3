from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.config import load_config, validate_condition
from phase_a.data import MultimodalHistoryDataset, collate_multimodal
from phase_a.engine import evaluate
from phase_a.io import write_json
from phase_a.models import PhaseAM2Direct
from phase_a.paths import model_dir, primary_feature_cache, protocol_dir, secondary_feature_cache, signal_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Missing-modality and sensor time-offset stress tests")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    condition = validate_condition(args.condition)
    if condition not in {"A3", "A4", "A5", "A6", "A7"}:
        raise ValueError("Stress suite applies to learned multimodal A3-A7 models")
    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(model_dir(config, condition, args.participant, args.seed) / "last.pth",
                            map_location="cpu", weights_only=False)
    model = PhaseAM2Direct(
        condition, config["feature_dim"], config["d_model"], config["num_heads"],
        config["max_history"], config["dropout"], config["modality_dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    protocols = protocol_dir(config, args.participant)
    scenarios = [("clean", (), 0.0, 0.0), ("secondary_missing", ("secondary",), 0.0, 0.0),
                 ("emg_missing", ("emg",), 0.0, 0.0), ("imu_missing", ("imu",), 0.0, 0.0),
                 ("wearables_missing", ("emg", "imu"), 0.0, 0.0),
                 ("all_new_modalities_missing", ("secondary", "emg", "imu"), 0.0, 0.0)]
    for fraction in config["stress_time_offsets_fraction"]:
        if condition in {"A5", "A6", "A7"}:
            scenarios.append((f"emg_offset_{fraction:+.2f}", (), fraction, 0.0))
        if condition in {"A4", "A6", "A7"}:
            scenarios.append((f"imu_offset_{fraction:+.2f}", (), 0.0, fraction))
        if condition in {"A6", "A7"}:
            scenarios.append((f"both_sensors_offset_{fraction:+.2f}", (), fraction, fraction))
    for name, dropped, emg_offset, imu_offset in scenarios:
        for split, manifest in (("test_all", "test_all.jsonl"), ("test_fault", "test_fault.jsonl")):
            secondary_path = (secondary_feature_cache(config, args.participant, args.seed, "test")
                              if condition in {"A3", "A7"}
                              else primary_feature_cache(config, args.participant, args.seed, "test"))
            dataset = MultimodalHistoryDataset(
                primary_feature_cache(config, args.participant, args.seed, "test"),
                secondary_path,
                signal_cache(config, args.participant, "test"), protocols / manifest,
                tuple(dropped), 0.0, emg_offset, imu_offset,
            )
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_multimodal)
            evaluate(model, loader, device, checkpoint["node_to_tier3"],
                     model_dir(config, condition, args.participant, args.seed) / "stress_results" / name, split)
        print(f"completed stress scenario: {name}", flush=True)
    write_json(model_dir(config, condition, args.participant, args.seed) / "stress_completed.json", {
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "scenarios": [name for name, _, _, _ in scenarios],
        "splits": ["test_all", "test_fault"],
    })


if __name__ == "__main__":
    main()
