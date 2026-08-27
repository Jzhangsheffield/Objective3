from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.bilateral_cache import build_bilateral_signal_caches
from phase_a.supplementary import base_protocol_dir, bilateral_cache_dir, load_supplementary_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free bilateral EMG/IMU caches")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "bilateral_supplementary_experiments.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_supplementary_config(args.config)
    if config.get("signal_data", {}).get("scope") != "bilateral":
        raise ValueError("build_bilateral_signal_cache requires signal_data.scope=bilateral")
    protocols = base_protocol_dir(config, args.participant)
    result = build_bilateral_signal_caches(
        config["base"]["dataset_root"], protocols / "train.jsonl", protocols / "test_all.jsonl",
        protocols / "test_normal.jsonl", protocols / "test_fault.jsonl",
        bilateral_cache_dir(config, args.participant), args.participant, config["signal_data"],
        int(config["base"]["emg_target_length"]), int(config["base"]["imu_target_length"]), args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
