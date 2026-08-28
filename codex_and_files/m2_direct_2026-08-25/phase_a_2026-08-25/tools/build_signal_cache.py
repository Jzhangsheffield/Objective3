from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.cache import build_signal_caches
from phase_a.config import load_config
from phase_a.paths import protocol_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free right-hand EMG/IMU caches")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    protocols = protocol_dir(config, args.participant)
    output = Path(config["output_root"]) / "signal_cache" / f"{args.participant}_as_test"
    result = build_signal_caches(
        config["dataset_root"], protocols / "train.jsonl", protocols / "test_all.jsonl", output,
        config["emg_target_length"], config["imu_target_length"], args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
