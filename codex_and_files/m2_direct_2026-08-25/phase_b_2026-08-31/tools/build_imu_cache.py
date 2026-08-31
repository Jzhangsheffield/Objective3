from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_b.config import load_config
from phase_b.data import build_imu_caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-free right-hand IMU cache")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = build_imu_caches(
        config["dataset_root"], args.train_manifest, args.test_manifest, args.output_dir,
        target_length=int(config["imu"]["target_length"]), overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
