from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from sensor_boundary.annotations import load_run_index
from sensor_boundary.config import load_config
from sensor_boundary.protocols import prepare_protocols


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and validate the established LOSO protocol")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"])
    report = prepare_protocols(
        index,
        cfg["paths"]["protocol_source_root"],
        cfg["paths"]["protocol_root"],
        list(cfg["data"]["participants"]),
    )
    print(f"Prepared {len(report['folds'])} LOSO folds at {cfg['paths']['protocol_root']}")


if __name__ == "__main__":
    main()
