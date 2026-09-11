from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_run_index
from boundary_experiment.config import load_config
from boundary_experiment.protocols import prepare_boundary_protocols


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate existing Atomic-tail LOSO protocols to continuous runs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    runs = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    report = prepare_boundary_protocols(
        runs, cfg["paths"]["atomic_project_root"], cfg["paths"]["protocol_root"],
        cfg["data"]["camera_id"], list(cfg["data"]["participants"]),
    )
    print(f"Prepared {len(report['folds'])} LOSO folds at {cfg['paths']['protocol_root']}")


if __name__ == "__main__":
    main()
