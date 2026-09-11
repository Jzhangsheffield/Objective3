from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from sensor_boundary.annotations import load_annotation, load_run_index
from sensor_boundary.config import load_config
from sensor_boundary.preprocessing import EMG_CHANNELS, IMU_CHANNELS
from sensor_boundary.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation of MindRove files and annotations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--deep", action="store_true", help="Read full timestamp columns for every run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"])
    modality = cfg["sensor"]["modality"]
    channels = IMU_CHANNELS if modality == "imu" else EMG_CHANNELS
    required = {"board_ts", *channels}
    problems: list[str] = []
    participants: Counter[str] = Counter()
    checked = 0
    for info in index.values():
        participants[info.participant] += 1
        try:
            annotation = load_annotation(info)
            for path in (info.left_csv, info.right_csv):
                if not path.is_file():
                    raise FileNotFoundError(path)
                header = set(pd.read_csv(path, nrows=0).columns)
                if not required <= header:
                    raise ValueError(f"Missing columns {sorted(required - header)} in {path}")
                if args.deep:
                    timestamps = pd.read_csv(path, usecols=["board_ts"])["board_ts"].dropna()
                    if len(timestamps) < 2 or not timestamps.is_monotonic_increasing:
                        raise ValueError(f"Invalid timestamps in {path}")
                    if float(timestamps.iloc[-1]) < annotation["frame_start"] or float(timestamps.iloc[0]) > annotation["frame_end"]:
                        raise ValueError(f"No sensor/annotation overlap in {path}")
            checked += 1
        except Exception as error:
            problems.append(f"{info.sample_name}: {error}")
    source = Path(cfg["paths"]["protocol_source_root"])
    if not source.is_dir():
        problems.append(f"Missing protocol_source_root: {source}")
    report = {
        "status": "ok" if not problems else "failed",
        "modality": modality,
        "runs": len(index),
        "checked_runs": checked,
        "participants": dict(participants),
        "deep": args.deep,
        "problems": problems,
    }
    target = Path(cfg["paths"]["validation_root"]) / modality / "setup_validation.json"
    write_json(target, report)
    print(report)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
