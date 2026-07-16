from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print MindRove normalization arrays as KEY=VALUE lines for a BAT script."
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--signal", required=True, choices=("emg", "imu"))
    parser.add_argument("--target-len", required=True, type=int)
    return parser.parse_args()


def validate(values: object, expected: int, name: str, positive: bool = False) -> list[float]:
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains NaN or Inf")
    if positive and not all(value > 0.0 for value in result):
        raise ValueError(f"{name} must contain only positive values")
    return result


def format_values(values: list[float]) -> str:
    return " ".join(format(value, ".17g") for value in values)


def main() -> None:
    args = parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    length_key = str(args.target_len)
    if args.signal == "emg":
        valid_lengths = (256, 512, 1024, 2048)
        if args.target_len not in valid_lengths:
            raise ValueError(f"Unsupported EMG target length: {args.target_len}")
        groups = data["emg"][length_key]["groups"]
        left = groups["left_emg"]
        right = groups["right_emg"]
        expected_channels = 8
    else:
        valid_lengths = (64, 128, 256, 512)
        if args.target_len not in valid_lengths:
            raise ValueError(f"Unsupported IMU target length: {args.target_len}")
        groups = data["imu"][length_key]["groups"]
        left = groups["left_imu"]
        right = groups["right_imu"]
        expected_channels = 6

    left_mean = validate(left["mean"], expected_channels, "left mean")
    left_std = validate(left["std"], expected_channels, "left std", positive=True)
    right_mean = validate(right["mean"], expected_channels, "right mean")
    right_std = validate(right["std"], expected_channels, "right std", positive=True)

    print(f"LEFT_SIGNAL_MEAN={format_values(left_mean)}")
    print(f"LEFT_SIGNAL_STD={format_values(left_std)}")
    print(f"RIGHT_SIGNAL_MEAN={format_values(right_mean)}")
    print(f"RIGHT_SIGNAL_STD={format_values(right_std)}")


if __name__ == "__main__":
    main()
