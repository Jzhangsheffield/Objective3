from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_a.supplementary import experiment_spec, load_supplementary_config, signal_channels
from phase_a.supplementary_models import SensorM2Direct, SignalDirectClassifier


def main() -> None:
    config = load_supplementary_config(PACKAGE_ROOT / "config" / "supplementary_experiments.json")
    for condition in (f"S{index}" for index in range(5, 13)):
        spec = experiment_spec(config, condition)
        classes = 35 if spec["task"] == "direct_node" else 31
        length = int(config["base"]["emg_target_length" if spec["modality"] == "emg" else "imu_target_length"])
        model = SignalDirectClassifier(config, spec["encoder"], signal_channels(spec["modality"]), classes)
        signal = torch.randn(2, signal_channels(spec["modality"]), length)
        feature = model.forward_features(signal)
        logits = model(signal)
        assert feature.shape == (2, 512), (condition, feature.shape)
        assert logits.shape == (2, classes), (condition, logits.shape)
        logits.square().mean().backward()
        print(f"PASS {condition}: feature={tuple(feature.shape)}, logits={tuple(logits.shape)}")
    for history_length in (0, 4):
        model = SensorM2Direct()
        batch = {
            "current_feature": torch.randn(2, 512),
            "history_features": torch.randn(2, history_length, 512),
            "history_position_ids": torch.arange(history_length, 0, -1).repeat(2, 1),
            "history_padding_mask": torch.zeros(2, history_length, dtype=torch.bool),
        }
        logits, diagnostics = model(batch)
        assert logits.shape == (2, 35)
        assert diagnostics["fused_feature"].shape == (2, 512)
        logits.square().mean().backward()
        print(f"PASS SensorM2Direct history={history_length}")


if __name__ == "__main__":
    main()
