from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_a.models import PhaseAM2Direct


def synthetic_batch(batch_size=2, history=4):
    available = torch.ones(batch_size, dtype=torch.bool)
    history_available = torch.ones(batch_size, history, dtype=torch.bool)
    return {
        "current_primary": torch.randn(batch_size, 512),
        "current_secondary": torch.randn(batch_size, 512),
        "current_emg": torch.randn(batch_size, 8, 512),
        "current_imu": torch.randn(batch_size, 6, 256),
        "current_secondary_available": available.clone(),
        "current_emg_available": available.clone(), "current_imu_available": available.clone(),
        "history_primary": torch.randn(batch_size, history, 512),
        "history_secondary": torch.randn(batch_size, history, 512),
        "history_emg": torch.randn(batch_size, history, 8, 512),
        "history_imu": torch.randn(batch_size, history, 6, 256),
        "history_secondary_available": history_available.clone(),
        "history_emg_available": history_available.clone(), "history_imu_available": history_available.clone(),
        "history_position_ids": torch.arange(history, 0, -1).repeat(batch_size, 1),
        "history_padding_mask": torch.zeros(batch_size, history, dtype=torch.bool),
    }


def main() -> None:
    for condition in ("A1", "A3", "A4", "A5", "A6", "A7"):
        model = PhaseAM2Direct(condition)
        batch = synthetic_batch()
        logits, diagnostics = model(batch)
        assert logits.shape == (2, 35)
        logits.square().mean().backward()
        if condition != "A1":
            model.eval()
            missing = synthetic_batch()
            for key in ("secondary", "emg", "imu"):
                missing[f"current_{key}_available"].zero_()
                missing[f"history_{key}_available"].zero_()
            missing_logits, _ = model(missing)
            assert torch.isfinite(missing_logits).all()
        print(f"PASS {condition}")


if __name__ == "__main__":
    main()
