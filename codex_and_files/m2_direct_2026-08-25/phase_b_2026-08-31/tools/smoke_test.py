from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_b.calibration import QualityGate, apply_quality_gate, apply_static_fusion, fit_static_fusion
from phase_b.models import IMUResNet10, JointFusionModel, symmetric_contrastive_loss


def batch(batch_size: int = 3, history: int = 2) -> dict[str, torch.Tensor]:
    result = {"available": torch.ones(batch_size, 3, dtype=torch.bool)}
    for name, temporal_length, temporal_dim in (("cam0", 4, 128), ("cam1", 4, 128), ("imu", 8, 512)):
        result[f"current_{name}_global"] = torch.randn(batch_size, 512)
        result[f"current_{name}_temporal"] = torch.randn(batch_size, temporal_length, temporal_dim)
        result[f"history_{name}_global"] = torch.randn(batch_size, history, 512)
        result[f"history_{name}_temporal"] = torch.randn(batch_size, history, temporal_length, temporal_dim)
    result["history_padding_mask"] = torch.zeros(batch_size, history, dtype=torch.bool)
    result["history_position_ids"] = torch.arange(history, 0, -1).repeat(batch_size, 1)
    result["node_target"] = torch.arange(batch_size) % 35
    return result


def main() -> None:
    torch.manual_seed(7)
    imu = IMUResNet10()
    assert imu(torch.randn(3, 6, 256)).shape == (3, 35)
    assert imu.forward_tokens(torch.randn(3, 6, 256)).shape == (3, 8, 512)
    for condition, use_history, alignment in (("B3", False, False), ("B4", True, False), ("B5", True, True)):
        model = JointFusionModel(
            {"cam0": 128, "cam1": 128, "imu": 512}, use_history=use_history,
            modality_dropout=0.2, d_model=64, num_heads=4, bottleneck_tokens=2,
            layers=1, soft_alignment=alignment, output_dim=128, num_nodes=35, max_history=35,
        )
        logits, diagnostics = model(batch())
        assert logits.shape == (3, 35), condition
        assert set(diagnostics["unimodal_logits"]) == {"cam0", "cam1", "imu"}
        symmetric_contrastive_loss(diagnostics["camera_embedding"], diagnostics["imu_embedding"], 0.1).backward()
    probabilities = torch.softmax(torch.randn(24, 3, 35), dim=-1)
    targets = torch.randint(0, 35, (24,))
    fitted = fit_static_fusion(probabilities, targets, steps=3)
    assert apply_static_fusion(probabilities, fitted["log_temperature"], fitted["weight_logits"]).shape == (24, 35)
    gate = QualityGate()
    fused, weights = apply_quality_gate(gate, probabilities, fitted["log_temperature"])
    assert fused.shape == (24, 35) and weights.shape == (24, 3)
    print("Phase B synthetic smoke test: PASS")


if __name__ == "__main__":
    main()
