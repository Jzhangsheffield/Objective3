from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from phase_b1_imu_m2.common import add_phase_b_to_path, load_config
from phase_b1_imu_m2.data import IMUFeatureHistoryDataset, collate_history
from phase_b1_imu_m2.model import IMUM2Direct


def main() -> None:
    config = load_config()
    add_phase_b_to_path(config)
    from phase_b.calibration import apply_static_fusion, fit_static_fusion

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        records = [
            {"sample_name": f"A_R1_{index}", "participant": "A", "run": "R1",
             "annotation_row_index": index, "node_idx": index + 1,
             "tier3_id": index, "stage_id": 0}
            for index in range(4)
        ]
        torch.save({"features": torch.randn(4, 512), "records": records}, root / "features.pt")
        (root / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        dataset = IMUFeatureHistoryDataset(root / "features.pt", root / "manifest.jsonl")
        batch = collate_history([dataset[index] for index in range(len(dataset))])
        assert batch["history_features"].shape == (4, 3, 512)
        assert batch["history_padding_mask"].tolist() == [
            [True, True, True], [False, True, True],
            [False, False, True], [False, False, False],
        ]
        model = IMUM2Direct()
        logits, diagnostics = model(batch)
        assert logits.shape == (4, 35)
        assert diagnostics["history_attention"].shape == (4, 4, 1, 4)
        loss = torch.nn.functional.cross_entropy(logits, batch["node_target"])
        loss.backward()
        assert torch.isfinite(loss)

    probabilities = torch.softmax(torch.randn(12, 3, 35), dim=-1)
    targets = torch.randint(0, 35, (12,))
    fitted = fit_static_fusion(probabilities, targets, steps=3)
    fused = apply_static_fusion(probabilities, fitted["log_temperature"], fitted["weight_logits"])
    assert fused.shape == (12, 35)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(12), atol=1e-5)
    print("PASS: config, actual-history dataset, IMU M2 forward/backward, and B1 fusion")


if __name__ == "__main__":
    main()
