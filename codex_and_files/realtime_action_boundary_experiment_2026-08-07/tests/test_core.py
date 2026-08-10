from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchvision.io import write_png

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_experiment.metrics import match_events, segments_from_binary
from boundary_experiment.annotations import RunInfo
from boundary_experiment.features import extract_run_features, load_feature_cache
from boundary_experiment.models import CausalBoundaryTCN
from boundary_experiment.online import CausalBoundaryStateMachine


class DummyBackbone(torch.nn.Module):
    def forward_features(self, clips):
        return clips.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)


class CoreTests(unittest.TestCase):
    def test_multiworker_frame_loader_preserves_causal_anchor_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            camera = root / "camera"
            camera.mkdir()
            rows = []
            for index in range(12):
                name = f"frame_{index:03d}.png"
                write_png(torch.full((3, 8, 8), index * 10, dtype=torch.uint8), str(camera / name))
                rows.append({
                    "frame_idx": index + 1, "original_frame_idx": index + 101,
                    "frame_name": name, "timestamp": f"t{index:03d}",
                    "action": "background", "object": "none", "mark": "none",
                    "segment_no": 1, "segment_start_idx": 101, "segment_end_idx": 112,
                    "segment_start": "t000", "segment_end": "t011",
                })
            annotation = root / "frames.csv"
            with annotation.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            checkpoint = root / "dummy.pth"
            checkpoint.write_bytes(b"dummy")
            info = RunInfo("run_test", "A", "run_1", root, camera, annotation)
            output = root / "features.pt"
            config = {
                "stride_frames": 2, "batch_size": 3, "clip_frames": 4,
                "num_workers": 2, "frame_loader_batch_size": 4,
                "prefetch_factor": 2, "persistent_workers": True,
                "pin_memory": False, "rgb_size": 8,
                "boundary_label_radius_frames": 0,
                "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
            }
            extract_run_features(info, DummyBackbone(), output, torch.device("cpu"), config, checkpoint)
            cache = load_feature_cache(output)
            self.assertEqual(cache["anchor_row_index"].tolist(), [0, 2, 4, 6, 8, 10])
            self.assertEqual(tuple(cache["features"].shape), (6, 1))
            self.assertAlmostEqual(float(cache["features"][1, 0]), 7.5 / 255.0, places=5)
            self.assertEqual(cache["metadata"]["frame_loader_num_workers"], 2)

    def test_causal_prefix_invariance(self):
        torch.manual_seed(1)
        model = CausalBoundaryTCN(feature_dim=8, hidden_dim=16, num_layers=3, dropout=0.0).eval()
        prefix = torch.randn(1, 20, 8)
        full = torch.cat([prefix, torch.randn(1, 10, 8)], dim=1)
        with torch.no_grad():
            short = model(prefix)["state_logits"]
            long = model(full)["state_logits"][:, :20]
        self.assertTrue(torch.allclose(short, long, atol=1e-6))

    def test_event_matching_is_one_to_one(self):
        result = match_events([10], [9, 11], tolerance=2)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)

    def test_segments(self):
        self.assertEqual(segments_from_binary([0, 1, 1, 0, 1]), [(1, 2), (4, 4)])

    def test_state_machine_preserves_gap_when_merge_disabled(self):
        machine = CausalBoundaryStateMachine(start_debounce=1, end_debounce=1, min_action_steps=1, merge_gap_steps=0)
        emitted = []
        values = [(0.9, 0.9, 0.0), (0.1, 0.0, 0.9), (0.1, 0.0, 0.0), (0.9, 0.9, 0.0), (0.1, 0.0, 0.9)]
        for i, row in enumerate(values): emitted.extend(machine.update(i, *row))
        self.assertEqual([(x.start_index, x.end_index) for x in emitted], [(0, 1), (3, 4)])


if __name__ == "__main__":
    unittest.main()
