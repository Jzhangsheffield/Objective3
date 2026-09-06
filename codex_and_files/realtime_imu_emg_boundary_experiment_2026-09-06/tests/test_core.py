from __future__ import annotations

import unittest

import numpy as np
import torch

from sensor_boundary.annotations import targets_on_grid
from sensor_boundary.models import CausalSensorBoundaryModel
from sensor_boundary.online import run_state_machine
from sensor_boundary.preprocessing import _causal_previous_sample, _make_windows
from sensor_boundary.utils import parse_camera_timestamp


class CoreTests(unittest.TestCase):
    def test_timestamp_matches_board_epoch(self):
        self.assertAlmostEqual(parse_camera_timestamp("20260313_093657_246168"), 1773394617.246168, places=5)

    def test_previous_sample_resampling_is_causal(self):
        times = np.asarray([1.0, 2.0, 3.0])
        values = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)
        result, valid = _causal_previous_sample(times, values, np.asarray([1.5, 2.5]), 1.0)
        np.testing.assert_array_equal(result[:, 0], [10.0, 20.0])
        np.testing.assert_array_equal(valid, [1.0, 1.0])

    def test_local_windows_never_contain_future_samples(self):
        signals = np.arange(6, dtype=np.float32)[:, None]
        windows = _make_windows(signals, np.asarray([0, 2, 5]), 3)
        np.testing.assert_array_equal(windows[:, :, 0], [[0, 0, 0], [0, 1, 2], [3, 4, 5]])

    def test_adjacent_action_segments_keep_two_boundaries(self):
        annotation = {
            "segments": [
                {"segment_no": 1, "action": "grip", "object": "x", "start": 1.0, "end": 1.4},
                {"segment_no": 2, "action": "place", "object": "x", "start": 1.5, "end": 1.9},
            ]
        }
        targets = targets_on_grid(annotation, np.arange(1.0, 2.0, 0.1), 0)
        self.assertEqual(int(targets["exact_start"].sum()), 2)
        self.assertEqual(int(targets["exact_end"].sum()), 2)
        self.assertEqual(set(targets["segment_id"].tolist()), {1, 2})

    def test_pending_merge_emits_one_segment(self):
        state = [0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]
        starts = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
        ends = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
        segments = run_state_machine(state, starts, ends, {
            "start_threshold": 0.55,
            "end_threshold": 0.55,
            "action_threshold": 0.55,
            "start_debounce_steps": 1,
            "end_debounce_steps": 1,
            "min_action_steps": 1,
            "merge_gap_steps": 2,
        })
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0]["start_index"], segments[0]["end_index"]), (0, 6))
        self.assertEqual(segments[0]["emitted_at_index"], 8)

    def test_model_shape(self):
        cfg = {
            "input_channels": 14,
            "local_encoder_channels": [8, 16],
            "local_feature_dim": 12,
            "tcn_hidden_dim": 16,
            "tcn_num_layers": 2,
            "tcn_kernel_size": 3,
            "dropout": 0.0,
        }
        model = CausalSensorBoundaryModel(cfg)
        output = model(torch.randn(2, 7, 50, 14))
        self.assertEqual(tuple(output["state_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["start_logits"].shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
