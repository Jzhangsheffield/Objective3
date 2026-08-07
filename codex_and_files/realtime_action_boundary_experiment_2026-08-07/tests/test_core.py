from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_experiment.metrics import match_events, segments_from_binary
from boundary_experiment.models import CausalBoundaryTCN
from boundary_experiment.online import CausalBoundaryStateMachine


class CoreTests(unittest.TestCase):
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
