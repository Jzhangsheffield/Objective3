import unittest
import numpy as np
from boundary_experiment.metrics import evaluate_run

class V3SegmentTests(unittest.TestCase):
    def test_adjacent_actions_remain_separate(self):
        state = np.ones(6, dtype=int)
        metrics = evaluate_run(state, state, [0, 3], [2, 5], [(0, 2), (3, 5)], [0],
                               target_segment_ids=np.array([1, 1, 1, 2, 2, 2]))
        self.assertEqual(metrics['gt_segment_count'], 2)
        self.assertEqual(metrics['segmental_f1']['50']['f1'], 1.0)
