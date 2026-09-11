import unittest
import torch
from sensor_boundary.data import collate_chunks

class ContextLossTests(unittest.TestCase):
    def test_context_and_padding_do_not_contribute(self):
        def row(length, context):
            return dict(windows=torch.zeros(length, 5, 2),state=torch.zeros(length),
                        start=torch.zeros(length),end=torch.zeros(length),
                        context_steps=context,sample_name='dummy',start_offset=0)
        batch=collate_chunks([row(8, 0),row(6, 3)])
        self.assertEqual(batch['mask'].sum(dim=1).tolist(), [8, 3])
        self.assertEqual(batch['mask'][1].tolist(), [False]*3+[True]*3+[False]*2)
