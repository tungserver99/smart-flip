import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantization.flatquant import _ActivationCollector


class ActivationCollectorTests(unittest.TestCase):
    def test_collector_limits_stored_rows_but_keeps_exact_running_mean(self):
        collector = _ActivationCollector(sample_limit=3, track_mean=True, storage_dtype=torch.float16)

        collector.add(torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], dtype=torch.float32))
        collector.add(torch.tensor([[[9.0, 11.0], [13.0, 15.0]]], dtype=torch.float32))

        samples = collector.get_sample_rows()
        mean = collector.get_mean()

        self.assertEqual(samples.shape, (3, 2))
        self.assertEqual(samples.dtype, torch.float32)
        self.assertTrue(torch.allclose(mean, torch.tensor([7.0, 9.0], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
