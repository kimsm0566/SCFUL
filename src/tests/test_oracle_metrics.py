import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.models.semantic_codec import Posterior
from src.utils.artifacts import create_run_artifacts
from src.utils.metrics import symmetric_diagonal_kl


class OracleMetricTests(unittest.TestCase):
    def test_symmetric_kl_is_zero_for_equal_posteriors_and_positive_otherwise(self):
        posterior = Posterior(
            mu=torch.tensor([[0.0, 0.0]]), logvar=torch.tensor([[0.0, 0.0]])
        )
        shifted = Posterior(
            mu=torch.tensor([[1.0, 0.0]]), logvar=torch.tensor([[0.0, 0.0]])
        )

        self.assertEqual(symmetric_diagonal_kl(posterior, posterior).item(), 0.0)
        self.assertAlmostEqual(
            symmetric_diagonal_kl(posterior, shifted).item(), 0.5, places=6
        )

    def test_artifacts_allocate_distinct_run_directories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = create_run_artifacts(root, seed=3)
            second = create_run_artifacts(root, seed=3)

            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertTrue((first.run_dir / "checkpoints").is_dir())
            self.assertTrue((second.run_dir / "metrics").is_dir())


if __name__ == "__main__":
    unittest.main()
