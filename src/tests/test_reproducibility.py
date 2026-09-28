import os
import unittest
from unittest.mock import patch

from src.utils.reproducibility import set_global_seed


class ReproducibilityTests(unittest.TestCase):
    def test_deterministic_seed_sets_cublas_workspace_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            set_global_seed(17, deterministic=True)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")


if __name__ == "__main__":
    unittest.main()
