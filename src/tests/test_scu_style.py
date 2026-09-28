import unittest

import torch
from torch.utils.data import TensorDataset

from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.utils.scu import SCUConfig, adapt_scu_style


class SCUStyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = VariationalDeepJSCC(
            VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
        )
        self.seeds = {
            "data_order": 11,
            "posterior": 12,
            "channel": 13,
        }

    def test_forget_only_uses_only_the_declared_forgetting_loss(self):
        target = TensorDataset(
            torch.zeros(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)
        )
        retained = TensorDataset(
            torch.ones(4, 3, 32, 32), torch.ones(4, dtype=torch.long)
        )

        result = adapt_scu_style(
            original=self.model,
            target_dataset=target,
            retained_dataset=retained,
            config=SCUConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-4,
                lambda_kl=1.0,
                lambda_rec=0.1,
                lambda_ss=9.0,
                beta=1e-4,
                epsilon=1e-8,
                snr_db=10.0,
            ),
            device=torch.device("cpu"),
            mode="forget_only",
            seeds=self.seeds,
        )

        self.assertEqual(result.steps, 1)
        self.assertEqual(result.loss_trace[0]["mode"], "forget_only")
        self.assertEqual(result.loss_trace[0]["optimized_terms"], ["forget_loss"])


if __name__ == "__main__":
    unittest.main()
