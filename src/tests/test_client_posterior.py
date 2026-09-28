import unittest

import torch

from src.evaluation.posterior_audit import (
    fit_moment_matched_diagonal_gaussian,
    rbf_mmd_squared,
)


class ClientPosteriorTests(unittest.TestCase):
    def test_moment_matched_variance_contains_posterior_and_between_sample_variance(self):
        mu = torch.tensor([[0.0], [2.0]])
        logvar = torch.log(torch.tensor([[1.0], [1.0]]))

        summary = fit_moment_matched_diagonal_gaussian(
            mu, logvar, minimum_variance=1e-6
        )

        self.assertAlmostEqual(summary.mu.item(), 1.0, places=6)
        self.assertAlmostEqual(summary.logvar.exp().item(), 2.0, places=6)

    def test_single_gaussian_gap_exceeds_identical_empirical_draw_noise(self):
        empirical = torch.tensor([[-4.0], [-4.0], [4.0], [4.0]])
        single_gaussian_samples = torch.zeros_like(empirical)

        empirical_noise = rbf_mmd_squared(empirical, empirical)
        single_gap = rbf_mmd_squared(empirical, single_gaussian_samples)

        self.assertEqual(empirical_noise.item(), 0.0)
        self.assertGreater(single_gap.item(), empirical_noise.item())


if __name__ == "__main__":
    unittest.main()
