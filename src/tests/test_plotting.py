import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.evaluation.posterior_audit import ClientGaussianSummary
from src.utils.plotting import (
    save_client_partition_figure,
    save_encoder_retrain_gap_figure,
    save_federated_classification_curve_figures,
    save_federated_training_curve_figures,
    save_oracle_figures,
    save_selected_core_metrics_figure,
    save_target_client_class_distribution_figure,
)


def make_minimal_oracle_prediction_fixture():
    inputs = torch.zeros(2, 1, 28, 28)
    references = [
        {
            "dataset_split": "mnist_train",
            "source_index": 0,
            "probe_partition": "target_train_member",
            "support_group": "shared_high",
        },
        {
            "dataset_split": "mnist_train",
            "source_index": 1,
            "probe_partition": "target_train_member",
            "support_group": "target_specific",
        },
    ]
    models = {
        model_name: {
            "mu": torch.zeros(2, 2),
            "posterior_mean_reconstruction": torch.zeros(2, 1, 28, 28),
        }
        for model_name in ("original", "retrain", "scu_forget_only", "scu_self_sharing")
    }
    predictions = {
        "target_train_member": {
            "references": references,
            "inputs": inputs,
            "models": models,
        }
    }
    summaries = [
        ClientGaussianSummary(
            client_id="client_0",
            mean=[0.0, 0.0],
            variance=[1.0, 1.0],
            fit_count=2,
            evaluation_count=1,
            metrics={
                "mmd_single": 0.2,
                "mmd_empirical_noise": 0.1,
                "mean_abs_offdiag_correlation": 0.0,
                "label_mixture_nll_gain_per_dim": 0.0,
            },
        )
    ]
    return predictions, summaries


class PlottingTests(unittest.TestCase):
    def test_selected_core_metric_figure_uses_nonempty_registered_rows(self):
        rows = [
            {
                "model": model,
                "output_pul_percent": pul,
                "output_marker_positive_percent": (50.0, 0.0, 1.0)[position],
                "retain_fid_to_real": 20.0 + position,
                "clean_psnr_db": 28.0 + position,
                "erased_psnr_db": 27.5 + position,
                "clean_ssim": 0.90 + 0.01 * position,
                "erased_ssim": 0.89 + 0.01 * position,
                "clean_dfil": idi + 1,
                "retain_dfil": idi + 2,
                "erased_dfil": idi + 0.5,
                "running_time_seconds": 10 + position,
            }
            for position, (model, pul, idi) in enumerate(
                (
                    ("original", 0.0, 1.0),
                    ("retrain_without_client_0", 100.0, 0.0),
                    ("FSCUv4", 80.0, 0.2),
                )
            )
        ]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with mock.patch("src.utils.plotting.plt.close") as close_figure:
                save_selected_core_metrics_figure(list(reversed(rows)), output_dir)
                figure = close_figure.call_args.args[0]
            try:
                expected_order = [row["model"] for row in rows]
                for axis in figure.axes:
                    self.assertEqual(
                        [tick.get_text() for tick in axis.get_xticklabels()], expected_order
                    )
                self.assertEqual(
                    [bar.get_height() for bar in figure.axes[0].patches],
                    [50.0, 0.0, 1.0],
                )
                self.assertEqual(len(figure.axes[3].containers), 2)
                self.assertEqual(len(figure.axes[4].containers), 2)
                self.assertIn("fixed-region", figure.axes[0].get_title())
            finally:
                import matplotlib.pyplot as plt
                plt.close(figure)
            output = output_dir / "selected_core_metrics.png"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_encoder_retrain_gap_figure_is_backed_by_all_four_metrics(self):
        rows = []
        for position, model in enumerate(("original", "retrain", "unlearned")):
            value = float(position) * 0.01
            rows.append(
                {
                    "model": model,
                    "clean_latent_mean_mse_to_retrain": value,
                    "erased_latent_mean_mse_to_retrain": value + 0.01,
                    "clean_posterior_symmetric_kl_to_retrain": value + 0.02,
                    "erased_posterior_symmetric_kl_to_retrain": value + 0.03,
                }
            )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_encoder_retrain_gap_figure(rows, output_dir)

            self.assertTrue(
                (output_dir / "encoder_retrain_gap_comparison.png").is_file()
            )

    def test_target_client_distribution_writes_csv_ready_rows_and_figures(self):
        manifest = {
            "partition": "equal_size_dirichlet",
            "class_labels": [0, 1, 2],
            "clients": [
                {"client_id": 0, "train_label_count": [5, 4, 1]},
                {"client_id": 1, "train_label_count": [1, 2, 7]},
            ],
        }
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_client_partition_figure(
                manifest, output_dir, target_client_id=1
            )
            rows = save_target_client_class_distribution_figure(
                manifest, 1, output_dir
            )

            self.assertTrue(
                (output_dir / "client_class_distribution.png").is_file()
            )
            self.assertTrue(
                (output_dir / "target_client_class_distribution.png").is_file()
            )
            self.assertEqual([row["train_examples"] for row in rows], [1, 2, 7])
            self.assertEqual(rows[2]["class_name"], "bird")
            self.assertEqual(rows[2]["rank_by_count"], 1)
            self.assertAlmostEqual(rows[2]["train_fraction"], 0.7)

    def test_save_oracle_figures_writes_all_three_png_files(self):
        predictions, summaries = make_minimal_oracle_prediction_fixture()
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_oracle_figures(
                predictions, summaries, output_dir, probes_per_group=1
            )

            self.assertTrue((output_dir / "reconstruction_grid.png").is_file())
            self.assertTrue((output_dir / "latent_comparison.png").is_file())
            self.assertTrue(
                (output_dir / "client_gaussian_diagnostic.png").is_file()
            )

    def test_save_federated_training_curve_figures_writes_two_png_files(self):
        history = []
        for round_id in range(3):
            history.append(
                {
                    "phase": "original",
                    "round": round_id,
                    "objective_loss": None if round_id == 0 else 1.0 / round_id,
                    "reconstruction_loss": (
                        None if round_id == 0 else 0.03 / round_id
                    ),
                    "kl_loss": None if round_id == 0 else 0.2 / round_id,
                    "reconstruction_weight": 30.0,
                    "beta": 0.0001,
                    "test_psnr_db": 12.0 + round_id,
                    "test_rmse_255": 64.0 - round_id,
                }
            )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_federated_training_curve_figures(history, output_dir, "iid")

            self.assertTrue((output_dir / "iid_training_loss.png").is_file())
            self.assertTrue(
                (output_dir / "iid_training_performance.png").is_file()
            )

    def test_reconstruction_curve_filenames_include_noniid_split(self):
        history = []
        for round_id in range(2):
            history.append(
                {
                    "phase": "original",
                    "round": round_id,
                    "objective_loss": None if round_id == 0 else 1.0,
                    "reconstruction_loss": None if round_id == 0 else 0.8,
                    "kl_loss": None if round_id == 0 else 2.0,
                    "reconstruction_weight": 1.0,
                    "beta": 0.0001,
                    "test_psnr_db": 12.0 + round_id,
                    "test_rmse_255": 64.0 - round_id,
                }
            )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_federated_training_curve_figures(
                history, output_dir, "equal_size_dirichlet"
            )

            self.assertTrue(
                (output_dir / "equal_size_dirichlet_training_loss.png").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "equal_size_dirichlet_training_performance.png"
                ).is_file()
            )

    def test_save_federated_classification_curves_writes_ce_accuracy_png(self):
        history = []
        for round_id in range(3):
            history.append(
                {
                    "phase": "original",
                    "round": round_id,
                    "train_cross_entropy": (
                        None if round_id == 0 else 2.0 / round_id
                    ),
                    "train_accuracy": (
                        None if round_id == 0 else 0.1 * round_id
                    ),
                    "test_cross_entropy": 2.1 - 0.1 * round_id,
                    "test_accuracy": 0.1 + 0.05 * round_id,
                }
            )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_federated_classification_curve_figures(
                history, output_dir, "iid"
            )

            self.assertTrue(
                (output_dir / "iid_classification_training.png").is_file()
            )

    def test_classification_curve_filename_includes_noniid_split(self):
        history = [
            {
                "phase": "original",
                "round": 0,
                "train_cross_entropy": None,
                "train_accuracy": None,
                "test_cross_entropy": 2.1,
                "test_accuracy": 0.1,
            },
            {
                "phase": "original",
                "round": 1,
                "train_cross_entropy": 2.0,
                "train_accuracy": 0.2,
                "test_cross_entropy": 1.9,
                "test_accuracy": 0.3,
            },
        ]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_federated_classification_curve_figures(
                history, output_dir, "fedosd_balanced_pathological"
            )

            self.assertTrue(
                (
                    output_dir
                    / "fedosd_balanced_pathological_classification_training.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
