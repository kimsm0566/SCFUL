import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.oracle_behavior import _build_codec, run_oracle_behavior
from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_ARCHITECTURE,
    VariationalDeepJSCC,
)


def cifar10_comparison_config():
    return {
        "experiment_name": "controlled_cifar10_behavior",
        "device": "auto",
        "deterministic": True,
        "dataset": {
            "name": "CIFAR10",
            "data_root": "data/cifar10",
            "download": False,
            "transform": "ToTensor",
        },
        "codec": {
            "architecture": LEGACY_DEEPJSCC_ARCHITECTURE,
            "latent_dim": 64,
            "latent_channels": 1,
            "logvar_min": -8.0,
            "logvar_max": 4.0,
            "epsilon_power": 1e-8,
            "kl_reduction": "elementwise_mean",
        },
        "split": {
            "shared_labels": [3, 4, 5, 6],
            "shared_per_label": 8,
            "target_specific_label": 1,
            "retained_clients": 4,
        },
        "audit": {"target_per_group": 4, "retained_total": 18},
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "beta": 1e-4,
            "snr_db": 10.0,
            "num_workers": 0,
        },
        "scu": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "lambda_kl": 1.0,
            "lambda_rec": 0.1,
            "lambda_ss": 1.0,
            "beta": 1e-4,
            "epsilon": 1e-8,
            "snr_db": 10.0,
            "num_workers": 0,
            "optimizer_state": "new_adam_from_original_checkpoint",
            "retained_sampling": "shuffled_without_replacement_then_recycle",
        },
        "evaluation": {
            "batch_size": 16,
            "awgn_snr_db": 10.0,
            "awgn_repetitions": 2,
            "bootstrap_resamples": 20,
            "figure_probes_per_group": 2,
        },
        "client_posterior": {
            "fit_fraction": 0.8,
            "batch_size": 16,
            "minimum_variance": 1e-6,
            "mmd_sample_limit": 32,
            "bootstrap_resamples": 8,
            "bootstrap_sample_size": 16,
        },
    }


def test_controlled_cifar10_builder_uses_variational_deepjscc_by_default():
    assert isinstance(_build_codec("CIFAR10", {}), VariationalDeepJSCC)


def test_cifar10_comparison_smoke_writes_expected_artifact_schema():
    with TemporaryDirectory() as directory:
        run_dir = run_oracle_behavior(
            cifar10_comparison_config(),
            seed=3,
            output_root=Path(directory),
            smoke_test=True,
        )

        required = [
            "config.json",
            "environment.json",
            "split_manifest.json",
            "metrics/per_sample.csv",
            "metrics/summary.json",
            "metrics/training_by_epoch.csv",
            "predictions/probes.pt",
            "checkpoints/initial.pt",
            "checkpoints/original.pt",
            "checkpoints/retrain.pt",
            "checkpoints/scu_forget_only.pt",
            "checkpoints/scu_self_sharing.pt",
            "figures/reconstruction_grid.png",
        ]
        for relative_path in required:
            assert (run_dir / relative_path).is_file(), relative_path

        manifest = json.loads((run_dir / "split_manifest.json").read_text())
        assert manifest["dataset"]["name"] == "CIFAR10"
        assert manifest["dataset"]["smoke_subset"]["train_size"] == 200
        assert manifest["dataset"]["smoke_subset"]["test_size"] == 120
        assert manifest["assertions"]["retained_target_specific_count"] == 0

        with (run_dir / "metrics/per_sample.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert {row["dataset_split"] for row in rows} <= {
            "cifar10_train",
            "cifar10_test",
        }
        assert {row["model"] for row in rows} >= {
            "original",
            "retrain",
            "scu_forget_only",
            "scu_self_sharing",
        }

        with (run_dir / "metrics/training_by_epoch.csv").open(encoding="utf-8") as handle:
            training_rows = list(csv.DictReader(handle))
        assert {row["model"] for row in training_rows} >= {
            "original",
            "retrain",
            "scu_forget_only",
            "scu_self_sharing",
        }
        assert {row["epoch"] for row in training_rows} == {"0"}


def test_cifar10_v3_comparison_smoke_completes_with_standard_vib():
    config = cifar10_comparison_config()
    config["codec"] = {
        "architecture": DEEPJSCC_ARCHITECTURE,
        "latent_dim": 64,
        "latent_channels": 1,
    }

    with TemporaryDirectory() as directory:
        run_dir = run_oracle_behavior(
            config,
            seed=5,
            output_root=Path(directory),
            smoke_test=True,
        )

        saved_config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        assert saved_config["codec"]["architecture"] == DEEPJSCC_ARCHITECTURE
        assert (run_dir / "run.log").read_text(
            encoding="utf-8"
        ).splitlines()[-1] == "completed"
        assert (run_dir / "checkpoints/original.pt").is_file()
