import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.oracle_behavior import run_oracle_behavior
from src.utils.config import DEFAULT_ORACLE_CONFIG_PATH, load_json_config


class OracleBehaviorSmokeTests(unittest.TestCase):
    def test_synthetic_smoke_run_writes_required_artifacts_and_schema(self):
        with TemporaryDirectory() as directory:
            run_dir = run_oracle_behavior(
                load_json_config(DEFAULT_ORACLE_CONFIG_PATH),
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
                "predictions/probes.pt",
                "client_posterior/per_client_gaussian.csv",
                "client_posterior/summary.json",
                "figures/reconstruction_grid.png",
                "figures/latent_comparison.png",
                "figures/client_gaussian_diagnostic.png",
                "checkpoints/initial.pt",
                "checkpoints/original.pt",
                "checkpoints/retrain.pt",
                "checkpoints/retrain_replica.pt",
                "checkpoints/scu_forget_only.pt",
                "checkpoints/scu_self_sharing.pt",
            ]
            for relative_path in required:
                self.assertTrue((run_dir / relative_path).is_file(), relative_path)

            with (run_dir / "metrics/per_sample.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                set(
                    (
                        "dataset_split",
                        "source_index",
                        "probe_partition",
                        "support_group",
                        "model",
                        "evaluation_condition",
                        "channel_repetitions",
                        "mse_input",
                        "fg_output",
                        "fg_mu",
                        "fg_latent",
                    )
                ).issubset(row),
                True,
            )
            self.assertEqual(
                (run_dir / "run.log").read_text(encoding="utf-8").splitlines()[-1],
                "completed",
            )
            summary = json.loads(
                (run_dir / "metrics/summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                {"original", "retrain", "scu_self_sharing"}
                <= set(summary["training"])
            )


if __name__ == "__main__":
    unittest.main()
