import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils import trainer as trainer_module
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig


def test_trainer_import_does_not_require_unused_fl_packages():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        ["python", "-c", "import utils.trainer"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_trainer_runs_scu_for_one_client_codec():
    images = torch.cat(
        [torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)]
    )
    labels = torch.tensor([0, 0, 1, 1])
    node = SimpleNamespace(
        model=VariationalDeepJSCC(
            VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
        ),
        data=(images, labels),
    )
    args = SimpleNamespace(
        algorithm="SCU",
        dataset="cifar10",
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=2,
        lr=1e-4,
        snr_db=10.0,
        seed=0,
        scu_forget_ratio=0.5,
        scu_mode="self_sharing",
        lambda_kl=1.0,
        lambda_rec=0.1,
        lambda_ss=1.0,
        beta=1e-4,
        epsilon=1e-8,
    )

    result = trainer_module.Trainer(args).train(
        data_state={"node": node},
        comm=None,
        logger=None,
        args=args,
    )

    assert result["algorithm"] == "SCU"
    assert result["original_steps"] == 2
    assert result["steps"] == 1
    assert result["target_examples"] == 2
    assert result["retained_examples"] == 2
    assert len(result["target_indices"]) == 2
    assert len(result["retained_indices"]) == 2
    assert len(result["original_loss_trace"]) == 2
    assert len(result["scu_loss_trace"]) == 1
    assert any(
        not torch.equal(original, unlearned)
        for original, unlearned in zip(
            result["original_model"].parameters(), node.model.parameters()
        )
    )
