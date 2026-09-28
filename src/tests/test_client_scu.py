import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from src.models.semantic_codec import VariationalDeepJSCC
from src.utils.client import Client


def test_scu_client_uses_variational_deepjscc():
    args = SimpleNamespace(
        algorithm="SCU",
        dataset="cifar10",
        device=torch.device("cpu"),
        lr=1e-4,
        snr_db=10.0,
    )
    data = (
        torch.zeros(2, 3, 32, 32),
        torch.tensor([0, 1]),
    )

    client = Client(data, None, args, rank=0)

    assert isinstance(client.model, VariationalDeepJSCC)
