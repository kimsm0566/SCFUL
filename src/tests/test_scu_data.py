import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from src.data.data import create_scu_dataset, get_scu_dataloader, split_scu_data


def test_scu_data_helpers_normalize_and_split_client_data_deterministically():
    images = torch.full((10, 3, 32, 32), 255, dtype=torch.uint8)
    labels = torch.arange(10)

    dataset = create_scu_dataset((images, labels))
    target_data, retained_data, target_indices, retained_indices = split_scu_data(
        dataset, forget_ratio=0.3, seed=7
    )
    target_data_again, _, target_indices_again, _ = split_scu_data(
        dataset, forget_ratio=0.3, seed=7
    )

    batch_images, batch_labels = next(iter(get_scu_dataloader(dataset, 10, seed=1)))

    assert batch_images.shape == (10, 3, 32, 32)
    assert torch.all(batch_images == 1.0)
    assert torch.equal(batch_labels.sort().values, labels)
    assert len(target_data) == 3
    assert len(retained_data) == 7
    assert set(target_indices).isdisjoint(retained_indices)
    assert set(target_indices) | set(retained_indices) == set(range(10))
    assert target_indices == target_indices_again
    assert len(target_data_again) == 3
