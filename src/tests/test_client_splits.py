import unittest

import numpy as np
import torch

from src.data.data import (
    build_class_matched_target_retain_probe,
    build_controlled_noniid_split,
    build_fedosd_balanced_pathological_split,
    build_target_exclusive_label_split,
)


class _TinyCifar:
    def __init__(self, labels):
        self.targets = list(labels)
        self.data = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)


class ControlledClientSplitTests(unittest.TestCase):
    def test_controlled_split_is_complete_balanced_and_removes_digit_one(self):
        labels = torch.tensor([digit for digit in range(10) for _ in range(24)])

        split = build_controlled_noniid_split(
            labels,
            split_seed=7,
            shared_labels=(3, 4, 5, 6),
            shared_per_label=6,
            target_specific_label=1,
            retained_clients=4,
        )

        self.assertEqual(
            set(split.target_groups["target_specific"]), set(range(24, 48))
        )
        self.assertEqual(
            len(set().union(*split.client_indices.values())), len(labels)
        )
        self.assertEqual(
            sum(len(indices) for indices in split.client_indices.values()), len(labels)
        )

        retained_indices = [
            index
            for client_id in range(1, 5)
            for index in split.client_indices[f"client_{client_id}"]
        ]
        self.assertFalse(torch.any(labels[retained_indices] == 1))

        retained_sizes = [
            len(split.client_indices[f"client_{client_id}"])
            for client_id in range(1, 5)
        ]
        self.assertLessEqual(max(retained_sizes) - min(retained_sizes), 1)


def test_fedosd_balanced_pathological_split_has_two_labels_per_client():
    split = build_fedosd_balanced_pathological_split(
        np.repeat(np.arange(10), 20),
        np.repeat(np.arange(10), 10),
        n_clients=10,
        classes_per_client=2,
        seed=7,
    )

    assert split.train_allocation.sum(axis=1).tolist() == [20] * 10
    assert split.train_allocation.sum(axis=0).tolist() == [20] * 10
    assert (split.train_allocation > 0).sum(axis=1).tolist() == [2] * 10
    assert (split.train_allocation > 0).sum(axis=0).tolist() == [2] * 10
    assert np.flatnonzero(split.train_allocation[0]).tolist() == [0, 1]
    assert np.flatnonzero(split.train_allocation[1]).tolist() == [0, 1]


def test_target_exclusive_label_split_assigns_target_label_only_to_target_client():
    split = build_target_exclusive_label_split(
        np.repeat(np.arange(10), 18),
        np.repeat(np.arange(10), 10),
        n_clients=10,
        target_client_id=0,
        target_label=0,
        seed=7,
    )

    assert split.train_allocation[0].tolist() == [18] + [0] * 9
    assert split.test_allocation[0].tolist() == [10] + [0] * 9
    assert split.train_allocation[:, 0].tolist() == [18] + [0] * 9
    assert split.train_allocation.sum(axis=1).tolist() == [18] * 10


def test_client_leakage_probe_matches_classes_and_memberships():
    labels = np.repeat(np.arange(3), 20)
    dataset = _TinyCifar(labels)
    target_indices = np.concatenate(
        [np.arange(label * 20, label * 20 + 8) for label in range(3)]
    )
    retain_indices = np.asarray(
        [index for index in range(len(labels)) if index not in set(target_indices)]
    )
    probe = build_class_matched_target_retain_probe(
        dataset,
        target_indices,
        retain_indices,
        max_examples_per_class=6,
        evaluation_fraction=0.5,
        seed=4,
    )
    assert len(probe["images"]) == 3 * 2 * 6
    for class_id in range(3):
        for partition in ("training", "evaluation"):
            positions = np.flatnonzero(
                (probe["semantic_labels"].numpy() == class_id)
                & (probe["partitions"] == partition)
            )
            values, counts = np.unique(
                probe["membership"].numpy()[positions], return_counts=True
            )
            assert values.tolist() == [0, 1]
            assert counts[0] == counts[1]


if __name__ == "__main__":
    unittest.main()
