import numpy as np

from src.data.data import (
    build_equal_size_federated_split,
    build_iid_federated_split,
    partition_equal_size_dirichlet,
    reserve_class_balanced_public_indices,
    select_class_balanced_pool_subset,
)


def test_iid_split_is_class_balanced_and_uses_every_example_once():
    train_labels = np.repeat(np.arange(10), 50)
    test_labels = np.repeat(np.arange(10), 10)

    split = build_iid_federated_split(
        train_labels, test_labels, n_clients=10, seed=7
    )

    assert split.train_allocation.tolist() == [[5] * 10] * 10
    assert split.test_allocation.tolist() == [[1] * 10] * 10
    assigned_train = np.concatenate(split.train_indices)
    assigned_test = np.concatenate(split.test_indices)
    assert len(np.unique(assigned_train)) == len(train_labels)
    assert len(np.unique(assigned_test)) == len(test_labels)


def test_equal_size_dirichlet_uses_every_example_once():
    labels = np.repeat(np.arange(10), 50)

    client_indices, allocation = partition_equal_size_dirichlet(
        labels, n_clients=10, alpha=0.1, seed=7
    )

    assigned_indices = np.concatenate(client_indices)
    assert [len(indices) for indices in client_indices] == [50] * 10
    assert len(assigned_indices) == len(labels)
    assert len(np.unique(assigned_indices)) == len(labels)
    assert allocation.sum(axis=1).tolist() == [50] * 10
    assert allocation.sum(axis=0).tolist() == [50] * 10


def test_equal_size_dirichlet_is_seed_deterministic_and_seed_sensitive():
    labels = np.repeat(np.arange(10), 50)

    _, first = partition_equal_size_dirichlet(labels, n_clients=10, alpha=0.1, seed=7)
    _, same_seed = partition_equal_size_dirichlet(
        labels, n_clients=10, alpha=0.1, seed=7
    )
    _, different_seed = partition_equal_size_dirichlet(
        labels, n_clients=10, alpha=0.1, seed=8
    )

    assert np.array_equal(first, same_seed)
    assert not np.array_equal(first, different_seed)


def test_federated_split_keeps_equal_train_and_test_client_sizes():
    train_labels = np.repeat(np.arange(10), 50)
    test_labels = np.repeat(np.arange(10), 10)

    split = build_equal_size_federated_split(
        train_labels,
        test_labels,
        n_clients=10,
        alpha=0.1,
        seed=7,
    )

    assert [len(indices) for indices in split.train_indices] == [50] * 10
    assert [len(indices) for indices in split.test_indices] == [10] * 10
    assert split.train_allocation.sum(axis=0).tolist() == [50] * 10
    assert split.test_allocation.sum(axis=0).tolist() == [10] * 10


def test_public_calibration_split_is_class_balanced_and_disjoint_from_clients():
    labels = np.repeat(np.arange(10), 50)

    public_indices, client_pool_indices = reserve_class_balanced_public_indices(
        labels, samples_per_class=4, seed=7
    )

    assert len(public_indices) == 40
    assert np.bincount(labels[public_indices], minlength=10).tolist() == [4] * 10
    assert len(client_pool_indices) == 460
    assert not np.intersect1d(public_indices, client_pool_indices).size


def test_client_pool_subsample_is_balanced_reproducible_and_tracks_unused_data():
    labels = np.repeat(np.arange(10), 50)
    candidates = np.arange(len(labels), dtype=np.int64)

    selected, unused = select_class_balanced_pool_subset(
        labels, candidates, total_size=300, seed=7
    )
    selected_again, unused_again = select_class_balanced_pool_subset(
        labels, candidates, total_size=300, seed=7
    )

    assert np.array_equal(selected, selected_again)
    assert np.array_equal(unused, unused_again)
    assert np.bincount(labels[selected], minlength=10).tolist() == [30] * 10
    assert len(unused) == 200
    assert not np.intersect1d(selected, unused).size
    assert len(np.unique(np.concatenate([selected, unused]))) == len(labels)
