import hashlib
import json
import torch
import numpy as np
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision.transforms as transforms

from src.utils.reproducibility import derive_seed

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def remaining_classifier_source(config, manifest):
    """Canonical clean-data scenario; independent of codec/observer weights."""
    if config["dataset"]["name"] != "CIFAR10":
        raise ValueError("remaining classifier supports only CIFAR10")
    target = int(config["unlearning"]["target_client_id"])
    clients = sorted(manifest["clients"], key=lambda row: int(row["client_id"]))
    if [int(row["client_id"]) for row in clients] != list(
        range(int(config["federation"]["n_clients"]))
    ):
        raise ValueError("source client IDs must be complete and unique")
    if not 0 <= target < len(clients) or len(clients) < 2:
        raise ValueError("source needs a target and at least one remaining client")
    seen = set()
    rows = []
    for row in clients:
        indices = row["train_indices"]
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 50000 for i in indices):
            raise ValueError("invalid CIFAR10 source indices")
        if not indices or len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError("source client indices must be nonempty and disjoint")
        seen.update(indices)
        rows.append({"client_id": int(row["client_id"]), "train_indices": indices})
    scenario = {
        "dataset": "torchvision-CIFAR10-official-train-50000",
        "partition": manifest["partition"],
        "split_seed": int(manifest["seed"]),
        "target_client_id": target,
        "clients": rows,
        "public_calibration": manifest.get("public_calibration", {}),
        "input": "clean_raw_images_and_original_labels_no_poison_injection",
    }
    fingerprint = hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return scenario, fingerprint


def split_remaining_classifier_indices(indices, labels, validation_fraction, seed, max_samples=None):
    """Split only one remaining client's indices; never access target images."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(np.asarray(indices, dtype=np.int64))
    if max_samples is not None:
        indices = indices[:int(max_samples)]
    if len(indices) < 2:
        raise ValueError("each remaining client needs at least two examples")
    # Stratify within each client when possible; singleton classes stay in training.
    labels = np.asarray(labels)
    train, validation = [], []
    for label in np.unique(labels[indices]):
        group = indices[labels[indices] == label]
        count = min(len(group) - 1, max(1, int(len(group) * validation_fraction)))
        validation.extend(group[:count].tolist())
        train.extend(group[count:].tolist())
    if not validation:
        validation.append(train.pop())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def prepare_remaining_classifier_data(comm, args):
    """MPI: server gets indices/counts only; each worker reads its own clean images."""
    from src.utils.config import load_json_config, load_remaining_classifier_config

    if len(args.source_run) != 1:
        raise ValueError("remaining_classifier requires exactly one --source-run")
    if getattr(args, "observer_run", None) is not None or getattr(args, "reference_run", None) is not None:
        raise ValueError("remaining classifier always starts fresh; observer/reference warm starts are forbidden")
    source_run = Path(args.source_run[0]).resolve()
    source_config = load_json_config(source_run / "config.json")
    manifest = load_json_config(source_run / "split_manifest.json")
    scenario, fingerprint = remaining_classifier_source(source_config, manifest)
    config = load_remaining_classifier_config(args.config, smoke=args.smoke)
    remaining = [row for row in scenario["clients"] if row["client_id"] != scenario["target_client_id"]]
    if comm.Get_size() != len(remaining) + 1:
        raise ValueError(f"remaining_classifier needs {len(remaining) + 1} MPI ranks (server + remaining clients)")
    rank = comm.Get_rank()
    local = {"train_data": None, "validation_data": None, "client_id": None}
    local_manifest, error = None, None
    try:
        if rank > 0:
            row = remaining[rank - 1]
            dataset = load_cifar10_train_dataset(source_config["dataset"], WORKSPACE_ROOT)
            labels = dataset_labels(dataset)
            train_indices, validation_indices = split_remaining_classifier_indices(
                row["train_indices"], labels, config["validation_fraction"],
                derive_seed(args.seed, f"observer-split-client-{row['client_id']}"),
                max_samples=20 if args.smoke else None,
            )
            local = {
                "client_id": row["client_id"],
                "train_data": select_cifar10_client_data(dataset, train_indices),
                "validation_data": select_cifar10_client_data(dataset, validation_indices),
            }
            local_manifest = {
                "client_id": row["client_id"],
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "train_class_counts": np.bincount(labels[train_indices], minlength=10).tolist(),
                "validation_class_counts": np.bincount(labels[validation_indices], minlength=10).tolist(),
            }
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [value for value in comm.allgather(error) if value]
    if errors:
        raise RuntimeError("; ".join(errors))
    split_rows = comm.gather(local_manifest, root=0)
    return {
        **local, "config": config, "source_config": source_config,
        "source_run": str(source_run), "source_scenario": scenario,
        "source_fingerprint": fingerprint,
        "client_splits": [row for row in split_rows if row] if rank == 0 else None,
    }


def _index_digest(indices):
    payload = ",".join(str(index) for index in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shuffled_indices(indices, generator):
    if len(indices) == 0:
        return []
    order = torch.randperm(len(indices), generator=generator)
    return indices[order].tolist()


class ControlledClientSplit:
    """Complete, mutually exclusive controlled client partition."""

    def __init__(self, client_indices, target_groups, shared_labels,
                 target_specific_label, split_seed, labels):
        self.client_indices = client_indices
        self.target_groups = target_groups
        self.shared_labels = shared_labels
        self.target_specific_label = target_specific_label
        self.split_seed = split_seed
        self.labels = labels

    def manifest(self):
        all_indices = [
            index for indices in self.client_indices.values() for index in indices
        ]
        unique_indices = set(all_indices)
        retained_indices = [
            index
            for client_id, indices in self.client_indices.items()
            if client_id != "client_0"
            for index in indices
        ]
        retained_sizes = [
            len(indices)
            for client_id, indices in self.client_indices.items()
            if client_id != "client_0"
        ]
        client_label_counts = {
            client_id: {
                str(label): int((self.labels[indices] == label).sum().item())
                for label in range(10)
            }
            for client_id, indices in self.client_indices.items()
        }
        coverage = (
            len(all_indices) == len(self.labels)
            and len(unique_indices) == len(self.labels)
            and unique_indices == set(range(len(self.labels)))
        )
        return {
            "split_seed": self.split_seed,
            "shared_labels": list(self.shared_labels),
            "target_specific_label": self.target_specific_label,
            "client_indices": self.client_indices,
            "client_index_sha256": {
                client_id: _index_digest(indices)
                for client_id, indices in self.client_indices.items()
            },
            "target_groups": self.target_groups,
            "target_group_sha256": {
                group: _index_digest(indices)
                for group, indices in self.target_groups.items()
            },
            "client_total_counts": {
                client_id: len(indices)
                for client_id, indices in self.client_indices.items()
            },
            "client_label_counts": client_label_counts,
            "assertions": {
                "complete_coverage": coverage,
                "mutually_exclusive": len(all_indices) == len(unique_indices),
                "retained_target_specific_count": int(
                    (self.labels[retained_indices] == self.target_specific_label)
                    .sum().item()
                ),
                "retained_size_difference": max(retained_sizes) - min(retained_sizes),
            },
        }


def build_controlled_noniid_split(
    labels, split_seed, shared_labels=(3, 4, 5, 6), shared_per_label=1_500,
    target_specific_label=1, retained_clients=4,
):
    """Build a controlled split with target-only and shared semantics."""
    labels = labels.detach().cpu().flatten().to(torch.long)
    if retained_clients < 1:
        raise ValueError("retained_clients must be positive")
    if target_specific_label in shared_labels:
        raise ValueError("target_specific_label must not be a shared label")
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    selected_shared = []
    for label in shared_labels:
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        if len(candidates) < shared_per_label:
            raise ValueError(
                f"label {label} has {len(candidates)} samples, expected at least {shared_per_label}"
            )
        selected_shared.extend(_shuffled_indices(candidates, generator)[:shared_per_label])
    target_specific = torch.nonzero(
        labels == target_specific_label, as_tuple=False
    ).flatten().tolist()
    target_set = set(selected_shared) | set(target_specific)
    if len(target_set) != len(selected_shared) + len(target_specific):
        raise AssertionError("target groups overlap")
    retained_by_client = {
        f"client_{client_id}": [] for client_id in range(1, retained_clients + 1)
    }
    cursor = 0
    for label in range(int(labels.max().item()) + 1):
        if label == target_specific_label:
            continue
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        retained_for_digit = torch.tensor(
            [index for index in candidates.tolist() if index not in target_set],
            dtype=torch.long,
        )
        for offset, index in enumerate(_shuffled_indices(retained_for_digit, generator)):
            client_id = 1 + ((cursor + offset) % retained_clients)
            retained_by_client[f"client_{client_id}"].append(index)
        cursor = (cursor + len(retained_for_digit)) % retained_clients
    client_indices = {
        "client_0": sorted(target_set),
        **{client_id: sorted(indices) for client_id, indices in retained_by_client.items()},
    }
    split = ControlledClientSplit(
        client_indices=client_indices,
        target_groups={
            "shared_high": sorted(selected_shared),
            "target_specific": sorted(target_specific),
        },
        shared_labels=shared_labels,
        target_specific_label=target_specific_label,
        split_seed=split_seed,
        labels=labels,
    )
    assertions = split.manifest()["assertions"]
    if not assertions["complete_coverage"] or not assertions["mutually_exclusive"]:
        raise AssertionError("controlled split must cover every sample exactly once")
    if assertions["retained_target_specific_count"] != 0:
        raise AssertionError("retained data contains target-specific samples")
    if assertions["retained_size_difference"] > 1:
        raise AssertionError("retained client sizes are unbalanced")
    return split


class ProbeRef:
    def __init__(self, dataset_split, source_index, probe_partition, support_group):
        self.dataset_split = dataset_split
        self.source_index = source_index
        self.probe_partition = probe_partition
        self.support_group = support_group


class ProbeDataset(Dataset):
    """Dataset wrapper that retains audit metadata for every sample."""

    def __init__(self, source, refs):
        self.source = source
        self.refs = refs

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, position):
        reference = self.refs[position]
        image, label = self.source[reference.source_index][:2]
        return image, torch.as_tensor(label), torch.tensor(position)


def _take_random(indices, count, generator):
    if len(indices) < count:
        raise ValueError(f"expected at least {count} candidate probes, found {len(indices)}")
    order = torch.randperm(len(indices), generator=generator)[:count].tolist()
    return [indices[position] for position in order]


def select_audit_indices(
    split, train_labels, test_labels, probe_seed, target_per_group=256,
    retained_total=2_000, dataset_name="cifar10",
):
    """Select deterministic member, target-heldout, and retained probes."""
    train_labels = train_labels.detach().cpu().flatten().to(torch.long)
    test_labels = test_labels.detach().cpu().flatten().to(torch.long)
    if target_per_group % len(split.shared_labels) != 0:
        raise ValueError("target_per_group must be divisible by the number of shared labels")
    generator = torch.Generator(device="cpu").manual_seed(probe_seed)
    per_shared_label = target_per_group // len(split.shared_labels)
    member_shared = []
    heldout_shared = []
    for label in split.shared_labels:
        member_candidates = [
            index for index in split.target_groups["shared_high"]
            if int(train_labels[index]) == label
        ]
        heldout_candidates = torch.nonzero(
            test_labels == label, as_tuple=False
        ).flatten().tolist()
        member_shared.extend(_take_random(member_candidates, per_shared_label, generator))
        heldout_shared.extend(_take_random(heldout_candidates, per_shared_label, generator))
    member_specific = _take_random(
        split.target_groups["target_specific"], target_per_group, generator
    )
    heldout_specific = _take_random(
        torch.nonzero(test_labels == split.target_specific_label, as_tuple=False)
        .flatten().tolist(), target_per_group, generator,
    )
    retained_labels = [
        label for label in range(int(test_labels.max().item()) + 1)
        if label != split.target_specific_label
    ]
    base_count, remainder = divmod(retained_total, len(retained_labels))
    retained_heldout = []
    for position, label in enumerate(retained_labels):
        count = base_count + int(position < remainder)
        candidates = torch.nonzero(test_labels == label, as_tuple=False).flatten().tolist()
        retained_heldout.extend(_take_random(candidates, count, generator))

    def refs(dataset_split, indices, partition, group):
        return [ProbeRef(dataset_split, index, partition, group) for index in indices]

    train_split = f"{dataset_name}_train"
    test_split = f"{dataset_name}_test"
    return {
        "target_train_member": refs(
            train_split, member_shared, "target_train_member", "shared_high"
        ) + refs(train_split, member_specific, "target_train_member", "target_specific"),
        "target_heldout": refs(
            test_split, heldout_shared, "target_heldout", "shared_high"
        ) + refs(test_split, heldout_specific, "target_heldout", "target_specific"),
        "retained_heldout": refs(
            test_split, retained_heldout, "retained_heldout", "retained"
        ),
    }


def _fit_dirichlet_quotas(weights, client_totals, class_totals):
    """양수 weight를 지정한 client/class margin을 갖는 연속 quota로 맞춘다."""
    quotas = np.maximum(weights.astype(np.float64), np.finfo(np.float64).tiny)
    client_totals = np.asarray(client_totals, dtype=np.float64)
    class_totals = np.asarray(class_totals, dtype=np.float64)

    for _ in range(10_000):
        quotas *= (client_totals / quotas.sum(axis=1))[:, None]
        quotas *= (class_totals / quotas.sum(axis=0))[None, :]
        if np.allclose(quotas.sum(axis=1), client_totals, atol=1e-8) and np.allclose(
            quotas.sum(axis=0), class_totals, atol=1e-8
        ):
            return quotas

    raise RuntimeError("failed to fit Dirichlet quotas to client and class totals")


def _add_flow_edge(graph, source, target, capacity, cost):
    forward = [target, len(graph[target]), capacity, cost]
    backward = [source, len(graph[source]), 0, -cost]
    graph[source].append(forward)
    graph[target].append(backward)


def _round_quotas(quotas, client_totals, class_totals):
    """연속 quota를 행·열 합이 보존되는 정수 allocation으로 반올림한다."""
    allocation = np.floor(quotas).astype(np.int64)
    row_remaining = np.asarray(client_totals, dtype=np.int64) - allocation.sum(axis=1)
    column_remaining = np.asarray(class_totals, dtype=np.int64) - allocation.sum(axis=0)

    if np.any(row_remaining < 0) or np.any(column_remaining < 0):
        raise RuntimeError("quota rounding produced a negative remaining margin")
    if row_remaining.sum() != column_remaining.sum():
        raise RuntimeError("quota rounding has inconsistent remaining margins")

    required_flow = int(row_remaining.sum())
    if required_flow == 0:
        return allocation

    n_clients, n_classes = allocation.shape
    source = n_clients + n_classes
    sink = source + 1
    graph = [[] for _ in range(sink + 1)]

    for client_id, count in enumerate(row_remaining):
        _add_flow_edge(graph, source, client_id, int(count), 0.0)
    for class_id, count in enumerate(column_remaining):
        _add_flow_edge(graph, n_clients + class_id, sink, int(count), 0.0)

    fractions = quotas - np.floor(quotas)
    for client_id in range(n_clients):
        for class_id in range(n_classes):
            if fractions[client_id, class_id] > 1e-12:
                _add_flow_edge(
                    graph,
                    client_id,
                    n_clients + class_id,
                    1,
                    -float(fractions[client_id, class_id]),
                )

    flow = 0
    while flow < required_flow:
        distances = [float("inf")] * len(graph)
        previous = [None] * len(graph)
        distances[source] = 0.0

        for _ in range(len(graph) - 1):
            updated = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_id, edge in enumerate(edges):
                    target, _, capacity, cost = edge
                    if capacity <= 0:
                        continue
                    candidate = distances[node] + cost
                    if candidate < distances[target] - 1e-15:
                        distances[target] = candidate
                        previous[target] = (node, edge_id)
                        updated = True
            if not updated:
                break

        if previous[sink] is None:
            raise RuntimeError("could not round Dirichlet quotas while preserving margins")

        node = sink
        while node != source:
            parent, edge_id = previous[node]
            edge = graph[parent][edge_id]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = parent
        flow += 1

    for client_id in range(n_clients):
        for edge in graph[client_id]:
            class_node, reverse_index, capacity, _ = edge
            if not n_clients <= class_node < n_clients + n_classes:
                continue
            reverse_edge = graph[class_node][reverse_index]
            if reverse_edge[2] == 1:
                allocation[client_id, class_node - n_clients] += 1

    return allocation


def partition_equal_size_dirichlet(labels, n_clients, alpha, seed):
    """Client 수는 같게, class 비율만 Dirichlet으로 치우치게 분할한다.

    모든 sample은 한 번씩 사용한다. 반환하는 allocation의 행은 client, 열은
    오름차순 class label이며, 각 행의 합은 동일한 client sample 수다.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if len(labels) % n_clients != 0:
        raise ValueError("the number of samples must divide evenly across clients")

    class_labels, class_totals = np.unique(labels, return_counts=True)
    client_total = len(labels) // n_clients
    rng = np.random.default_rng(seed)
    weights = rng.gamma(alpha, 1.0, size=(n_clients, len(class_labels)))
    quotas = _fit_dirichlet_quotas(
        weights,
        client_totals=np.full(n_clients, client_total, dtype=np.int64),
        class_totals=class_totals,
    )
    allocation = _round_quotas(
        quotas,
        client_totals=np.full(n_clients, client_total, dtype=np.int64),
        class_totals=class_totals,
    )

    client_indices = [[] for _ in range(n_clients)]
    for class_column, class_label in enumerate(class_labels):
        class_indices = np.flatnonzero(labels == class_label)
        rng.shuffle(class_indices)
        offset = 0
        for client_id in range(n_clients):
            count = int(allocation[client_id, class_column])
            client_indices[client_id].extend(class_indices[offset : offset + count])
            offset += count
        if offset != len(class_indices):
            raise RuntimeError("class allocation does not cover every sample")

    for indices in client_indices:
        rng.shuffle(indices)
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices], allocation


def _indices_from_allocation(labels, class_labels, allocation, seed):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(len(allocation))]
    for class_column, class_label in enumerate(class_labels):
        class_indices = np.flatnonzero(labels == class_label)
        rng.shuffle(class_indices)
        offset = 0
        for client_id in range(len(allocation)):
            count = int(allocation[client_id, class_column])
            client_indices[client_id].extend(class_indices[offset : offset + count])
            offset += count
        if offset != len(class_indices):
            raise RuntimeError("allocation does not cover every sample in a class")
    for indices in client_indices:
        rng.shuffle(indices)
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def build_iid_federated_split(train_labels, test_labels, n_clients, seed):
    """각 class를 모든 client에 동일하게 나누는 class-balanced IID split을 만든다."""
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_class_labels, train_class_totals = np.unique(
        train_labels, return_counts=True
    )
    test_class_labels, test_class_totals = np.unique(test_labels, return_counts=True)
    if not np.array_equal(train_class_labels, test_class_labels):
        raise ValueError("train and test labels must contain the same classes")
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if np.any(train_class_totals % n_clients != 0):
        raise ValueError("every train class total must divide evenly across clients")
    if np.any(test_class_totals % n_clients != 0):
        raise ValueError("every test class total must divide evenly across clients")

    train_allocation = np.tile(
        train_class_totals // n_clients, (n_clients, 1)
    )
    test_allocation = np.tile(
        test_class_totals // n_clients, (n_clients, 1)
    )
    train_indices = _indices_from_allocation(
        train_labels, train_class_labels, train_allocation, seed
    )
    test_indices = _indices_from_allocation(
        test_labels, test_class_labels, test_allocation, seed + 1
    )
    return EqualSizeFederatedSplit(
        class_labels=train_class_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_allocation=train_allocation,
        test_allocation=test_allocation,
    )


class EqualSizeFederatedSplit:
    """Train/test 모두 client 수를 맞춘 class-skew data split."""

    def __init__(
        self,
        class_labels,
        train_indices,
        test_indices,
        train_allocation,
        test_allocation,
    ):
        self.class_labels = class_labels
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_allocation = train_allocation
        self.test_allocation = test_allocation


def build_equal_size_federated_split(
    train_labels, test_labels, n_clients, alpha, seed
):
    """Train label skew를 유지하면서 public test도 client별로 균등 분할한다."""
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_indices, train_allocation = partition_equal_size_dirichlet(
        train_labels, n_clients, alpha, seed
    )
    class_labels, test_class_totals = np.unique(test_labels, return_counts=True)
    if not np.array_equal(class_labels, np.unique(train_labels)):
        raise ValueError("train and test labels must contain the same classes")
    if len(test_labels) % n_clients != 0:
        raise ValueError("the number of test samples must divide evenly across clients")

    test_quotas = _fit_dirichlet_quotas(
        train_allocation.astype(np.float64) + 1e-8,
        client_totals=np.full(n_clients, len(test_labels) // n_clients, dtype=np.int64),
        class_totals=test_class_totals,
    )
    test_allocation = _round_quotas(
        test_quotas,
        client_totals=np.full(n_clients, len(test_labels) // n_clients, dtype=np.int64),
        class_totals=test_class_totals,
    )
    test_indices = _indices_from_allocation(
        test_labels, class_labels, test_allocation, seed + 1
    )
    return EqualSizeFederatedSplit(
        class_labels=class_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_allocation=train_allocation,
        test_allocation=test_allocation,
    )


def _fedosd_balanced_pathological_allocation(
    class_totals, n_clients, classes_per_client
):
    """FedOSD의 balanced pathological label-slot 할당을 만든다."""
    class_totals = np.asarray(class_totals, dtype=np.int64)
    n_classes = len(class_totals)
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if classes_per_client <= 0:
        raise ValueError("classes_per_client must be positive")
    total_slots = n_clients * classes_per_client
    if total_slots % n_classes != 0:
        raise ValueError(
            "n_clients * classes_per_client must divide evenly by the number of classes"
        )

    clients_per_class = total_slots // n_classes
    if np.any(class_totals % clients_per_class != 0):
        raise ValueError(
            "every class total must divide evenly across its FedOSD client holders"
        )

    remaining_slots = np.full(n_clients, classes_per_client, dtype=np.int64)
    allocation = np.zeros((n_clients, n_classes), dtype=np.int64)
    for class_column, class_total in enumerate(class_totals):
        selected_clients = [
            client_id
            for client_id in range(n_clients)
            if remaining_slots[client_id] > 0
        ][:clients_per_class]
        if len(selected_clients) != clients_per_class:
            raise RuntimeError("FedOSD pathological label slots cannot be filled")

        per_client_count = int(class_total // clients_per_class)
        allocation[selected_clients, class_column] = per_client_count
        remaining_slots[selected_clients] -= 1

    if np.any(remaining_slots != 0):
        raise RuntimeError("FedOSD pathological allocation left unused client label slots")
    return allocation


def build_fedosd_balanced_pathological_split(
    train_labels, test_labels, n_clients, classes_per_client, seed
):
    """FedOSD CIFAR-10 `pat`, `balance=True`와 같은 train/test split을 만든다.

    원본 clone은 train/test를 합친 뒤 client 내부에서 다시 나누지만, 이 helper는
    train/test를 섞지 않아 평가 sample의 학습 유입을 막는다. Label-slot incidence와
    client별 총 sample 수는 같은 방식으로 유지한다.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_class_labels, train_class_totals = np.unique(train_labels, return_counts=True)
    test_class_labels, test_class_totals = np.unique(test_labels, return_counts=True)
    if not np.array_equal(train_class_labels, test_class_labels):
        raise ValueError("train and test labels must contain the same classes")

    train_allocation = _fedosd_balanced_pathological_allocation(
        train_class_totals, n_clients, classes_per_client
    )
    test_allocation = _fedosd_balanced_pathological_allocation(
        test_class_totals, n_clients, classes_per_client
    )
    train_indices = _indices_from_allocation(
        train_labels, train_class_labels, train_allocation, seed
    )
    test_indices = _indices_from_allocation(
        test_labels, test_class_labels, test_allocation, seed + 1
    )
    return EqualSizeFederatedSplit(
        class_labels=train_class_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_allocation=train_allocation,
        test_allocation=test_allocation,
    )


def build_target_exclusive_label_split(
    train_labels,
    test_labels,
    n_clients,
    target_client_id,
    target_label,
    seed,
):
    """Target client만 한 label을 갖는 strict mask 검증용 split을 만든다.

    `target_label`의 모든 train/test sample은 `target_client_id`에 배정한다.
    나머지 label은 나머지 client에 하나씩 배정한다. 따라서 target label은
    data allocation 관점에서 명확한 target-specific semantic의 정답 역할을 한다.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_class_labels, train_class_totals = np.unique(train_labels, return_counts=True)
    test_class_labels, test_class_totals = np.unique(test_labels, return_counts=True)
    if not np.array_equal(train_class_labels, test_class_labels):
        raise ValueError("train and test labels must contain the same classes")
    if len(train_class_labels) != n_clients:
        raise ValueError(
            "target_exclusive_label requires one client per class label"
        )
    if target_label not in train_class_labels:
        raise ValueError("target_label must exist in the dataset")
    if not 0 <= target_client_id < n_clients:
        raise ValueError("target_client_id must identify an existing client")
    if len(set(train_class_totals.tolist())) != 1:
        raise ValueError("train class totals must be equal for this strict split")
    if len(set(test_class_totals.tolist())) != 1:
        raise ValueError("test class totals must be equal for this strict split")

    remaining_labels = [
        int(label) for label in train_class_labels if int(label) != target_label
    ]
    remaining_clients = [
        client_id for client_id in range(n_clients) if client_id != target_client_id
    ]
    label_to_client = {target_label: target_client_id}
    label_to_client.update(dict(zip(remaining_labels, remaining_clients)))

    train_allocation = np.zeros((n_clients, len(train_class_labels)), dtype=np.int64)
    test_allocation = np.zeros((n_clients, len(test_class_labels)), dtype=np.int64)
    for class_column, label in enumerate(train_class_labels):
        client_id = label_to_client[int(label)]
        train_allocation[client_id, class_column] = train_class_totals[class_column]
        test_allocation[client_id, class_column] = test_class_totals[class_column]

    train_indices = _indices_from_allocation(
        train_labels, train_class_labels, train_allocation, seed
    )
    test_indices = _indices_from_allocation(
        test_labels, test_class_labels, test_allocation, seed + 1
    )
    return EqualSizeFederatedSplit(
        class_labels=train_class_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_allocation=train_allocation,
        test_allocation=test_allocation,
    )


def _cifar10_data_root(dataset_config, workspace_root):
    data_root = Path(str(dataset_config["data_root"]))
    if not data_root.is_absolute():
        data_root = workspace_root / data_root
    return data_root


def _load_cifar10_split(dataset_config, workspace_root, train):
    name = str(dataset_config["name"])
    if name != "CIFAR10":
        raise ValueError(f"unsupported CIFAR-10 dataset name: {name}")
    return torchvision.datasets.CIFAR10(
        str(_cifar10_data_root(dataset_config, workspace_root)),
        train=train,
        transform=transforms.ToTensor(),
        download=bool(dataset_config.get("download", False)),
    )


def load_cifar10_train_dataset(dataset_config, workspace_root):
    return _load_cifar10_split(dataset_config, workspace_root, True)


def load_cifar10_test_dataset(dataset_config, workspace_root):
    return _load_cifar10_split(dataset_config, workspace_root, False)


def dataset_labels(dataset):
    """Return the labels stored by torchvision CIFAR-10."""
    return np.asarray(dataset.targets, dtype=np.int64)


def select_class_balanced_subset(dataset, samples_per_class, seed):
    """Select a deterministic, real CIFAR-10 subset for quick smoke checks."""
    labels = dataset_labels(dataset)
    generator = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"label {int(label)} has fewer than {samples_per_class} samples"
            )
        generator.shuffle(candidates)
        selected.extend(candidates[:samples_per_class].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return Subset(dataset, selected.tolist()), labels[selected], selected


def reserve_class_balanced_public_indices(labels, samples_per_class, seed):
    """FL 분할 전에 class별 같은 수의 공개 calibration index를 분리한다."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    generator = np.random.default_rng(seed)
    public_indices = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < samples_per_class:
            raise ValueError(
                f"label {int(label)} has fewer than {samples_per_class} public samples"
            )
        generator.shuffle(label_indices)
        public_indices.extend(label_indices[:samples_per_class])

    public_indices = np.asarray(public_indices, dtype=np.int64)
    generator.shuffle(public_indices)
    is_public = np.zeros(len(labels), dtype=bool)
    is_public[public_indices] = True
    client_pool_indices = np.flatnonzero(~is_public).astype(np.int64)
    return public_indices, client_pool_indices


def select_class_balanced_pool_subset(labels, candidate_indices, total_size, seed):
    """Select a reproducible class-balanced subset from a candidate index pool."""

    labels = np.asarray(labels, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if candidate_indices.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if len(np.unique(candidate_indices)) != len(candidate_indices):
        raise ValueError("candidate_indices must not contain duplicates")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= len(labels)):
        raise ValueError("candidate_indices contain an out-of-range index")
    if not 0 < total_size <= len(candidate_indices):
        raise ValueError("total_size must be within the candidate pool")

    candidate_labels = labels[candidate_indices]
    unique_labels = np.unique(candidate_labels)
    base_quota, remainder = divmod(int(total_size), len(unique_labels))
    generator = np.random.default_rng(seed)
    selected = []
    for position, label in enumerate(unique_labels):
        quota = base_quota + int(position < remainder)
        label_indices = candidate_indices[candidate_labels == label].copy()
        if len(label_indices) < quota:
            raise ValueError(
                f"label {int(label)} has fewer than {quota} candidate samples"
            )
        generator.shuffle(label_indices)
        selected.extend(label_indices[:quota])

    selected_indices = np.asarray(selected, dtype=np.int64)
    generator.shuffle(selected_indices)
    is_selected = np.isin(candidate_indices, selected_indices, assume_unique=True)
    unused_indices = candidate_indices[~is_selected]
    return selected_indices, unused_indices


def select_cifar10_client_data(dataset, indices):
    """한 client가 실제로 보유할 CIFAR-10 images와 labels만 추출한다."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = dataset_labels(dataset)[indices]
    if hasattr(dataset, "data"):
        images = np.asarray(dataset.data)[indices].transpose(0, 3, 1, 2)
    else:
        images = torch.stack([dataset.images[int(index)] for index in indices])
    return images, labels


def select_representation_buffer_indices(labels, max_examples, seed, mode="stratified"):
    """Select a reproducible, non-prefix subset for an erased latent buffer."""

    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("buffer labels must be a non-empty vector")
    if max_examples is None:
        max_examples = len(labels)
    max_examples = min(int(max_examples), len(labels))
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if mode not in ("sequential", "random", "stratified"):
        raise ValueError(f"unsupported erased buffer sampling mode: {mode}")

    if mode == "sequential":
        return torch.arange(max_examples, dtype=torch.long)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if mode == "random":
        return torch.randperm(len(labels), generator=generator)[:max_examples]

    buckets = []
    for label in torch.unique(labels, sorted=True):
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        order = torch.randperm(len(candidates), generator=generator)
        buckets.append(candidates[order].tolist())

    selected = []
    position = 0
    while len(selected) < max_examples:
        added = False
        for bucket in buckets:
            if position < len(bucket):
                selected.append(bucket[position])
                added = True
                if len(selected) == max_examples:
                    break
        if not added:
            break
        position += 1
    selected = torch.tensor(selected, dtype=torch.long)
    return selected[torch.randperm(len(selected), generator=generator)]


def augment_cifar10_contrastive_batch(
    images,
    generator,
    crop_padding=4,
    horizontal_flip_probability=0.5,
):
    """Create one reproducible random-crop/flip view without changing RGB scale."""

    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
        raise ValueError("CIFAR-10 contrastive augmentation expects (B, 3, 32, 32)")
    crop_padding = int(crop_padding)
    flip_probability = float(horizontal_flip_probability)
    if crop_padding < 0:
        raise ValueError("crop_padding must be non-negative")
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")

    if crop_padding > 0:
        padded = torch.nn.functional.pad(
            images,
            (crop_padding, crop_padding, crop_padding, crop_padding),
            mode="reflect",
        )
        limit = 2 * crop_padding + 1
        row_offsets = torch.randint(
            0,
            limit,
            (len(images),),
            device=images.device,
            generator=generator,
        )
        column_offsets = torch.randint(
            0,
            limit,
            (len(images),),
            device=images.device,
            generator=generator,
        )
        views = torch.stack(
            [
                padded[index, :, row : row + 32, column : column + 32]
                for index, (row, column) in enumerate(
                    zip(row_offsets.tolist(), column_offsets.tolist())
                )
            ]
        )
    else:
        views = images.clone()

    if flip_probability > 0:
        flip_mask = torch.rand(
            len(images), device=images.device, generator=generator
        ) < flip_probability
        views[flip_mask] = views[flip_mask].flip(dims=(3,))
    return views


def inject_scu_cifar10_backdoor(
    client_data,
    poison_ratio,
    base_label=1,
    trigger_label=2,
    seed=0,
):
    """Inject SCU's CIFAR-10 1 -> 2, white 2x2 trigger into local data.

    The returned tensors are copies.  The trigger is placed only in the data
    supplied by the caller, which lets the federated runner keep every poisoned
    sample at the target client.
    """

    if not 0.0 < poison_ratio <= 1.0:
        raise ValueError("SCU poison_ratio must be in (0, 1]")
    images, labels = client_data
    images = torch.as_tensor(images).clone()
    labels = torch.as_tensor(labels, dtype=torch.long).clone()
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
        raise ValueError("SCU backdoor requires CIFAR images shaped (N, 3, 32, 32)")
    if len(images) != len(labels):
        raise ValueError("SCU backdoor images and labels must have the same length")

    poison_count = int(len(images) * poison_ratio)
    if poison_count <= 0:
        raise ValueError("SCU poison_ratio selected no target-client samples")
    candidates = torch.nonzero(labels == int(base_label), as_tuple=False).flatten()
    if len(candidates) < poison_count:
        raise ValueError(
            "target client has fewer base-label samples than the requested SCU poison count"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = candidates[torch.randperm(len(candidates), generator=generator)[:poison_count]]
    patch_value = 255 if not torch.is_floating_point(images) or images.max() > 1 else 1.0
    images[selected, :, 28:30, 28:30] = patch_value
    labels[selected] = int(trigger_label)
    metadata = {
        "protocol": "SCU_CIFAR10_backdoor",
        "poison_ratio_of_target_client": float(poison_ratio),
        "poison_count": int(poison_count),
        "base_label": int(base_label),
        "trigger_label": int(trigger_label),
        "target_local_positions": selected.tolist(),
        "patch": {
            "shape": "2x2",
            "position": "zero-based rows and columns 28-29",
            "value_after_ToTensor": 1.0,
            "channels": "all RGB channels",
        },
        "seed": int(seed),
    }
    return (images, labels), metadata


def build_scu_cifar10_backdoor_probe(
    test_data,
    base_label=1,
    trigger_label=2,
    max_examples=None,
):
    """Patch clean base-label test images and relabel them for SCU ASR."""

    images, labels = test_data
    images = torch.as_tensor(images).clone()
    labels = torch.as_tensor(labels, dtype=torch.long)
    selected = torch.nonzero(labels == int(base_label), as_tuple=False).flatten()
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("backdoor probe max_examples must be positive")
        selected = selected[: int(max_examples)]
    if len(selected) == 0:
        raise ValueError("backdoor probe has no base-label examples")
    probe_images = images[selected].clone()
    patch_value = (
        255 if not torch.is_floating_point(probe_images) or probe_images.max() > 1 else 1.0
    )
    probe_images[:, :, 28:30, 28:30] = patch_value
    probe_labels = torch.full(
        (len(probe_images),), int(trigger_label), dtype=torch.long
    )
    return probe_images, probe_labels


def fscu_split_seed(seed, config):
    split_type = config["split"].get("type", "equal_size_dirichlet")
    offset = config["split"].get("seed", 0)
    return derive_seed(seed, f"{split_type}-split:{offset}")


def _copy_fscu_split(split, train_indices):
    values = dict(vars(split))
    values["train_indices"] = train_indices
    return type(split)(**values)


def _build_fscu_split(train_labels, test_labels, config, split_seed):
    split_config = config["split"]
    n_clients = int(config["federation"]["n_clients"])
    split_type = split_config.get("type", "equal_size_dirichlet")
    if split_type == "iid":
        return build_iid_federated_split(
            train_labels, test_labels, n_clients=n_clients, seed=split_seed
        )
    if split_type == "equal_size_dirichlet":
        return build_equal_size_federated_split(
            train_labels,
            test_labels,
            n_clients=n_clients,
            alpha=split_config["alpha"],
            seed=split_seed,
        )
    if split_type == "fedosd_balanced_pathological":
        return build_fedosd_balanced_pathological_split(
            train_labels,
            test_labels,
            n_clients=n_clients,
            classes_per_client=split_config["classes_per_client"],
            seed=split_seed,
        )
    return build_target_exclusive_label_split(
        train_labels,
        test_labels,
        n_clients=n_clients,
        target_client_id=config["unlearning"]["target_client_id"],
        target_label=split_config["target_label"],
        seed=split_seed,
    )


def build_fscu_split_manifest(config, split, split_seed, data_state):
    clients = []
    for client_id, indices in enumerate(split.train_indices):
        index_array = np.asarray(indices, dtype=np.int64)
        clients.append(
            {
                "client_id": client_id,
                "train_indices": index_array.tolist(),
                "test_indices": np.asarray(
                    split.test_indices[client_id], dtype=np.int64
                ).tolist(),
                "train_index_sha256": hashlib.sha256(
                    index_array.tobytes()
                ).hexdigest(),
                "train_label_count": split.train_allocation[client_id].tolist(),
                "test_label_count": split.test_allocation[client_id].tolist(),
                "train_total": int(len(indices)),
                "test_total": int(len(split.test_indices[client_id])),
            }
        )
    manifest = {
        "partition": config["split"].get("type", "equal_size_dirichlet"),
        "seed": int(split_seed),
        "class_labels": split.class_labels.tolist(),
        "train_allocation": split.train_allocation.tolist(),
        "test_allocation": split.test_allocation.tolist(),
        "clients": clients,
        "public_calibration": data_state["public_metadata"],
        "client_training_pool": data_state["training_pool_metadata"],
    }
    if data_state["classifier_metadata"] is not None:
        manifest["semantic_classifier_training"] = data_state[
            "classifier_metadata"
        ]
    for name in ("alpha", "classes_per_client", "target_label"):
        if name in config["split"]:
            manifest[name] = config["split"][name]
    if data_state["backdoor_metadata"] is not None:
        manifest["target_client_backdoor"] = data_state["backdoor_metadata"]
    return manifest


def uses_external_semantic_classifier(config):
    """Return whether FSCU should reuse an independent classifier run."""

    source_run = config.get("semantic_classifier", {}).get("source_run")
    return isinstance(source_run, str) and bool(source_run.strip())


def prepare_fscu_data(comm, config, split_seed):
    """Assign rank-local FSCU data without centralizing raw client datasets."""
    rank = comm.Get_rank()
    n_clients = int(config["federation"]["n_clients"])
    dataset_config = config["dataset"]
    target_client_id = int(
        config.get("unlearning", {}).get("target_client_id", 0)
    )
    target_rank = target_client_id + 1
    is_training_curves = config.get("algorithm") == "training_curves"
    is_classification_task = config.get("task") == "classification"
    has_external_classifier = uses_external_semantic_classifier(config)

    if rank == 0:
        test_dataset = load_cifar10_test_dataset(dataset_config, WORKSPACE_ROOT)
        payload = comm.recv(source=1, tag=701)
        backdoor_metadata = (
            comm.recv(source=target_rank, tag=702)
            if config.get("backdoor") is not None
            else None
        )
        test_data = select_cifar10_client_data(
            test_dataset, np.arange(len(test_dataset), dtype=np.int64)
        )
        data_state = {
            "client_data": None,
            "clean_client_data": None,
            "split": payload["split"],
            "test_data": test_data,
            "client_test_indices": payload["split"].test_indices,
            "public_data": payload["public_data"],
            "public_metadata": payload["public_metadata"],
            "classifier_data": payload["classifier_data"],
            "classifier_metadata": payload["classifier_metadata"],
            "training_pool_metadata": payload["training_pool_metadata"],
            "backdoor_metadata": backdoor_metadata,
            "split_seed": int(split_seed),
        }
        data_state["split_manifest"] = build_fscu_split_manifest(
            config, data_state["split"], split_seed, data_state
        )
        return data_state

    train_dataset = load_cifar10_train_dataset(dataset_config, WORKSPACE_ROOT)
    test_dataset = load_cifar10_test_dataset(dataset_config, WORKSPACE_ROOT)
    train_labels = dataset_labels(train_dataset)
    public_config = config["public_data"]
    public_indices, client_pool_indices = reserve_class_balanced_public_indices(
        train_labels,
        int(public_config["samples_per_class"]),
        int(public_config.get("seed", split_seed)),
    )
    public_data = None
    if not is_training_curves:
        public_data = select_cifar10_client_data(train_dataset, public_indices)
    public_labels, public_counts = np.unique(
        train_labels[public_indices], return_counts=True
    )
    public_metadata = {
        "source": "cifar10_train_heldout_before_client_split",
        "samples_per_class": int(public_config["samples_per_class"]),
        "size": int(len(public_indices)),
        "seed": int(public_config.get("seed", split_seed)),
        "index_sha256": hashlib.sha256(public_indices.tobytes()).hexdigest(),
        "label_count": {
            str(label): int(count)
            for label, count in zip(public_labels, public_counts)
        },
    }

    requested_pool_size = n_clients * int(
        config["federation"]["samples_per_client"]
    )
    if requested_pool_size > len(client_pool_indices):
        raise ValueError("requested client samples exceed the non-public pool")
    pool_seed = derive_seed(split_seed, "client-training-pool")
    if requested_pool_size < len(client_pool_indices):
        client_pool_indices, unused_indices = select_class_balanced_pool_subset(
            train_labels,
            client_pool_indices,
            requested_pool_size,
            pool_seed,
        )
    else:
        unused_indices = np.empty(0, dtype=np.int64)
    selected_labels, selected_counts = np.unique(
        train_labels[client_pool_indices], return_counts=True
    )
    training_pool_metadata = {
        "source": "class_balanced_subset_after_public_reservation",
        "candidate_size": int(requested_pool_size + len(unused_indices)),
        "selected_size": int(len(client_pool_indices)),
        "unused_size": int(len(unused_indices)),
        "seed": int(pool_seed),
        "selected_index_sha256": hashlib.sha256(
            client_pool_indices.tobytes()
        ).hexdigest(),
        "unused_index_sha256": hashlib.sha256(
            unused_indices.tobytes()
        ).hexdigest(),
        "selected_label_count": {
            str(label): int(count)
            for label, count in zip(selected_labels, selected_counts)
        },
    }
    classifier_data = None
    classifier_metadata = None
    if (
        not is_training_curves
        and not is_classification_task
        and not has_external_classifier
    ):
        classifier_samples_per_class = int(
            config["semantic_classifier"]["training"]["samples_per_class"]
        )
        classifier_size = classifier_samples_per_class * len(
            np.unique(train_labels)
        )
        if classifier_size > len(unused_indices):
            raise ValueError(
                "semantic classifier samples exceed the clean pool unused by clients"
            )
        classifier_indices, _ = select_class_balanced_pool_subset(
            train_labels,
            unused_indices,
            classifier_size,
            derive_seed(split_seed, "semantic-classifier-training-pool"),
        )
        classifier_data = select_cifar10_client_data(
            train_dataset, classifier_indices
        )
        classifier_labels, classifier_counts = np.unique(
            train_labels[classifier_indices], return_counts=True
        )
        classifier_metadata = {
            "source": "clean_class_balanced_cifar10_train_unused_by_clients_or_public_calibration",
            "samples_per_class": classifier_samples_per_class,
            "size": int(len(classifier_indices)),
            "seed": int(
                derive_seed(split_seed, "semantic-classifier-training-pool")
            ),
            "index_sha256": hashlib.sha256(
                classifier_indices.tobytes()
            ).hexdigest(),
            "label_count": {
                str(label): int(count)
                for label, count in zip(classifier_labels, classifier_counts)
            },
            "overlaps_public_calibration": False,
            "overlaps_client_training_pool": False,
        }

    relative_split = _build_fscu_split(
        train_labels[client_pool_indices],
        dataset_labels(test_dataset),
        config,
        split_seed,
    )
    split = _copy_fscu_split(
        relative_split,
        [
            client_pool_indices[relative_indices]
            for relative_indices in relative_split.train_indices
        ],
    )
    client_id = rank - 1
    client_data = select_cifar10_client_data(
        train_dataset, split.train_indices[client_id]
    )
    clean_client_data = None
    backdoor_metadata = None
    if (
        config.get("backdoor") is not None
        and client_id == target_client_id
    ):
        # Keep the unmodified source only on the target client.  Classifier
        # training uses it to apply crop/flip before stamping the fixed trigger,
        # so an augmented copy of the original patch cannot survive as a ghost
        # trigger at a second location.
        clean_client_data = client_data
        backdoor = config["backdoor"]
        client_data, backdoor_metadata = inject_scu_cifar10_backdoor(
            client_data,
            poison_ratio=backdoor["poison_ratio"],
            base_label=backdoor["base_label"],
            trigger_label=backdoor["trigger_label"],
            seed=derive_seed(split_seed, "scu-target-backdoor"),
        )
    if rank == 1:
        comm.send(
            {
                "split": split,
                "public_data": public_data,
                "public_metadata": public_metadata,
                "classifier_data": classifier_data,
                "classifier_metadata": classifier_metadata,
                "training_pool_metadata": training_pool_metadata,
            },
            dest=0,
            tag=701,
        )
    if backdoor_metadata is not None:
        comm.send(backdoor_metadata, dest=0, tag=702)
    return {
        "client_data": client_data,
        "clean_client_data": clean_client_data,
        "split": None,
        "test_data": None,
        "client_test_indices": None,
        # Public calibration data is intentionally shared with every client.
        # The held-out public set is never mixed into private client training data.
        "public_data": public_data,
        "public_metadata": None,
        "classifier_data": None,
        "classifier_metadata": None,
        "training_pool_metadata": None,
        "backdoor_metadata": backdoor_metadata,
        "split_seed": int(split_seed),
        "split_manifest": None,
    }


def prepare_scu_data(comm, args):
    """Load a reproducible single-client CIFAR-10 SCU baseline subset."""
    if comm.Get_size() != 1 or int(args.n_clients) != 1:
        raise ValueError("SCU runs with one MPI process and one client")
    if args.dataset != "cifar10":
        raise ValueError("SCU now supports CIFAR-10 only")
    dataset_config = {
        "name": "CIFAR10",
        "data_root": str(Path(args.data_path) / "cifar10"),
        "download": True,
    }
    train = load_cifar10_train_dataset(dataset_config, WORKSPACE_ROOT)
    test = load_cifar10_test_dataset(dataset_config, WORKSPACE_ROOT)
    requested = int(args.n_client_data)
    if not 2 <= requested <= len(train):
        raise ValueError("n_client_data must be between 2 and the CIFAR-10 train size")
    generator = np.random.default_rng(args.seed)
    selected = generator.permutation(len(train))[:requested]
    client_data = select_cifar10_client_data(train, selected)
    test_data = select_cifar10_client_data(
        test, np.arange(len(test), dtype=np.int64)
    )
    return {
        "client_data": client_data,
        "test_data": test_data,
    }


def create_scu_dataset(data, dataset_name="cifar10"):
    """Convert one client's CIFAR-10 arrays to the SCU tensor dataset."""
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("SCU requires client data as (images, labels)")

    images, labels = data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    if dataset_name != "cifar10":
        raise ValueError("SCU now supports CIFAR-10 only")
    expected_shape = (3, 32, 32)

    if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
        raise ValueError(
            f"SCU {dataset_name} images must be shaped (N, {expected_shape[0]}, "
            f"{expected_shape[1]}, {expected_shape[2]})"
        )
    if len(images) != len(labels):
        raise ValueError("SCU images and labels must have the same length")
    if len(images) < 2:
        raise ValueError("SCU requires at least two client training samples")

    if images.max() > 1.0:
        images = images / 255.0
    return TensorDataset(images, labels)


def split_scu_data(dataset, forget_ratio, seed):
    """한 client dataset을 삭제할 D^t와 유지할 D^r로 seed 고정 분할한다."""
    if not 0.0 < forget_ratio < 1.0:
        raise ValueError("scu_forget_ratio must be between 0 and 1")

    target_count = int(len(dataset) * forget_ratio)
    target_count = max(1, min(target_count, len(dataset) - 1))
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()
    target_indices = shuffled_indices[:target_count]
    retained_indices = shuffled_indices[target_count:]

    target_data = Subset(dataset, target_indices)
    retained_data = Subset(dataset, retained_indices)
    return target_data, retained_data, target_indices, retained_indices


def get_scu_dataloader(dataset, batch_size, seed):
    """SCU 학습에서 매 epoch 재현 가능하게 shuffle되는 DataLoader를 만든다."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
