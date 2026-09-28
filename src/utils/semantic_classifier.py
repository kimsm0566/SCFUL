"""Training helpers for the clean classifier kept outside the FL codec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.data.data import augment_cifar10_contrastive_batch
from src.models.semantic_classifier import (
    CIFAR10_CLASSIFIER_ARCHITECTURE,
    Cifar10Classifier,
    Cifar10ClassifierConfig,
)
from src.utils.reproducibility import derive_seed


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_CONFIG_FIELDS = (
    "architecture",
    "base_channels",
    "group_count",
    "num_classes",
    "channel_mean",
    "channel_std",
)


def _normalized_tensor_data(data):
    images, labels = data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    if len(images) != len(labels):
        raise ValueError("classifier images and labels must have equal length")
    if len(images) == 0:
        raise ValueError("classifier data must not be empty")
    return images, labels


def make_semantic_classifier(config, state=None, device="cpu", frozen=True):
    """Construct the separate CIFAR-10 classifier from an effective config."""

    values = dict(config["semantic_classifier"])
    architecture = values.pop("architecture", CIFAR10_CLASSIFIER_ARCHITECTURE)
    values.pop("training", None)
    values.pop("source_run", None)
    values.pop("source_state_sha256", None)
    values.pop("source_checkpoint_sha256", None)
    if architecture != CIFAR10_CLASSIFIER_ARCHITECTURE:
        raise ValueError(
            "semantic_classifier.architecture must be "
            f"{CIFAR10_CLASSIFIER_ARCHITECTURE}; received {architecture}"
        )
    model = Cifar10Classifier(Cifar10ClassifierConfig(**values)).to(device)
    if state is not None:
        model.load_state_dict(state)
    if frozen:
        model.freeze()
    return model


def classifier_state_sha256(state):
    """Hash a classifier state independently of torch serialization details."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required classifier source file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid classifier source JSON at {path}: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"classifier source JSON must contain an object: {path}")
    return value


def _is_completed_classifier_run(path):
    required = (
        "config.json",
        "run_metadata.json",
        "split_manifest.json",
        "metrics/training_curve_summary.json",
        "checkpoints/original.pt",
    )
    return path.is_dir() and all((path / name).is_file() for name in required)


def resolve_external_classifier_run(source_run, workspace_root=None):
    """Resolve one completed classifier run without allowing workspace escape."""

    workspace = Path(
        WORKSPACE_ROOT if workspace_root is None else workspace_root
    ).resolve()
    if not isinstance(source_run, str) or not source_run.strip():
        raise ValueError("semantic_classifier.source_run must be a non-empty path")
    requested = Path(source_run)
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (workspace / requested).resolve()
    )
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(
            "semantic_classifier.source_run must remain inside the workspace"
        ) from error
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"semantic classifier source directory does not exist: {resolved}"
        )
    if _is_completed_classifier_run(resolved):
        return resolved
    candidates = sorted(
        child
        for child in resolved.iterdir()
        if _is_completed_classifier_run(child)
    )
    if len(candidates) != 1:
        raise ValueError(
            "semantic_classifier.source_run must be a completed classification "
            "run or contain exactly one completed run: "
            f"{resolved}"
        )
    return candidates[0]


def _classifier_architecture_values(section):
    architecture = section.get(
        "architecture", CIFAR10_CLASSIFIER_ARCHITECTURE
    )
    values = {
        name: section[name]
        for name in _CLASSIFIER_CONFIG_FIELDS
        if name != "architecture" and name in section
    }
    normalized = Cifar10ClassifierConfig(**values)
    return {
        "architecture": architecture,
        "base_channels": normalized.base_channels,
        "group_count": normalized.group_count,
        "num_classes": normalized.num_classes,
        "channel_mean": list(normalized.channel_mean),
        "channel_std": list(normalized.channel_std),
    }


def _load_classifier_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "external semantic classifier checkpoint must be a non-empty state dict"
        )
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(
            "external semantic classifier checkpoint contains non-tensor state"
        )
    return {
        name: value.detach().cpu().clone() for name, value in state.items()
    }


def load_external_semantic_classifier(
    config,
    test_data,
    device,
    workspace_root=None,
):
    """Load, verify, and independently evaluate a completed classifier run."""

    classifier_config = config["semantic_classifier"]
    source_run = resolve_external_classifier_run(
        classifier_config["source_run"], workspace_root=workspace_root
    )
    workspace = Path(
        WORKSPACE_ROOT if workspace_root is None else workspace_root
    ).resolve()
    source_config = _read_json_object(source_run / "config.json")
    source_metadata = _read_json_object(source_run / "run_metadata.json")
    source_summary = _read_json_object(
        source_run / "metrics" / "training_curve_summary.json"
    )
    source_split = _read_json_object(source_run / "split_manifest.json")
    if (
        source_config.get("task") != "classification"
        or source_config.get("algorithm") != "training_curves"
        or source_config.get("model", {}).get("kind") != "cifar10_classifier"
        or source_metadata.get("task") != "classification"
        or source_summary.get("task") != "classification"
    ):
        raise ValueError(
            "semantic_classifier.source_run is not a completed classification "
            "training_curves run"
        )

    expected_architecture = _classifier_architecture_values(classifier_config)
    source_architecture = _classifier_architecture_values(source_config["model"])
    if source_architecture != expected_architecture:
        raise ValueError(
            "external semantic classifier architecture/configuration differs "
            f"from semantic_classifier: {source_architecture!r} != "
            f"{expected_architecture!r}"
        )

    source_pool = source_split.get("client_training_pool", {})
    source_pool_hash = source_pool.get("selected_index_sha256")
    if not isinstance(source_pool_hash, str) or not source_pool_hash:
        raise ValueError(
            "external classifier split manifest has no client training-pool hash"
        )

    checkpoint_path = (source_run / "checkpoints" / "original.pt").resolve()
    try:
        checkpoint_path.relative_to(source_run)
    except ValueError as error:
        raise ValueError(
            "external semantic classifier checkpoint escapes its source run"
        ) from error
    checkpoint_hash = _file_sha256(checkpoint_path)
    expected_checkpoint_hash = classifier_config.get(
        "source_checkpoint_sha256"
    )
    if (
        expected_checkpoint_hash is not None
        and checkpoint_hash != expected_checkpoint_hash.lower()
    ):
        raise ValueError("external semantic classifier checkpoint SHA-256 mismatch")

    state = _load_classifier_state(checkpoint_path)
    state_hash = classifier_state_sha256(state)
    expected_state_hash = classifier_config.get("source_state_sha256")
    if expected_state_hash is not None and state_hash != expected_state_hash.lower():
        raise ValueError("external semantic classifier state SHA-256 mismatch")

    classifier = make_semantic_classifier(
        config, state=state, device=device, frozen=True
    )
    evaluation_batch_size = int(
        config.get("evaluation", {}).get(
            "batch_size",
            classifier_config.get("training", {}).get("batch_size", 128),
        )
    )
    test_metrics = evaluate_frozen_classifier(
        classifier,
        test_data,
        batch_size=evaluation_batch_size,
        device=device,
    )
    metadata = {
        "architecture": source_architecture["architecture"],
        "architecture_config": source_architecture,
        "role": "frozen_external_semantic_observer",
        "preparation": "loaded_from_completed_classification_training_curves",
        "source_run": str(source_run),
        "source_run_workspace_relative": str(source_run.relative_to(workspace)),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_relative": str(checkpoint_path.relative_to(source_run)),
        "source_checkpoint_sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "source_split_type": source_split.get("partition"),
        "source_split_seed": source_split.get("seed"),
        "source_split_training_pool_index_sha256": source_pool_hash,
        "source_global_seed": source_metadata.get("global_seed"),
        "raw_cifar10_test_loss": test_metrics["loss"],
        "raw_cifar10_test_accuracy": test_metrics["accuracy"],
        "raw_cifar10_test_examples": test_metrics["examples"],
    }
    return state, metadata, checkpoint_path


def _evaluate_classifier(model, data, batch_size, device):
    images, labels = _normalized_tensor_data(data)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    loss_total = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_images)
            loss_total += float(
                functional.cross_entropy(
                    logits, batch_labels, reduction="sum"
                ).cpu()
            )
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            examples += len(batch_labels)
    model.train(was_training)
    return {
        "loss": loss_total / examples,
        "accuracy": correct / examples,
        "examples": examples,
    }


def train_semantic_classifier(config, train_data, test_data, seed, device):
    """Train one clean observer on data disjoint from every FL client."""

    settings = config["semantic_classifier"]["training"]
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    learning_rate = float(settings["learning_rate"])
    weight_decay = float(settings["weight_decay"])
    crop_padding = int(settings["augmentation_crop_padding"])
    flip_probability = float(
        settings["augmentation_horizontal_flip_probability"]
    )
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("classifier epochs, batch_size, and learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("classifier weight_decay must be non-negative")

    device = torch.device(device)
    train_images, train_labels = _normalized_tensor_data(train_data)
    loader_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(seed, "semantic-classifier-loader")
    )
    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    augmentation_generator = torch.Generator(
        device=generator_device
    ).manual_seed(derive_seed(seed, "semantic-classifier-augmentation"))

    fork_devices = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        initialization_seed = derive_seed(seed, "semantic-classifier-initialization")
        torch.manual_seed(initialization_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(initialization_seed)
        model = make_semantic_classifier(
            config, state=None, device=device, frozen=False
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    history = []
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        correct = 0
        examples = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = augment_cifar10_contrastive_batch(
                images,
                augmentation_generator,
                crop_padding=crop_padding,
                horizontal_flip_probability=flip_probability,
            )
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += len(labels)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "train_loss": loss_total / examples,
                "train_accuracy": correct / examples,
            }
        )

    test_metrics = _evaluate_classifier(
        model, test_data, batch_size=batch_size, device=device
    )
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    metadata = {
        "architecture": CIFAR10_CLASSIFIER_ARCHITECTURE,
        "training_seed": int(seed),
        "epochs": epochs,
        "history": history,
        "raw_cifar10_test_loss": test_metrics["loss"],
        "raw_cifar10_test_accuracy": test_metrics["accuracy"],
        "raw_cifar10_test_examples": test_metrics["examples"],
        "state_sha256": classifier_state_sha256(state),
        "role": "frozen_clean_observer_and_semantic_loss_model",
        "overlaps_federated_client_training_data": False,
    }
    return state, metadata


def evaluate_frozen_classifier(classifier, data, batch_size, device):
    """Expose raw/reconstructed-image classifier accuracy for diagnostics."""

    return _evaluate_classifier(classifier, data, batch_size, torch.device(device))


def load_remaining_classifier(source_run, codec_config, codec_manifest, device="cpu", allow_smoke=False):
    """Verify target/split provenance before using a new common frozen observer."""
    from src.data.data import remaining_classifier_source

    source_run = Path(source_run).resolve()
    metadata = _read_json_object(source_run / "observer_manifest.json")
    if metadata.get("kind") != "remaining_client_classifier_v1" or metadata.get("completed") is not True:
        raise ValueError("observer must be a completed remaining-client classifier")
    protocol = metadata["protocol"]
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != metadata["protocol_sha256"]:
        raise ValueError("observer protocol SHA-256 mismatch")
    if _read_json_object(source_run / "config.json") != protocol["config"]:
        raise ValueError("observer training config mismatch")
    if _classifier_architecture_values(protocol["config"]["model"]) != metadata["architecture_config"]:
        raise ValueError("observer architecture metadata mismatch")
    if metadata.get("initialization") != "random_from_scratch_no_original_classifier" or metadata.get("codec_gradient") is not False:
        raise ValueError("observer must be independently initialized and detached from codec")
    if protocol["smoke_test"] and not allow_smoke:
        raise ValueError("smoke observer cannot be used for a full audit")
    scenario, fingerprint = remaining_classifier_source(codec_config, codec_manifest)
    if protocol["source_fingerprint"] != fingerprint or metadata["source_scenario"] != scenario:
        raise ValueError("observer target/client split fingerprint mismatch")
    expected = {row["client_id"]: set(row["train_indices"]) for row in scenario["clients"] if row["client_id"] != scenario["target_client_id"]}
    rows = protocol["client_splits"]
    if len(rows) != len(expected) or {row["client_id"] for row in rows} != set(expected):
        raise ValueError("observer includes target or omits remaining clients")
    for row in rows:
        train, validation = row["train_indices"], row["validation_indices"]
        union = set(train) | set(validation)
        if (not train or not validation or len(union) != len(train) + len(validation)
                or not union <= expected[row["client_id"]]
                or (not protocol["smoke_test"] and union != expected[row["client_id"]])):
            raise ValueError("observer train/validation indices are invalid or overlap target")
    recorded_split = _read_json_object(source_run / "split_manifest.json")
    if recorded_split.get("clients") != rows or recorded_split.get("source_scenario") != scenario:
        raise ValueError("observer split artifact mismatch")
    checkpoint = source_run / "checkpoints/observer.pt"
    if _file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("observer checkpoint SHA-256 mismatch")
    state = _load_classifier_state(checkpoint)
    if classifier_state_sha256(state) != metadata["state_sha256"]:
        raise ValueError("observer state SHA-256 mismatch")
    classifier = make_semantic_classifier(
        {"semantic_classifier": metadata["architecture_config"]}, state=state, device=device, frozen=True,
    )
    return classifier, {**metadata, "observer_run": str(source_run)}
