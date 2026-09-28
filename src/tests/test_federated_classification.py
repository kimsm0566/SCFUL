from pathlib import Path

import torch

import src.utils.federated as federated_module

from src.models.semantic_classifier import (
    Cifar10Classifier,
    Cifar10ClassifierConfig,
)
from src.utils.config import load_training_curve_config
from src.utils.federated import (
    LocalClassificationConfig,
    train_local_classification,
)
from src.utils.fscu_runtime import (
    _evaluate_federated_classification_state,
    format_federated_classification_round_log,
    make_federated_classifier,
    make_initial_federated_classifier_state,
)


ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_classifier_800rounds.json"
)


def _small_classification_config():
    config = load_training_curve_config(CLASSIFICATION_CONFIG)
    config["model"]["base_channels"] = 8
    config["model"]["group_count"] = 4
    return config


def test_classification_curve_config_has_no_codec_or_observer_path():
    config = load_training_curve_config(CLASSIFICATION_CONFIG)

    assert config["task"] == "classification"
    assert config["model"]["kind"] == "cifar10_classifier"
    assert config["federation"]["persist_client_optimizer_state"] is False
    assert "codec" not in config
    assert "semantic_classifier" not in config
    assert "unlearning" not in config
    assert "semantic_weight" not in config["training"]
    assert "reconstruction_weight" not in config["training"]
    assert "snr_db" not in config["training"]


def test_local_classification_updates_only_classifier_state_and_reports_ce_accuracy():
    model = Cifar10Classifier(
        Cifar10ClassifierConfig(base_channels=8, group_count=4)
    )
    config = LocalClassificationConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        augmentation="none",
        device="cpu",
    )
    images = torch.rand(8, 3, 32, 32)
    labels = torch.arange(8, dtype=torch.long) % 10
    before = {name: value.clone() for name, value in model.state_dict().items()}

    state, metrics = train_local_classification(
        model, (images, labels), config, seed=7
    )

    assert set(metrics) == {"cross_entropy", "accuracy", "examples", "steps"}
    assert metrics["cross_entropy"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert any(not torch.equal(before[name], state[name]) for name in before)
    assert all("encoder" not in name and "decoder" not in name for name in state)


def test_backdoor_crop_flip_uses_clean_source_then_stamps_one_fixed_trigger(monkeypatch):
    class RecordingClassifier(torch.nn.Module):
        seen_inputs = []

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            type(self).seen_inputs.append(inputs.detach().cpu().clone())
            pooled = inputs.mean(dim=(1, 2, 3)) * self.scale
            return pooled.unsqueeze(1).repeat(1, 10)

    def deterministic_horizontal_flip(images, **_):
        return images.flip(dims=(3,))

    monkeypatch.setattr(
        federated_module,
        "augment_cifar10_contrastive_batch",
        deterministic_horizontal_flip,
    )
    clean_images = torch.zeros(2, 3, 32, 32)
    clean_images[0, :, 28:30, 28:30] = 0.25
    clean_labels = torch.tensor([1, 0])
    poisoned_images = clean_images.clone()
    poisoned_labels = clean_labels.clone()
    poisoned_images[0, :, 28:30, 28:30] = 1.0
    poisoned_labels[0] = 2
    config = LocalClassificationConfig(
        batch_size=2,
        local_epochs=1,
        learning_rate=1e-3,
        augmentation="cifar10_crop_flip_v1",
        preserve_fixed_backdoor_trigger=True,
        device="cpu",
    )
    RecordingClassifier.seen_inputs = []

    train_local_classification(
        RecordingClassifier(),
        (poisoned_images, poisoned_labels),
        config,
        seed=7,
        clean_client_data=(clean_images, clean_labels),
        fixed_trigger_positions=[0],
    )

    seen = torch.cat(RecordingClassifier.seen_inputs)
    triggered = seen[
        seen[:, :, 28:30, 28:30].flatten(start_dim=1).eq(1.0).all(dim=1)
    ]
    assert len(triggered) == 1
    assert torch.all(triggered[0, :, 28:30, 28:30] == 1.0)
    assert torch.all(triggered[0, :, 28:30, 2:4] == 0.25)
    assert not torch.any(triggered[0, :, 28:30, 2:4] == 1.0)


def test_classifier_initialization_and_evaluation_never_build_codec_metrics():
    config = _small_classification_config()
    state = make_initial_federated_classifier_state(config, seed=3)
    model = make_federated_classifier(config)
    model.load_state_dict(state)
    metrics = _evaluate_federated_classification_state(
        config,
        torch.device("cpu"),
        state,
        (torch.rand(4, 3, 32, 32), torch.arange(4, dtype=torch.long)),
        batch_size=2,
    )

    assert set(metrics) == {
        "test_cross_entropy",
        "test_accuracy",
        "test_examples",
    }
    assert all("encoder" not in name and "decoder" not in name for name in state)


def test_classifier_round_log_contains_only_ce_and_accuracy():
    line = format_federated_classification_round_log(
        {
            "phase": "original",
            "round": 2,
            "train_cross_entropy": 1.25,
            "train_accuracy": 0.5,
            "test_cross_entropy": 1.5,
            "test_accuracy": 0.4,
        },
        total_rounds=10,
    )

    assert "train_ce=1.250000" in line
    assert "train_accuracy=50.00%" in line
    assert "test_ce=1.500000" in line
    assert "test_accuracy=40.00%" in line
    assert "mse" not in line
    assert "psnr" not in line
    assert "semantic" not in line
