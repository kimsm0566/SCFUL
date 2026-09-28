import math

import pytest
import torch

import src.utils.federated as federated_module
from src.models.semantic_classifier import Cifar10Classifier, Cifar10ClassifierConfig
from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    VariationalDeepJSCC,
    VariationalDeepJSCCConfig,
    variational_deepjscc_config_for_architecture,
)
from src.utils.federated import (
    LocalReconstructionConfig,
    average_model_states,
    train_local_reconstruction,
    train_local_reconstruction_with_optimizer_state,
)
from src.utils.fscu_runtime import (
    _evaluate_federated_training_state,
    effective_federated_learning_rate,
    format_federated_round_log,
)


def test_weighted_average_model_states_respects_sample_counts():
    states = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([5.0])}]

    result = average_model_states(states, sample_counts=[1, 3])

    assert torch.equal(result["weight"], torch.tensor([4.0]))


def test_round_log_contains_losses_and_train_and_test_metrics():
    line = format_federated_round_log(
        {
            "phase": "original",
            "round": 3,
            "learning_rate": 1e-4,
            "objective_loss": 1.25,
            "reconstruction_loss": 30.72,
            "train_mse": 0.01,
            "kl_loss": 0.2,
            "semantic_loss": 2.1,
            "semantic_accuracy": 0.4,
            "test_normalized_mse": 0.02,
            "test_psnr_db": 16.9897,
            "test_accuracy": 0.35,
            "test_posterior_std_mean": 0.1,
            "test_posterior_logvar_mean": -4.0,
            "test_prior_kl_loss": 5.0,
        },
        total_rounds=10,
    )

    assert "round=3/10" in line
    assert "train_mse=0.010000" in line
    assert "train_psnr_db=20.000000" in line
    assert "train_accuracy=40.00%" in line
    assert "objective_loss=1.250000" in line
    assert "learning_rate=0.000100" in line
    assert "reconstruction_loss=30.720000" in line
    assert "kl_loss=0.200000" in line
    assert "semantic_loss=2.100000" in line
    assert "test_mse=0.020000" in line
    assert "test_accuracy=35.00%" in line


def test_reconstruction_round_log_reports_unavailable_classifier_fields():
    line = format_federated_round_log(
        {
            "phase": "original",
            "round": 1,
            "objective_loss": 0.5,
            "reconstruction_loss": 30.72,
            "train_mse": 0.01,
            "kl_loss": 0.2,
            "test_normalized_mse": 0.02,
            "test_psnr_db": 16.9897,
        },
        total_rounds=10,
    )

    assert "train_mse=0.010000" in line
    assert "test_mse=0.020000" in line
    assert "train_accuracy=n/a" in line
    assert "semantic_loss=n/a" in line
    assert "test_accuracy=n/a" in line


class FixedZeroCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images, **kwargs):
        del kwargs
        output = type("Output", (), {})()
        output.reconstruction = torch.zeros_like(images) + 0.0 * self.dummy
        output.posterior = None
        return output

    def prior_kl(self, posterior):
        del posterior
        return 0.0 * self.dummy


class FailOnForwardObserver(torch.nn.Module):
    def freeze(self):
        self.eval()
        return self

    def forward(self, images):
        del images
        raise AssertionError("zero semantic weight must not call the observer")


def test_local_reconstruction_uses_versioned_loss_and_example_weighting():
    images = torch.cat(
        [torch.ones(2, 1, 2, 2), torch.zeros(1, 1, 2, 2)],
        dim=0,
    )
    cases = [
        ("batchmean_sum", 8.0 / 3.0),
        ("elementwise_mean", 2.0 / 3.0),
    ]
    for kl_reduction, expected_reconstruction in cases:
        config = LocalReconstructionConfig(
            batch_size=2,
            local_epochs=1,
            learning_rate=1e-3,
            beta=0.0,
            snr_db=20.0,
            kl_reduction=kl_reduction,
            gradient_clip_norm=None,
            device="cpu",
        )

        _, metrics = train_local_reconstruction(
            FixedZeroCodec(),
            (images, torch.zeros(3, dtype=torch.long)),
            config,
            seed=3,
        )

        assert metrics["reconstruction_loss"] == pytest.approx(
            expected_reconstruction
        )
        assert metrics["objective_loss"] == pytest.approx(
            expected_reconstruction
        )
        assert metrics["train_mse"] == pytest.approx(2.0 / 3.0)
        assert metrics["processed_examples"] == 3
        assert metrics["steps"] == 2


def test_zero_semantic_weight_never_connects_observer_to_codec_gradient():
    config = LocalReconstructionConfig(
        batch_size=2,
        local_epochs=1,
        learning_rate=1e-3,
        beta=0.0,
        semantic_weight=0.0,
        snr_db=20.0,
        device="cpu",
    )

    state, metrics = train_local_reconstruction(
        FixedZeroCodec(),
        (torch.rand(2, 3, 32, 32), torch.tensor([0, 1])),
        config,
        seed=3,
        semantic_classifier=FailOnForwardObserver(),
    )

    assert state["dummy"].item() == 0.0
    assert metrics["semantic_loss"] is None
    assert metrics["semantic_accuracy"] is None


def test_effective_learning_rate_changes_on_configured_one_based_round():
    config = {
        "training": {
            "learning_rate": 1e-3,
            "learning_rate_decay_round": 3,
            "learning_rate_after_decay": 2e-4,
        }
    }

    assert effective_federated_learning_rate(config, 2) == 1e-3
    assert effective_federated_learning_rate(config, 3) == 2e-4


def test_sample_posterior_evaluation_is_repeatable_and_records_diagnostics():
    model_config = variational_deepjscc_config_for_architecture(
        DEEPJSCC_ARCHITECTURE,
        latent_dim=64,
        latent_channels=1,
    )
    model = VariationalDeepJSCC(model_config)
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    config = {
        "codec": {
            "architecture": DEEPJSCC_ARCHITECTURE,
            **vars(model_config),
        },
        "evaluation": {
            "snr_db": 20.0,
            "posterior_mode": "sample",
            "monte_carlo_repetitions": 2,
        },
    }
    data = (
        torch.rand(3, 3, 32, 32),
        torch.zeros(3, dtype=torch.long),
    )

    first = _evaluate_federated_training_state(
        config, torch.device("cpu"), state, data, 2, 17
    )
    second = _evaluate_federated_training_state(
        config, torch.device("cpu"), state, data, 2, 17
    )

    assert first == second
    assert first["test_posterior_mode"] == "sample"
    assert first["test_monte_carlo_repetitions"] == 2
    assert first["test_examples"] == 3
    for name in (
        "test_normalized_mse",
        "test_posterior_std_mean",
        "test_posterior_logvar_mean",
        "test_prior_kl_loss",
    ):
        assert math.isfinite(first[name])


def test_local_reconstruction_update_changes_codec_weights():
    model = VariationalDeepJSCC(
        VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    )
    images = torch.rand(8, 3, 32, 32)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    config = LocalReconstructionConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        snr_db=10.0,
        device="cpu",
    )

    state, metrics = train_local_reconstruction(
        model, (images, torch.zeros(8, dtype=torch.long)), config, seed=1
    )

    assert any(not torch.equal(before[name], state[name]) for name in before)
    assert metrics["reconstruction_loss"] >= 0.0
    assert metrics["examples"] == 8


def test_local_crop_flip_uses_seeded_view_as_input_and_reconstruction_target(
    monkeypatch,
):
    class IdentityCodec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, images, **kwargs):
            del kwargs
            output = type("Output", (), {})()
            output.reconstruction = images + self.bias
            output.posterior = None
            return output

        def prior_kl(self, posterior):
            del posterior
            return self.bias.square()

    observed_seeds = []

    def fixed_augmentation(images, generator, **kwargs):
        del kwargs
        observed_seeds.append(generator.initial_seed())
        return torch.full_like(images, 0.25)

    monkeypatch.setattr(
        federated_module,
        "augment_cifar10_contrastive_batch",
        fixed_augmentation,
    )
    config = LocalReconstructionConfig(
        batch_size=2,
        local_epochs=1,
        learning_rate=1e-3,
        beta=0.0,
        snr_db=20.0,
        augmentation="cifar10_crop_flip_v1",
        device="cpu",
    )

    _, metrics = train_local_reconstruction(
        IdentityCodec(),
        (torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)),
        config,
        seed=11,
    )

    assert observed_seeds == [14]
    assert metrics["reconstruction_loss"] == 0.0


def test_local_adam_state_can_be_reused_across_federated_rounds():
    config_model = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(config_model)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.zeros(8, dtype=torch.long)
    config = LocalReconstructionConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        snr_db=10.0,
        device="cpu",
    )

    first_state, first_metrics, optimizer_state = (
        train_local_reconstruction_with_optimizer_state(
            model,
            (images, labels),
            config,
            seed=1,
        )
    )
    next_global_model = VariationalDeepJSCC(config_model)
    next_global_model.load_state_dict(first_state)
    config.learning_rate = 2e-4
    _, second_metrics, next_optimizer_state = (
        train_local_reconstruction_with_optimizer_state(
            next_global_model,
            (images, labels),
            config,
            seed=2,
            optimizer_state=optimizer_state,
        )
    )

    assert first_metrics["optimizer_state_reused"] is False
    assert second_metrics["optimizer_state_reused"] is True
    assert second_metrics["learning_rate"] == 2e-4
    assert next_optimizer_state["param_groups"][0]["lr"] == 2e-4
    assert optimizer_state["state"]
    assert next_optimizer_state["state"]
    first_steps = [
        float(state["step"])
        for state in optimizer_state["state"].values()
        if "step" in state
    ]
    next_steps = [
        float(state["step"])
        for state in next_optimizer_state["state"].values()
        if "step" in state
    ]
    assert first_steps
    assert min(next_steps) > min(first_steps)
    for state in next_optimizer_state["state"].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"


def test_frozen_separate_classifier_updates_codec_but_not_classifier():
    model = VariationalDeepJSCC(
        VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    )
    classifier = Cifar10Classifier(
        Cifar10ClassifierConfig(base_channels=8, group_count=4)
    ).freeze()
    classifier_before = {
        name: value.clone() for name, value in classifier.state_dict().items()
    }
    codec_before = {name: value.clone() for name, value in model.state_dict().items()}
    config = LocalReconstructionConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        beta=0.0,
        snr_db=20.0,
        reconstruction_weight=0.0,
        semantic_weight=1.0,
        device="cpu",
    )

    codec_state, metrics = train_local_reconstruction(
        model,
        (torch.rand(4, 3, 32, 32), torch.arange(4)),
        config,
        seed=7,
        semantic_classifier=classifier,
    )

    assert any(
        not torch.equal(codec_before[name], codec_state[name])
        for name in codec_before
    )
    assert all(
        torch.equal(classifier_before[name], classifier.state_dict()[name])
        for name in classifier_before
    )
    assert all(not name.startswith("features.") for name in codec_state)
    assert metrics["semantic_loss"] is not None
