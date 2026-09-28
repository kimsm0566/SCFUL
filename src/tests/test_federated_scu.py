import torch

from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.utils.fscu import (
    extract_erased_representation_buffer,
    forget_target_client,
    train_server_public_contrastive_compensation,
)


def test_target_joint_unlearning_changes_encoder_and_decoder():
    codec_config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(codec_config)
    original = VariationalDeepJSCC(codec_config)
    original.load_state_dict(model.state_dict())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    config = {
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 1e-3,
        "snr_db": 10.0,
        "lambda_kl": 1.0,
        "lambda_rec": 0.1,
        "semantic_weight": 0.0,
        "device": "cpu",
    }
    state, metrics = forget_target_client(
        model,
        (torch.rand(8, 3, 32, 32), torch.zeros(8, dtype=torch.long)),
        original,
        config,
        seed=3,
    )
    assert not torch.equal(before["encoder.0.weight"], state["encoder.0.weight"])
    assert not torch.equal(
        before["decoder.0.weight"], state["decoder.0.weight"]
    )
    assert metrics["steps"] == 2


def test_ib_joint_unlearning_reports_target_prior_kl_ratio():
    codec_config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(codec_config)
    original = VariationalDeepJSCC(codec_config)
    original.load_state_dict(model.state_dict())
    config = {
        "batch_size": 4,
        "epochs": 2,
        "learning_rate": 1e-3,
        "snr_db": 10.0,
        "lambda_kl": 1.0,
        "lambda_rec": 0.1,
        "forgetting_objective": "prior_kl_collapse",
        "target_information_ratio_threshold": 0.99,
        "semantic_weight": 0.0,
        "device": "cpu",
    }

    _, metrics = forget_target_client(
        model,
        (torch.rand(8, 3, 32, 32), torch.zeros(8, dtype=torch.long)),
        original,
        config,
        seed=7,
    )

    assert metrics["forgetting_objective"] == "prior_kl_collapse"
    assert metrics["epochs_completed"] <= 2
    assert metrics["original_target_information"]["mean_prior_kl"] > 0.0
    assert metrics["final_target_information"]["mean_ratio"] > 0.0
    assert "target_prior_kl" in metrics["history"][0]


def test_public_compensation_updates_full_codec_with_detached_buffer():
    codec_config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(codec_config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    erased_data = (
        torch.rand(8, 3, 32, 32),
        torch.zeros(8, dtype=torch.long),
    )
    erased_buffer = extract_erased_representation_buffer(
        model,
        erased_data,
        batch_size=4,
        device="cpu",
        max_examples=6,
        seed=9,
        sampling_mode="stratified",
    )
    config = {
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 1e-3,
        "temperature": 0.2,
        "lambda_contrastive": 1.0,
        "lambda_public_reconstruction": 1.0,
        "lambda_public_prior_kl": 0.0,
        "public_positive_mode": "deterministic_augmentation",
        "augmentation_crop_padding": 4,
        "augmentation_horizontal_flip_probability": 0.5,
        "snr_db": 10.0,
        "semantic_weight": 0.0,
        "device": "cpu",
    }
    state, metrics = train_server_public_contrastive_compensation(
        model,
        (torch.rand(8, 3, 32, 32), torch.ones(8, dtype=torch.long)),
        erased_buffer,
        config,
        seed=17,
    )
    assert tuple(erased_buffer.shape) == (6, 64)
    assert not erased_buffer.requires_grad
    assert torch.allclose(
        erased_buffer.norm(dim=1), torch.full((6,), 8.0), atol=1e-5
    )
    assert not torch.equal(before["encoder.0.weight"], state["encoder.0.weight"])
    assert not torch.equal(
        before["decoder.0.weight"], state["decoder.0.weight"]
    )
    assert metrics["erased_buffer_bytes"] == 6 * 64 * 4
    assert metrics["steps"] == 2
    assert metrics["public_positive_mode"] == "deterministic_augmentation"
    assert "posterior_logvar_mean" in metrics
    assert "posterior_entropy_mean" in metrics
    assert "pre_posterior_logvar_mean" in metrics
    assert "post_posterior_logvar_mean" in metrics
