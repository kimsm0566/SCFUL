import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.semantic_codec import (
    BATCHMEAN_SUM_KL_REDUCTION,
    BOUNDED_LOGVAR_PARAMETERIZATION,
    CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION,
    DEEPJSCC_ARCHITECTURE,
    DEEPJSCC_CODEC_FORMAT_VERSION,
    ELEMENTWISE_MEAN_KL_REDUCTION,
    LEGACY_DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION,
    SOFTPLUS_STD_PARAMETERIZATION,
    SUPPORTED_DEEPJSCC_ARCHITECTURES,
    Posterior,
    VariationalDeepJSCC,
    VariationalDeepJSCCConfig,
    deepjscc_config_defaults_for_architecture,
    variational_deepjscc_config_for_architecture,
)
from src.utils.reconstruction import diagonal_kl_to_standard_normal
from src.utils.scu import diagonal_kl


def test_variational_deepjscc_has_published_cnn_topology_and_v3_shapes():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    inputs = torch.rand(2, 3, 32, 32)

    output = model.reconstruct_training_sample(inputs, snr_db=20.0)
    encoder_convolutions = [
        module for module in model.encoder if isinstance(module, nn.Conv2d)
    ]
    decoder_convolutions = [
        module
        for module in model.decoder
        if isinstance(module, nn.ConvTranspose2d)
    ]

    assert len(encoder_convolutions) == 5
    assert len(decoder_convolutions) == 5
    assert all(module.kernel_size == (5, 5) for module in encoder_convolutions)
    assert all(module.kernel_size == (5, 5) for module in decoder_convolutions)
    assert output.posterior.mu.shape == (2, 1024)
    assert output.posterior.logvar.shape == (2, 1024)
    assert output.transmitted_latent.shape == (2, 1024)
    assert output.reconstruction.shape == (2, 3, 32, 32)
    assert all(
        name.startswith(("encoder.", "decoder."))
        for name in model.state_dict()
    )
    assert not any(name.startswith("approximator.") for name in model.state_dict())


def test_variational_deepjscc_exposes_named_encoder_audit_features():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    features = model(torch.rand(2, 3, 32, 32), mode="encoder_features")

    assert list(features) == [
        "encoder_block_1",
        "encoder_block_2",
        "encoder_block_3",
        "encoder_block_4",
        "posterior_mu",
        "posterior_logvar",
        "canonical_latent",
    ]
    assert features["encoder_block_1"].shape == (2, 16, 16, 16)
    assert features["encoder_block_4"].shape == (2, 32, 8, 8)
    assert features["posterior_mu"].shape == (2, 1024)
    assert features["posterior_logvar"].shape == (2, 1024)
    assert features["canonical_latent"].shape == (2, 1024)


def test_variational_deepjscc_power_normalizes_each_sample():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    latent = torch.randn(3, 1024)

    normalized = model.normalized_latent(latent)

    assert torch.allclose(
        normalized.square().mean(dim=1), torch.ones(3), atol=1e-6
    )


def test_variational_deepjscc_paired_awgn_uses_the_supplied_draw():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    latent = torch.randn(2, 1024)
    standard_noise = torch.randn_like(latent)

    received = model.transmit_with_standard_awgn(latent, standard_noise, 20.0)
    expected = model.normalized_latent(latent) + 0.1 * standard_noise

    assert torch.allclose(received, expected)


def test_variational_deepjscc_sampled_paired_awgn_fixes_both_random_draws():
    config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(config)
    inputs = torch.rand(2, 3, 32, 32)
    posterior_epsilon = torch.randn(2, 64)
    standard_noise = torch.randn(2, 64)

    first = model(
        inputs,
        mode="sampled_paired_awgn",
        snr_db=20.0,
        posterior_epsilon=posterior_epsilon,
        standard_noise=standard_noise,
    )
    second = model(
        inputs,
        mode="sampled_paired_awgn",
        snr_db=20.0,
        posterior_epsilon=posterior_epsilon,
        standard_noise=standard_noise,
    )
    expected_latent = model.sample_posterior_with_epsilon(
        first.posterior, posterior_epsilon
    )
    expected_received = (
        model.normalized_latent(expected_latent) + 0.1 * standard_noise
    )

    assert torch.equal(first.posterior.mu, second.posterior.mu)
    assert torch.equal(first.posterior.logvar, second.posterior.logvar)
    assert torch.equal(first.transmitted_latent, second.transmitted_latent)
    assert torch.equal(first.reconstruction, second.reconstruction)
    assert torch.allclose(first.transmitted_latent, expected_received)


def test_variational_deepjscc_sampled_paired_awgn_validates_draws():
    model = VariationalDeepJSCC(
        VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    )
    inputs = torch.rand(2, 3, 32, 32)
    draw = torch.randn(2, 64)

    with pytest.raises(ValueError, match="requires posterior_epsilon"):
        model(
            inputs,
            mode="sampled_paired_awgn",
            snr_db=20.0,
            standard_noise=draw,
        )
    with pytest.raises(ValueError, match="posterior epsilon shape"):
        model(
            inputs,
            mode="sampled_paired_awgn",
            snr_db=20.0,
            posterior_epsilon=torch.randn(2, 63),
            standard_noise=draw,
        )


def test_variational_deepjscc_rejects_inconsistent_latent_shape():
    with pytest.raises(ValueError, match="latent_dim"):
        VariationalDeepJSCC(
            VariationalDeepJSCCConfig(
                latent_dim=1023,
                latent_channels=16,
            )
        )


def test_variational_deepjscc_rejects_non_v3_kl_reduction():
    with pytest.raises(ValueError, match="batchmean_sum"):
        VariationalDeepJSCC(
            VariationalDeepJSCCConfig(kl_reduction="sum")
        )


def test_variational_deepjscc_is_lightweight():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert trainable_parameters == 169_283
    assert trainable_parameters < 200_000


def test_variational_deepjscc_v3_uses_batchmean_latent_sum_kl():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    zero = torch.zeros(2, 5)
    unit = torch.ones(2, 5)
    reference = Posterior(mu=zero, logvar=zero)
    shifted = Posterior(mu=unit, logvar=zero)

    assert model.prior_kl(shifted).item() == pytest.approx(2.5)
    assert model.posterior_kl_to(shifted, reference).item() == pytest.approx(2.5)
    assert diagonal_kl_to_standard_normal(
        shifted
    ).mean().item() == pytest.approx(0.5)
    assert diagonal_kl(shifted, reference).mean().item() == pytest.approx(0.5)


def test_variational_deepjscc_v2_preserves_elementwise_mean_kl():
    config = variational_deepjscc_config_for_architecture(
        LEGACY_DEEPJSCC_ARCHITECTURE
    )
    model = VariationalDeepJSCC(config)
    zero = torch.zeros(2, 5)
    unit = torch.ones(2, 5)
    reference = Posterior(mu=zero, logvar=zero)
    shifted = Posterior(mu=unit, logvar=zero)

    assert model.prior_kl(shifted).item() == pytest.approx(0.5)
    assert model.posterior_kl_to(shifted, reference).item() == pytest.approx(0.5)


def test_variational_deepjscc_v3_initializes_target_posterior_std():
    torch.manual_seed(0)
    config = VariationalDeepJSCCConfig(initial_logvar=-7.0)
    model = VariationalDeepJSCC(config)
    inputs = torch.rand(4, 3, 32, 32)
    posterior_layer = model.encoder[-1]
    rho_weight = posterior_layer.weight[config.latent_channels :]
    rho_bias = posterior_layer.bias[config.latent_channels :]

    posterior = model(inputs, mode="posterior")

    assert torch.count_nonzero(rho_weight).item() == 0
    assert torch.all(rho_bias == rho_bias[0])
    assert torch.allclose(
        posterior.logvar,
        torch.full_like(posterior.logvar, config.initial_logvar),
        atol=1e-6,
    )
    assert posterior.logvar.mul(0.5).exp().mean().item() == pytest.approx(
        0.030197, abs=1e-6
    )


def test_variational_deepjscc_softplus_std_has_no_silent_logvar_clamp():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    rho = torch.tensor([-20.0, 20.0], requires_grad=True)

    logvar = model.softplus_logvar(rho)
    expected = 2.0 * (F.softplus(rho) + model.config.posterior_std_min).log()
    logvar.sum().backward()

    assert torch.equal(logvar, expected)
    assert logvar[0].item() < -8.0
    assert logvar[1].item() > 4.0
    assert torch.all(rho.grad > 0)


def test_variational_deepjscc_v2_bounded_logvar_keeps_gradients():
    config = variational_deepjscc_config_for_architecture(
        LEGACY_DEEPJSCC_ARCHITECTURE
    )
    model = VariationalDeepJSCC(config)
    raw_logvar = torch.tensor([-12.0, 12.0], requires_grad=True)

    bounded = model.bounded_logvar(raw_logvar)
    bounded.sum().backward()

    assert torch.all(bounded > model.config.logvar_min)
    assert torch.all(bounded < model.config.logvar_max)
    assert torch.all(raw_logvar.grad > 0)


def test_variational_deepjscc_distillation_uses_same_posterior_mapping():
    model = VariationalDeepJSCC(VariationalDeepJSCCConfig())
    inputs = torch.rand(2, 3, 32, 32)
    standard_noise = torch.zeros(2, 1024)

    direct = model(inputs, mode="posterior")
    distillation = model(
        inputs,
        mode="distillation",
        snr_db=20.0,
        standard_noise=standard_noise,
    ).posterior

    assert torch.equal(direct.mu, distillation.mu)
    assert torch.equal(direct.logvar, distillation.logvar)


def test_variational_deepjscc_rejects_invalid_v3_posterior_config():
    target_std = torch.exp(torch.tensor(-7.0 / 2.0)).item()
    cases = [
        ({"posterior_std_min": -1e-6}, "posterior_std_min"),
        ({"posterior_std_min": target_std}, "greater than posterior_std_min"),
        ({"logvar_min": -8.0}, "must both be None"),
        ({"logvar_max": 4.0}, "must both be None"),
        ({"logvar_parameterization": "hard_clamp"}, "softplus_std_v1"),
        ({"codec_format_version": 1}, "format is incompatible"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            VariationalDeepJSCC(VariationalDeepJSCCConfig(**overrides))


def test_variational_deepjscc_rejects_invalid_v2_posterior_config():
    cases = [
        ({"logvar_min": -8.0, "logvar_max": -8.0}, "logvar_min"),
        ({"initial_logvar": -8.0}, "initial_logvar"),
        ({"epsilon_power": 0.0}, "epsilon_power"),
        ({"logvar_parameterization": "hard_clamp"}, "sigmoid_bounded_v1"),
        ({"kl_reduction": "sum"}, "elementwise_mean"),
    ]
    for overrides, message in cases:
        config = variational_deepjscc_config_for_architecture(
            LEGACY_DEEPJSCC_ARCHITECTURE,
            **overrides,
        )
        with pytest.raises(ValueError, match=message):
            VariationalDeepJSCC(config)


def test_variational_deepjscc_declares_v3_and_explicit_v2_defaults():
    current = deepjscc_config_defaults_for_architecture(DEEPJSCC_ARCHITECTURE)
    legacy = deepjscc_config_defaults_for_architecture(
        LEGACY_DEEPJSCC_ARCHITECTURE
    )

    assert DEEPJSCC_ARCHITECTURE == "variational_deepjscc_cifar10_v3"
    assert LEGACY_DEEPJSCC_ARCHITECTURE == "variational_deepjscc_cifar10_v2"
    assert SUPPORTED_DEEPJSCC_ARCHITECTURES == (
        DEEPJSCC_ARCHITECTURE,
        LEGACY_DEEPJSCC_ARCHITECTURE,
    )
    assert DEEPJSCC_CODEC_FORMAT_VERSION == 3
    assert CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION == 3
    assert LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION == 2
    assert current["latent_channels"] == 16
    assert current["latent_dim"] == 1024
    assert "logvar_min" not in current
    assert "logvar_max" not in current
    assert current["logvar_parameterization"] == SOFTPLUS_STD_PARAMETERIZATION
    assert current["kl_reduction"] == BATCHMEAN_SUM_KL_REDUCTION
    assert legacy["latent_channels"] == 4
    assert legacy["latent_dim"] == 256
    assert legacy["logvar_parameterization"] == BOUNDED_LOGVAR_PARAMETERIZATION
    assert legacy["kl_reduction"] == ELEMENTWISE_MEAN_KL_REDUCTION


def test_variational_deepjscc_v2_checkpoint_state_dict_still_loads_strictly():
    config = variational_deepjscc_config_for_architecture(
        LEGACY_DEEPJSCC_ARCHITECTURE
    )
    source = VariationalDeepJSCC(config)
    restored = VariationalDeepJSCC(
        variational_deepjscc_config_for_architecture(
            LEGACY_DEEPJSCC_ARCHITECTURE
        )
    )

    restored.load_state_dict(source.state_dict(), strict=True)

    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_variational_deepjscc_rejects_unknown_architecture_defaults():
    with pytest.raises(ValueError, match="unsupported"):
        deepjscc_config_defaults_for_architecture("unknown")
