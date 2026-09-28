"""Variational DeepJSCC codec used by every CIFAR-10 experiment."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


DEEPJSCC_ARCHITECTURE = "variational_deepjscc_cifar10_v3"
LEGACY_DEEPJSCC_ARCHITECTURE = "variational_deepjscc_cifar10_v2"
SUPPORTED_DEEPJSCC_ARCHITECTURES = (
    DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_ARCHITECTURE,
)
CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION = 3
LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION = 2
DEEPJSCC_CODEC_FORMAT_VERSION = CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION
SUPPORTED_DEEPJSCC_CODEC_FORMAT_VERSIONS = (
    CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION,
    LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION,
)
SOFTPLUS_STD_PARAMETERIZATION = "softplus_std_v1"
BOUNDED_LOGVAR_PARAMETERIZATION = "sigmoid_bounded_v1"
BATCHMEAN_SUM_KL_REDUCTION = "batchmean_sum"
ELEMENTWISE_MEAN_KL_REDUCTION = "elementwise_mean"


class Posterior:
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar


class CodecOutput:
    def __init__(self, reconstruction, posterior, transmitted_latent):
        self.reconstruction = reconstruction
        self.posterior = posterior
        self.transmitted_latent = transmitted_latent


class DistillationOutput:
    def __init__(
        self,
        reconstruction,
        posterior,
        transmitted_latent,
        encoder_shallow,
        decoder_shallow,
    ):
        self.reconstruction = reconstruction
        self.posterior = posterior
        self.transmitted_latent = transmitted_latent
        self.encoder_shallow = encoder_shallow
        self.decoder_shallow = decoder_shallow


class VariationalDeepJSCCConfig:
    """Configuration for the reconstruction-only CIFAR-10 DeepJSCC codec."""

    def __init__(
        self,
        latent_dim=8 * 8 * 16,
        latent_channels=16,
        logvar_min=None,
        logvar_max=None,
        initial_logvar=-7.0,
        posterior_std_min=1e-6,
        logvar_parameterization=SOFTPLUS_STD_PARAMETERIZATION,
        codec_format_version=DEEPJSCC_CODEC_FORMAT_VERSION,
        epsilon_power=1e-8,
        kl_reduction=BATCHMEAN_SUM_KL_REDUCTION,
    ):
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.initial_logvar = initial_logvar
        self.posterior_std_min = posterior_std_min
        self.logvar_parameterization = logvar_parameterization
        self.codec_format_version = codec_format_version
        self.epsilon_power = epsilon_power
        self.kl_reduction = kl_reduction


def deepjscc_config_defaults_for_architecture(architecture):
    """Return codec defaults for a registered checkpoint architecture."""

    common = {
        "initial_logvar": -7.0,
        "posterior_std_min": 1e-6,
        "epsilon_power": 1e-8,
    }
    if architecture == DEEPJSCC_ARCHITECTURE:
        return {
            **common,
            "latent_dim": 8 * 8 * 16,
            "latent_channels": 16,
            "logvar_parameterization": SOFTPLUS_STD_PARAMETERIZATION,
            "codec_format_version": CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION,
            "kl_reduction": BATCHMEAN_SUM_KL_REDUCTION,
        }
    if architecture == LEGACY_DEEPJSCC_ARCHITECTURE:
        return {
            **common,
            "latent_dim": 8 * 8 * 4,
            "latent_channels": 4,
            "logvar_min": -8.0,
            "logvar_max": 4.0,
            "logvar_parameterization": BOUNDED_LOGVAR_PARAMETERIZATION,
            "codec_format_version": LEGACY_DEEPJSCC_CODEC_FORMAT_VERSION,
            "kl_reduction": ELEMENTWISE_MEAN_KL_REDUCTION,
        }
    raise ValueError(
        "unsupported variational DeepJSCC architecture: "
        f"{architecture}; expected one of {SUPPORTED_DEEPJSCC_ARCHITECTURES}"
    )


def variational_deepjscc_config_for_architecture(architecture, **overrides):
    """Build a config with version-correct defaults and explicit overrides."""

    values = deepjscc_config_defaults_for_architecture(architecture)
    values.update(overrides)
    return VariationalDeepJSCCConfig(**values)


class VariationalDeepJSCC(nn.Module):
    """Original CIFAR-10 DeepJSCC path with a Gaussian posterior head.

    The encoder has the five 5x5 convolutional stages used by the original
    CIFAR-10 DeepJSCC topology. Its final stage emits posterior mean and scale
    parameter maps. A sampled latent is power-normalized, transmitted over
    AWGN, and decoded by the mirrored transposed-convolutional path.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or VariationalDeepJSCCConfig()
        expected_latent_dim = 8 * 8 * self.config.latent_channels
        if self.config.latent_dim != expected_latent_dim:
            raise ValueError(
                "variational DeepJSCC latent_dim must equal "
                f"8*8*latent_channels ({expected_latent_dim})"
            )
        if self.config.latent_channels <= 0:
            raise ValueError("variational DeepJSCC latent_channels must be positive")
        if not math.isfinite(float(self.config.initial_logvar)):
            raise ValueError("variational DeepJSCC initial_logvar must be finite")
        if self.config.codec_format_version not in (
            SUPPORTED_DEEPJSCC_CODEC_FORMAT_VERSIONS
        ):
            raise ValueError(
                "variational DeepJSCC checkpoint/config format is incompatible: "
                "expected a supported version in "
                f"{SUPPORTED_DEEPJSCC_CODEC_FORMAT_VERSIONS}, received "
                f"{self.config.codec_format_version}"
            )
        if not math.isfinite(float(self.config.epsilon_power)) or (
            self.config.epsilon_power <= 0
        ):
            raise ValueError("variational DeepJSCC epsilon_power must be positive")
        self._validate_posterior_configuration()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.Conv2d(
                32,
                2 * self.config.latent_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.config.latent_channels,
                32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.PReLU(32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.PReLU(16),
            nn.ConvTranspose2d(
                16,
                3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )
        self._initialize_posterior_scale_channels()

    def _validate_posterior_configuration(self):
        if (
            self.config.codec_format_version
            == CURRENT_DEEPJSCC_CODEC_FORMAT_VERSION
        ):
            if (
                self.config.logvar_min is not None
                or self.config.logvar_max is not None
            ):
                raise ValueError(
                    "variational DeepJSCC v3 logvar_min and logvar_max must "
                    "both be None"
                )
            if self.config.logvar_parameterization != SOFTPLUS_STD_PARAMETERIZATION:
                raise ValueError(
                    "variational DeepJSCC v3 requires softplus_std_v1 "
                    "posterior standard deviation"
                )
            posterior_std_min = float(self.config.posterior_std_min)
            if not math.isfinite(posterior_std_min) or posterior_std_min < 0.0:
                raise ValueError(
                    "variational DeepJSCC posterior_std_min must be finite and "
                    "non-negative"
                )
            try:
                target_std = math.exp(0.5 * self.config.initial_logvar)
            except OverflowError as error:
                raise ValueError(
                    "variational DeepJSCC initial posterior standard deviation "
                    "must be finite"
                ) from error
            if not math.isfinite(target_std) or target_std <= posterior_std_min:
                raise ValueError(
                    "variational DeepJSCC initial posterior standard deviation "
                    "must be finite and greater than posterior_std_min"
                )
            if self.config.kl_reduction != BATCHMEAN_SUM_KL_REDUCTION:
                raise ValueError(
                    "variational DeepJSCC v3 requires batchmean_sum KL"
                )
            return

        if self.config.logvar_parameterization != BOUNDED_LOGVAR_PARAMETERIZATION:
            raise ValueError(
                "variational DeepJSCC v2 requires sigmoid_bounded_v1 log-variance"
            )
        for name in ("logvar_min", "logvar_max"):
            value = float(getattr(self.config, name))
            if not math.isfinite(value):
                raise ValueError(f"variational DeepJSCC {name} must be finite")
        if not self.config.logvar_min < self.config.logvar_max:
            raise ValueError(
                "variational DeepJSCC logvar_min must be less than logvar_max"
            )
        initial_logvar_is_bounded = (
            self.config.logvar_min
            < self.config.initial_logvar
            < self.config.logvar_max
        )
        if not initial_logvar_is_bounded:
            raise ValueError(
                "variational DeepJSCC initial_logvar must lie strictly inside "
                "the configured bounds"
            )
        if self.config.kl_reduction != ELEMENTWISE_MEAN_KL_REDUCTION:
            raise ValueError(
                "variational DeepJSCC v2 requires elementwise_mean KL"
            )

    def _raw_logvar_for_value(self, value):
        fraction = (value - self.config.logvar_min) / (
            self.config.logvar_max - self.config.logvar_min
        )
        return math.log(fraction / (1.0 - fraction))

    def _inverse_softplus(self, value):
        return value + math.log(-math.expm1(-value))

    def _initialize_posterior_scale_channels(self):
        posterior_layer = self.encoder[-1]
        first_logvar_channel = self.config.latent_channels
        if self.config.logvar_parameterization == SOFTPLUS_STD_PARAMETERIZATION:
            target_std = math.exp(0.5 * self.config.initial_logvar)
            raw_scale_bias = self._inverse_softplus(
                target_std - self.config.posterior_std_min
            )
        else:
            raw_scale_bias = self._raw_logvar_for_value(
                self.config.initial_logvar
            )
        with torch.no_grad():
            posterior_layer.weight[first_logvar_channel:].zero_()
            posterior_layer.bias[first_logvar_channel:].fill_(raw_scale_bias)

    def bounded_logvar(self, raw_logvar):
        span = self.config.logvar_max - self.config.logvar_min
        return self.config.logvar_min + span * torch.sigmoid(raw_logvar)

    def softplus_logvar(self, rho):
        posterior_std = F.softplus(rho) + self.config.posterior_std_min
        return 2.0 * posterior_std.log()

    def posterior_from_encoded(self, encoded):
        mu_map, raw_scale_map = encoded.chunk(2, dim=1)
        if self.config.logvar_parameterization == SOFTPLUS_STD_PARAMETERIZATION:
            logvar_map = self.softplus_logvar(raw_scale_map)
        else:
            logvar_map = self.bounded_logvar(raw_scale_map)
        return Posterior(
            mu=mu_map.flatten(start_dim=1),
            logvar=logvar_map.flatten(start_dim=1),
        )

    def posterior(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
            raise ValueError("variational DeepJSCC requires (B, 3, 32, 32) images")
        return self.posterior_from_encoded(self.encoder(inputs))

    def normalized_latent(self, latent):
        norm = latent.norm(dim=1, keepdim=True)
        return (self.config.latent_dim**0.5) * latent / (
            norm + self.config.epsilon_power
        )

    def canonical_latent_from_posterior(self, posterior):
        return self.normalized_latent(posterior.mu)

    def canonical_latent(self, inputs):
        return self.canonical_latent_from_posterior(self.posterior(inputs))

    def sample_posterior(self, posterior, generator=None):
        if generator is None:
            epsilon = torch.randn_like(posterior.mu)
        else:
            epsilon = torch.randn(
                posterior.mu.shape,
                dtype=posterior.mu.dtype,
                device=posterior.mu.device,
                generator=generator,
            )
        return self.sample_posterior_with_epsilon(posterior, epsilon)

    def sample_posterior_with_epsilon(self, posterior, posterior_epsilon):
        if posterior_epsilon.shape != posterior.mu.shape:
            raise ValueError(
                "posterior epsilon shape must match the posterior latent shape"
            )
        epsilon = posterior_epsilon.to(
            device=posterior.mu.device, dtype=posterior.mu.dtype
        )
        return posterior.mu + posterior.logvar.mul(0.5).exp() * epsilon

    def awgn_std(self, snr_db):
        return 10.0 ** (-snr_db / 20.0)

    def prior_kl(self, posterior):
        elementwise_kl = 0.5 * (
            posterior.mu.square()
            + posterior.logvar.exp()
            - 1.0
            - posterior.logvar
        )
        return self._reduce_kl(elementwise_kl)

    def posterior_kl_to(self, current, reference):
        elementwise_kl = 0.5 * (
            reference.logvar
            - current.logvar
            + (current.logvar.exp() + (current.mu - reference.mu).square())
            / reference.logvar.exp()
            - 1.0
        )
        return self._reduce_kl(elementwise_kl)

    def _reduce_kl(self, elementwise_kl):
        if self.config.kl_reduction == BATCHMEAN_SUM_KL_REDUCTION:
            return elementwise_kl.flatten(start_dim=1).sum(dim=1).mean()
        return elementwise_kl.mean()

    def decode_latent(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError(
                "variational DeepJSCC latent must be shaped "
                f"(B, {self.config.latent_dim})"
            )
        return self.decoder(
            latent.view(-1, self.config.latent_channels, 8, 8)
        )

    def transmit_with_standard_awgn(self, latent, standard_noise, snr_db):
        transmitted = self.normalized_latent(latent)
        if standard_noise.shape != transmitted.shape:
            raise ValueError("standard AWGN shape must match the latent shape")
        return transmitted + standard_noise.to(
            device=transmitted.device, dtype=transmitted.dtype
        ) * self.awgn_std(snr_db)

    def decode_with_standard_awgn(self, inputs, standard_noise, snr_db):
        posterior = self.posterior(inputs)
        received = self.transmit_with_standard_awgn(
            posterior.mu, standard_noise, snr_db
        )
        return CodecOutput(self.decode_latent(received), posterior, received)

    def decode_sampled_with_standard_awgn(
        self,
        inputs,
        posterior_epsilon,
        standard_noise,
        snr_db,
    ):
        posterior = self.posterior(inputs)
        latent = self.sample_posterior_with_epsilon(
            posterior, posterior_epsilon
        )
        received = self.transmit_with_standard_awgn(
            latent, standard_noise, snr_db
        )
        return CodecOutput(self.decode_latent(received), posterior, received)

    def decode_posterior_mean(self, inputs):
        posterior = self.posterior(inputs)
        transmitted = self.normalized_latent(posterior.mu)
        return CodecOutput(self.decode_latent(transmitted), posterior, transmitted)

    def decode_with_awgn(self, inputs, noise):
        posterior = self.posterior(inputs)
        transmitted = self.normalized_latent(posterior.mu)
        if noise.shape != transmitted.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} does not match "
                f"latent shape {tuple(transmitted.shape)}"
            )
        received = transmitted + noise.to(
            device=transmitted.device, dtype=transmitted.dtype
        )
        return CodecOutput(self.decode_latent(received), posterior, received)

    def reconstruct_training_sample(
        self,
        inputs,
        snr_db,
        posterior_generator=None,
        channel_generator=None,
    ):
        posterior = self.posterior(inputs)
        latent = self.sample_posterior(posterior, posterior_generator)
        transmitted = self.normalized_latent(latent)
        noise = torch.randn(
            transmitted.shape,
            dtype=transmitted.dtype,
            device=transmitted.device,
            generator=channel_generator,
        ) * self.awgn_std(snr_db)
        received = transmitted + noise
        return CodecOutput(self.decode_latent(received), posterior, received)

    def distillation_forward(self, inputs, standard_noise, snr_db):
        encoded = inputs
        encoder_shallow = None
        for index, layer in enumerate(self.encoder):
            encoded = layer(encoded)
            if index == 1:
                encoder_shallow = encoded
        posterior = self.posterior_from_encoded(encoded)
        received = self.transmit_with_standard_awgn(
            posterior.mu, standard_noise, snr_db
        )
        decoded = received.view(-1, self.config.latent_channels, 8, 8)
        decoder_shallow = None
        for index, layer in enumerate(self.decoder):
            decoded = layer(decoded)
            if index == 7:
                decoder_shallow = decoded
        return DistillationOutput(
            decoded,
            posterior,
            received,
            encoder_shallow,
            decoder_shallow,
        )

    def forward(
        self,
        inputs,
        mode="training",
        snr_db=None,
        posterior_generator=None,
        channel_generator=None,
        standard_noise=None,
        noise=None,
        posterior_epsilon=None,
    ):
        if mode == "training":
            if snr_db is None:
                raise ValueError("training mode requires snr_db")
            return self.reconstruct_training_sample(
                inputs, snr_db, posterior_generator, channel_generator
            )
        if mode == "posterior":
            return self.posterior(inputs)
        if mode == "canonical_latent":
            return self.canonical_latent(inputs)
        if mode == "mean_reconstruction":
            return self.decode_posterior_mean(inputs)
        if mode == "paired_awgn":
            if standard_noise is None or snr_db is None:
                raise ValueError("paired_awgn mode requires standard_noise and snr_db")
            return self.decode_with_standard_awgn(inputs, standard_noise, snr_db)
        if mode == "sampled_paired_awgn":
            if (
                posterior_epsilon is None
                or standard_noise is None
                or snr_db is None
            ):
                raise ValueError(
                    "sampled_paired_awgn mode requires posterior_epsilon, "
                    "standard_noise, and snr_db"
                )
            return self.decode_sampled_with_standard_awgn(
                inputs,
                posterior_epsilon,
                standard_noise,
                snr_db,
            )
        if mode == "fixed_awgn":
            if noise is None:
                raise ValueError("fixed_awgn mode requires noise")
            return self.decode_with_awgn(inputs, noise)
        if mode == "distillation":
            if standard_noise is None or snr_db is None:
                raise ValueError("distillation mode requires standard_noise and snr_db")
            return self.distillation_forward(inputs, standard_noise, snr_db)
        raise ValueError(f"unsupported codec forward mode: {mode}")
