"""Independent raw-image CIFAR-10 classification model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


CIFAR10_CLASSIFIER_ARCHITECTURE = "cifar10_groupnorm_silu_cnn"
CIFAR10_CHANNEL_MEAN = (0.4914008, 0.4821590, 0.4465309)
CIFAR10_CHANNEL_STD = (0.2470322, 0.2434851, 0.2615879)


class Cifar10ClassifierConfig:
    """Configuration for the standalone classifier."""

    def __init__(
        self,
        base_channels=32,
        group_count=8,
        num_classes=10,
        channel_mean=CIFAR10_CHANNEL_MEAN,
        channel_std=CIFAR10_CHANNEL_STD,
    ):
        self.base_channels = int(base_channels)
        self.group_count = int(group_count)
        self.num_classes = int(num_classes)
        self.channel_mean = tuple(float(value) for value in channel_mean)
        self.channel_std = tuple(float(value) for value in channel_std)


class Cifar10Classifier(nn.Module):
    """Standalone GroupNorm CNN with fixed CIFAR-10 input normalization."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Cifar10ClassifierConfig()
        if self.config.base_channels <= 0:
            raise ValueError("classifier base_channels must be positive")
        if self.config.group_count <= 0:
            raise ValueError("classifier group_count must be positive")
        if self.config.num_classes <= 0:
            raise ValueError("classifier num_classes must be positive")
        if len(self.config.channel_mean) != 3 or len(self.config.channel_std) != 3:
            raise ValueError("classifier normalization requires three RGB values")
        if any(value <= 0 for value in self.config.channel_std):
            raise ValueError("classifier channel standard deviations must be positive")

        stage_channels = (
            self.config.base_channels,
            2 * self.config.base_channels,
            4 * self.config.base_channels,
        )
        if any(
            channels % self.config.group_count != 0
            for channels in stage_channels
        ):
            raise ValueError(
                "classifier group_count must divide every feature width"
            )

        self.register_buffer(
            "input_mean",
            torch.tensor(self.config.channel_mean, dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(self.config.channel_std, dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        first, second, third = stage_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, first),
            nn.SiLU(),
            nn.Conv2d(first, first, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, first),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first, second, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, second),
            nn.SiLU(),
            nn.Conv2d(second, second, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, second),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, third),
            nn.SiLU(),
            nn.Conv2d(third, third, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.config.group_count, third),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output = nn.Linear(third, self.config.num_classes)

    def normalize(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
            raise ValueError("CIFAR-10 classifier requires (B, 3, 32, 32) images")
        if not torch.is_floating_point(inputs):
            raise TypeError("CIFAR-10 classifier inputs must be floating point")
        return (inputs - self.input_mean) / self.input_std

    def freeze(self):
        """Freeze weights while preserving gradients with respect to the input."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def forward(self, inputs, mode="logits"):
        """Return logits by default or the requested penultimate representation."""

        features = self.features(self.normalize(inputs)).flatten(start_dim=1)
        if mode == "features":
            return features
        if mode == "normalized_features":
            normalized = functional.normalize(
                features, p=2.0, dim=1, eps=1e-8
            )
            return (features.shape[1] ** 0.5) * normalized
        if mode == "logits":
            return self.output(features)
        raise ValueError(
            "classifier mode must be logits, features, or normalized_features"
        )
