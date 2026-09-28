"""재현 가능한 VAE reconstruction trainer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models.semantic_codec import Posterior, VariationalDeepJSCC


class TrainingConfig:
    def __init__(
        self,
        epochs,
        batch_size,
        learning_rate,
        beta,
        snr_db,
        num_workers=0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.snr_db = snr_db
        self.num_workers = num_workers


class TrainingResult:
    def __init__(self, model, steps, examples, loss_trace, config):
        self.model = model
        self.steps = steps
        self.examples = examples
        self.loss_trace = loss_trace
        self.config = config


def diagonal_kl_to_standard_normal(posterior: Posterior) -> Tensor:
    """각 sample의 KL(q(z|x)||N(0,I))을 latent dimension 평균으로 반환한다."""
    return 0.5 * (
        posterior.mu.square() + posterior.logvar.exp() - 1.0 - posterior.logvar
    ).mean(dim=1)


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _epoch_loader(
    dataset: Dataset[Any], batch_size: int, seed: int, num_workers: int
) -> DataLoader[Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def batch_inputs(batch: Any) -> Tensor:
    """(image, label[, source_index]) batch에서 image만 꺼낸다."""
    if not isinstance(batch, (tuple, list)) or not batch:
        raise TypeError("dataset batch must contain an image tensor")
    inputs = batch[0]
    if not isinstance(inputs, Tensor):
        raise TypeError("dataset batch first item must be a tensor")
    return inputs


def train_reconstruction(
    model: VariationalDeepJSCC,
    dataset: Dataset[Any],
    config: TrainingConfig,
    device: torch.device,
    data_order_seed: int,
    posterior_seed: int,
    channel_seed: int,
) -> TrainingResult:
    """Posterior sample과 normalized AWGN channel을 사용해 VAE codec을 학습한다."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    posterior_generator = _device_generator(device, posterior_seed)
    channel_generator = _device_generator(device, channel_seed)
    loss_trace: list[dict[str, float]] = []
    steps = 0
    examples = 0

    for epoch in range(config.epochs):
        loader = _epoch_loader(
            dataset,
            batch_size=config.batch_size,
            seed=data_order_seed + epoch,
            num_workers=config.num_workers,
        )
        for batch in loader:
            inputs = batch_inputs(batch).to(device)
            output = model(
                inputs,
                mode="training",
                snr_db=config.snr_db,
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            reconstruction_loss = functional.mse_loss(
                output.reconstruction, inputs, reduction="mean"
            )
            kl_loss = diagonal_kl_to_standard_normal(output.posterior).mean()
            total_loss = reconstruction_loss + config.beta * kl_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_trace.append(
                {
                    "epoch": float(epoch),
                    "loss": float(total_loss.detach().cpu()),
                    "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                    "kl_loss": float(kl_loss.detach().cpu()),
                }
            )
            steps += 1
            examples += len(inputs)

    return TrainingResult(
        model=model,
        steps=steps,
        examples=examples,
        loss_trace=loss_trace,
        config=dict(vars(config)),
    )
