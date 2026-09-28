"""Whole target client용 SCU-style centralized VAE adaptation."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset

from src.models.semantic_codec import Posterior, VariationalDeepJSCC
from src.utils.reconstruction import (
    _device_generator,
    _epoch_loader,
    batch_inputs,
    diagonal_kl_to_standard_normal,
)


SCUMode = Literal["forget_only", "self_sharing"]


class SCUConfig:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        lambda_kl: float,
        lambda_rec: float,
        lambda_ss: float,
        beta: float,
        epsilon: float,
        snr_db: float,
        num_workers: int = 0,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_kl = lambda_kl
        self.lambda_rec = lambda_rec
        self.lambda_ss = lambda_ss
        self.beta = beta
        self.epsilon = epsilon
        self.snr_db = snr_db
        self.num_workers = num_workers


class AdaptationResult:
    def __init__(
        self,
        model: VariationalDeepJSCC,
        steps: int,
        target_examples: int,
        retained_examples: int,
        loss_trace: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.steps = steps
        self.target_examples = target_examples
        self.retained_examples = retained_examples
        self.loss_trace = loss_trace
        self.config = config


def diagonal_kl(posterior_p: Posterior, posterior_q: Posterior) -> Tensor:
    """각 sample의 KL(q_p || q_q)을 latent dimension 평균으로 계산한다."""
    return 0.5 * (
        posterior_q.logvar
        - posterior_p.logvar
        + (
            posterior_p.logvar.exp()
            + (posterior_p.mu - posterior_q.mu).square()
        )
        / posterior_q.logvar.exp()
        - 1.0
    ).mean(dim=1)


def _freeze_teacher(model: VariationalDeepJSCC, device: torch.device) -> VariationalDeepJSCC:
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def adapt_scu_style(
    original: VariationalDeepJSCC,
    target_dataset: Dataset[Any],
    retained_dataset: Dataset[Any],
    config: SCUConfig,
    device: torch.device,
    mode: SCUMode,
    seeds: dict[str, int],
) -> AdaptationResult:
    """`L_forget` 또는 명세의 dynamic self-sharing loss로 original을 갱신한다."""
    if mode not in ("forget_only", "self_sharing"):
        raise ValueError(f"unsupported SCU mode: {mode}")

    teacher = _freeze_teacher(original, device)
    learner = copy.deepcopy(original).to(device)
    learner.train()
    optimizer = torch.optim.Adam(learner.parameters(), lr=config.learning_rate)
    posterior_generator = _device_generator(device, seeds["posterior"])
    channel_generator = _device_generator(device, seeds["channel"])

    steps = 0
    target_examples = 0
    retained_examples = 0
    loss_trace: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        target_loader = _epoch_loader(
            target_dataset,
            batch_size=config.batch_size,
            seed=seeds["data_order"] + 2 * epoch,
            num_workers=config.num_workers,
        )
        retained_loader = _epoch_loader(
            retained_dataset,
            batch_size=config.batch_size,
            seed=seeds["data_order"] + 2 * epoch + 1,
            num_workers=config.num_workers,
        )
        retained_iterator = iter(retained_loader)

        for target_batch in target_loader:
            try:
                retained_batch = next(retained_iterator)
            except StopIteration:
                retained_loader = _epoch_loader(
                    retained_dataset,
                    batch_size=config.batch_size,
                    seed=seeds["data_order"] + 20_000 + epoch + steps,
                    num_workers=config.num_workers,
                )
                retained_iterator = iter(retained_loader)
                retained_batch = next(retained_iterator)

            target_inputs = batch_inputs(target_batch).to(device)
            retained_inputs = batch_inputs(retained_batch).to(device)
            with torch.no_grad():
                teacher_target = teacher(target_inputs, mode="posterior")

            target_output = learner(
                target_inputs,
                mode="training",
                snr_db=config.snr_db,
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            forget_kl = diagonal_kl(target_output.posterior, teacher_target).mean()
            target_mse = functional.mse_loss(
                target_output.reconstruction, target_inputs, reduction="mean"
            )
            forget_loss = config.lambda_kl * forget_kl - config.lambda_rec * target_mse

            optimized_terms = ["forget_loss"]
            retain_loss = None
            if mode == "forget_only":
                total_loss = forget_loss
                weight_forget = torch.ones((), device=device)
                weight_retain = torch.zeros((), device=device)
            else:
                retained_output = learner(
                    retained_inputs,
                    mode="training",
                    snr_db=config.snr_db,
                    posterior_generator=posterior_generator,
                    channel_generator=channel_generator,
                )
                retained_mse = functional.mse_loss(
                    retained_output.reconstruction, retained_inputs, reduction="mean"
                )
                retained_kl = diagonal_kl_to_standard_normal(
                    retained_output.posterior
                ).mean()
                retain_loss = config.beta * retained_kl + retained_mse
                denominator = forget_loss.detach().abs() + retain_loss.detach().abs() + config.epsilon
                weight_forget = forget_loss.detach().abs() / denominator
                weight_retain = retain_loss.detach().abs() / denominator
                total_loss = (
                    weight_forget * forget_loss
                    + weight_retain * config.lambda_ss * retain_loss
                )
                optimized_terms = ["forget_loss", "retain_loss"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
            optimizer.step()

            trace: dict[str, Any] = {
                "epoch": epoch,
                "mode": mode,
                "optimized_terms": optimized_terms,
                "loss": float(total_loss.detach().cpu()),
                "forget_loss": float(forget_loss.detach().cpu()),
                "forget_kl_q_u_to_q_g": float(forget_kl.detach().cpu()),
                "target_mse": float(target_mse.detach().cpu()),
                "weight_forget": float(weight_forget.detach().cpu()),
                "weight_retain": float(weight_retain.detach().cpu()),
            }
            if retain_loss is not None:
                trace["retain_loss"] = float(retain_loss.detach().cpu())
                trace["retained_mse"] = float(retained_mse.detach().cpu())
                trace["retained_kl_to_prior"] = float(retained_kl.detach().cpu())
            loss_trace.append(trace)
            steps += 1
            target_examples += len(target_inputs)
            retained_examples += len(retained_inputs)

    return AdaptationResult(
        model=learner,
        steps=steps,
        target_examples=target_examples,
        retained_examples=retained_examples,
        loss_trace=loss_trace,
        config=dict(vars(config)),
    )
