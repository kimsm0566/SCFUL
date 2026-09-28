"""Retraining oracle에 대한 reconstruction·posterior functional gap metric."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from torch import Tensor

from src.models.semantic_codec import Posterior


def diagonal_kl(
    posterior_p: Posterior,
    posterior_q: Posterior,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
) -> Tensor:
    """각 sample의 KL(q_p || q_q)을 latent dimension 전체에 합산한다."""
    p_logvar = posterior_p.logvar.clamp(logvar_min, logvar_max)
    q_logvar = posterior_q.logvar.clamp(logvar_min, logvar_max)
    return 0.5 * (
        q_logvar
        - p_logvar
        + (p_logvar.exp() + (posterior_p.mu - posterior_q.mu).square())
        / q_logvar.exp()
        - 1.0
    ).sum(dim=1)


def symmetric_diagonal_kl(
    posterior_p: Posterior,
    posterior_q: Posterior,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
) -> Tensor:
    """두 diagonal Gaussian posterior 사이의 대칭 KL을 반환한다."""
    return 0.5 * (
        diagonal_kl(posterior_p, posterior_q, logvar_min, logvar_max)
        + diagonal_kl(posterior_q, posterior_p, logvar_min, logvar_max)
    )


def per_sample_mse(left: Tensor, right: Tensor) -> Tensor:
    """Batch마다 한 scalar MSE를 반환한다."""
    if left.shape != right.shape:
        raise ValueError(f"MSE tensors must have matching shape, got {left.shape} and {right.shape}")
    return (left - right).flatten(start_dim=1).square().mean(dim=1)


_GROUP_COLUMNS = (
    "probe_partition",
    "support_group",
    "model",
    "evaluation_condition",
)
_METRIC_COLUMNS = ("mse_input", "fg_output", "fg_mu", "fg_latent")


def _bootstrap_interval(
    values: np.ndarray, generator: np.random.Generator, resamples: int
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value
    resampled_indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[resampled_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_records(
    records: list[dict[str, Any]], bootstrap_seed: int, resamples: int = 1_000
) -> dict[str, Any]:
    """동일 support/model/condition strata의 sample-level metric을 요약한다."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(str(record[column]) for column in _GROUP_COLUMNS)].append(record)

    generator = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group_records = groups[group_key]
        group_summary: dict[str, Any] = dict(zip(_GROUP_COLUMNS, group_key))
        group_summary["sample_count"] = len(group_records)
        group_summary["metrics"] = {}
        for metric in _METRIC_COLUMNS:
            values = np.asarray(
                [float(record[metric]) for record in group_records], dtype=np.float64
            )
            ci_low, ci_high = _bootstrap_interval(values, generator, resamples)
            group_summary["metrics"][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(np.median(values)),
                "bootstrap_95_ci": [ci_low, ci_high],
            }
        summaries.append(group_summary)

    return {
        "group_columns": list(_GROUP_COLUMNS),
        "metric_columns": list(_METRIC_COLUMNS),
        "bootstrap_resamples": resamples,
        "groups": summaries,
    }
