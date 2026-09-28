"""Client aggregated posterior의 단일 Gaussian 충분성 진단."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.semantic_codec import Posterior, VariationalDeepJSCC
from src.utils.reproducibility import derive_seed


class ClientPosteriorAuditConfig:
    def __init__(
        self,
        fit_fraction=0.8,
        batch_size=256,
        minimum_variance=1e-6,
        mmd_sample_limit=512,
        bootstrap_resamples=200,
        bootstrap_sample_size=256,
    ):
        self.fit_fraction = fit_fraction
        self.batch_size = batch_size
        self.minimum_variance = minimum_variance
        self.mmd_sample_limit = mmd_sample_limit
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_sample_size = bootstrap_sample_size


class ClientGaussianSummary:
    def __init__(
        self,
        client_id,
        mean,
        variance,
        metrics,
        fit_count,
        evaluation_count,
    ):
        self.client_id = client_id
        self.mean = mean
        self.variance = variance
        self.metrics = metrics
        self.fit_count = fit_count
        self.evaluation_count = evaluation_count

    def to_record(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "fit_count": self.fit_count,
            "evaluation_count": self.evaluation_count,
            "mean": self.mean,
            "variance": self.variance,
            **self.metrics,
        }


def fit_moment_matched_diagonal_gaussian(
    mu: Tensor, logvar: Tensor, minimum_variance: float
) -> Posterior:
    """Aggregated posterior의 total variance를 moment matching한다."""
    if mu.ndim != 2 or logvar.shape != mu.shape or len(mu) == 0:
        raise ValueError("mu and logvar must be non-empty tensors with matching (N, D) shape")
    aggregate_mu = mu.mean(dim=0, keepdim=True)
    aggregate_variance = (logvar.exp() + mu.square()).mean(dim=0, keepdim=True)
    aggregate_variance = (aggregate_variance - aggregate_mu.square()).clamp_min(
        minimum_variance
    )
    return Posterior(mu=aggregate_mu, logvar=aggregate_variance.log())


def _median_bandwidth(*samples: Tensor) -> float:
    joined = torch.cat(samples, dim=0)
    if len(joined) < 2:
        return 1.0
    distances_squared = torch.pdist(joined).square()
    positive = distances_squared[distances_squared > 0]
    if len(positive) == 0:
        return 1.0
    return float(positive.median().sqrt().clamp_min(1e-6).cpu())


def rbf_mmd_squared(x: Tensor, y: Tensor, bandwidth: float | None = None) -> Tensor:
    """Biased RBF MMD². `bandwidth=None`이면 combined median heuristic을 쓴다."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must have shape (N, D) and matching D")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("MMD inputs must be non-empty")
    bandwidth = bandwidth or _median_bandwidth(x, y)
    scale = 2.0 * bandwidth * bandwidth
    kernel_xx = torch.exp(-torch.cdist(x, x).square() / scale)
    kernel_yy = torch.exp(-torch.cdist(y, y).square() / scale)
    kernel_xy = torch.exp(-torch.cdist(x, y).square() / scale)
    return (kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()).clamp_min(0.0)


def _sample_rows(values: Tensor, limit: int, generator: torch.Generator) -> Tensor:
    if len(values) <= limit:
        return values
    positions = torch.randperm(len(values), generator=generator)[:limit]
    return values[positions]


def _sample_posterior(
    posterior: Posterior, generator: torch.Generator
) -> Tensor:
    epsilon = torch.randn(
        posterior.mu.shape,
        dtype=posterior.mu.dtype,
        device=posterior.mu.device,
        generator=generator,
    )
    return posterior.mu + posterior.logvar.mul(0.5).exp() * epsilon


def _diagonal_nll(samples: Tensor, posterior: Posterior) -> Tensor:
    variance = posterior.logvar.exp()
    return 0.5 * (
        ((samples - posterior.mu).square() / variance)
        + posterior.logvar
        + torch.log(torch.tensor(2.0 * torch.pi, device=samples.device))
    ).sum(dim=1)


def _label_mixture_nll_gain(
    fit_mu: Tensor,
    fit_logvar: Tensor,
    fit_labels: Tensor,
    evaluation_samples: Tensor,
    single: Posterior,
    minimum_variance: float,
) -> float:
    unique_labels = torch.unique(fit_labels).tolist()
    components: list[tuple[float, Posterior]] = []
    for label in unique_labels:
        mask = fit_labels == label
        components.append(
            (
                float(mask.float().mean()),
                fit_moment_matched_diagonal_gaussian(
                    fit_mu[mask], fit_logvar[mask], minimum_variance
                ),
            )
        )
    mixture_log_probabilities = torch.stack(
        [
            torch.log(torch.tensor(weight, device=evaluation_samples.device))
            - _diagonal_nll(evaluation_samples, component)
            for weight, component in components
        ],
        dim=1,
    )
    mixture_nll = -torch.logsumexp(mixture_log_probabilities, dim=1).mean()
    single_nll = _diagonal_nll(evaluation_samples, single).mean()
    return float(((single_nll - mixture_nll) / evaluation_samples.shape[1]).cpu())


def _distribution_shape_metrics(samples: Tensor) -> tuple[float, float]:
    centered = samples - samples.mean(dim=0, keepdim=True)
    standard_deviation = centered.square().mean(dim=0).sqrt().clamp_min(1e-8)
    standardized = centered / standard_deviation
    excess_kurtosis = standardized.pow(4).mean(dim=0) - 3.0
    if samples.shape[1] == 1:
        offdiag = torch.zeros((), device=samples.device)
    else:
        correlation = torch.corrcoef(samples.T)
        offdiag_mask = ~torch.eye(
            samples.shape[1], dtype=torch.bool, device=samples.device
        )
        offdiag = correlation[offdiag_mask].abs().mean()
    return float(offdiag.cpu()), float(excess_kurtosis.abs().mean().cpu())


def _bootstrap_mmd_ratio(
    empirical_a: Tensor,
    empirical_b: Tensor,
    single_samples: Tensor,
    bandwidth: float,
    config: ClientPosteriorAuditConfig,
    generator: torch.Generator,
) -> list[float]:
    sample_size = min(config.bootstrap_sample_size, len(empirical_a))
    ratios: list[float] = []
    for _ in range(config.bootstrap_resamples):
        indices = torch.randint(
            len(empirical_a), (sample_size,), generator=generator, device=empirical_a.device
        )
        mmd_single = rbf_mmd_squared(
            empirical_a[indices], single_samples[indices], bandwidth
        )
        mmd_noise = rbf_mmd_squared(
            empirical_a[indices], empirical_b[indices], bandwidth
        )
        ratios.append(float((mmd_single / (mmd_noise + 1e-12)).cpu()))
    quantiles = torch.tensor(ratios).quantile(torch.tensor([0.025, 0.975]))
    return [float(quantiles[0]), float(quantiles[1])]


def audit_client_gaussian_sufficiency(
    model: VariationalDeepJSCC,
    dataset: Dataset[Any],
    client_indices: dict[str, list[int]],
    labels: Tensor,
    config: ClientPosteriorAuditConfig,
    device: torch.device,
    seed: int,
) -> list[ClientGaussianSummary]:
    """각 client의 held-out aggregated posterior에 단일 Gaussian이 충분한지 진단한다."""
    with torch.no_grad():
        return _audit_client_gaussian_sufficiency(
            model,
            dataset,
            client_indices,
            labels,
            config,
            device,
            seed,
        )


def _audit_client_gaussian_sufficiency(
    model,
    dataset,
    client_indices,
    labels,
    config,
    device,
    seed,
):
    if not 0.0 < config.fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    model.eval()
    labels = labels.detach().cpu().to(torch.long)
    summaries: list[ClientGaussianSummary] = []

    for client_id, indices in client_indices.items():
        split_generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed, f"client-posterior-split:{client_id}")
        )
        permutation = torch.randperm(len(indices), generator=split_generator).tolist()
        fit_count = max(1, int(len(indices) * config.fit_fraction))
        fit_positions = permutation[:fit_count]
        evaluation_positions = permutation[fit_count:]
        if not evaluation_positions:
            raise ValueError(f"client {client_id} has no held-out posterior samples")
        fit_indices = [indices[position] for position in fit_positions]
        evaluation_indices = [indices[position] for position in evaluation_positions]

        def encode(selected_indices: list[int]) -> tuple[Tensor, Tensor, Tensor]:
            loader = DataLoader(
                Subset(dataset, selected_indices),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
            mu_chunks: list[Tensor] = []
            logvar_chunks: list[Tensor] = []
            label_chunks: list[Tensor] = []
            offset = 0
            for batch in loader:
                inputs = batch[0].to(device)
                batch_size = len(inputs)
                posterior = model.posterior(inputs)
                mu_chunks.append(posterior.mu.cpu())
                logvar_chunks.append(posterior.logvar.cpu())
                label_chunks.append(labels[selected_indices[offset : offset + batch_size]])
                offset += batch_size
            return torch.cat(mu_chunks), torch.cat(logvar_chunks), torch.cat(label_chunks)

        fit_mu, fit_logvar, fit_labels = encode(fit_indices)
        evaluation_mu, evaluation_logvar, _ = encode(evaluation_indices)
        single = fit_moment_matched_diagonal_gaussian(
            fit_mu, fit_logvar, config.minimum_variance
        )

        random_generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed, f"client-posterior-random:{client_id}")
        )
        evaluation_posterior = Posterior(mu=evaluation_mu, logvar=evaluation_logvar)
        empirical_a = _sample_posterior(evaluation_posterior, random_generator)
        empirical_b = _sample_posterior(evaluation_posterior, random_generator)
        single_samples = single.mu + single.logvar.mul(0.5).exp() * torch.randn(
            empirical_a.shape, generator=random_generator
        )
        metric_generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed, f"client-posterior-mmd:{client_id}")
        )
        empirical_a = _sample_rows(empirical_a, config.mmd_sample_limit, metric_generator)
        empirical_b = _sample_rows(empirical_b, config.mmd_sample_limit, metric_generator)
        single_samples = _sample_rows(single_samples, config.mmd_sample_limit, metric_generator)
        bandwidth = _median_bandwidth(empirical_a, empirical_b, single_samples)
        mmd_single = rbf_mmd_squared(empirical_a, single_samples, bandwidth)
        mmd_noise = rbf_mmd_squared(empirical_a, empirical_b, bandwidth)
        offdiag_correlation, excess_kurtosis = _distribution_shape_metrics(empirical_a)
        bootstrap_ci = _bootstrap_mmd_ratio(
            empirical_a,
            empirical_b,
            single_samples,
            bandwidth,
            config,
            metric_generator,
        )
        mixture_gain = _label_mixture_nll_gain(
            fit_mu,
            fit_logvar,
            fit_labels,
            empirical_a,
            single,
            config.minimum_variance,
        )
        summaries.append(
            ClientGaussianSummary(
                client_id=client_id,
                mean=single.mu.squeeze(0).tolist(),
                variance=single.logvar.exp().squeeze(0).tolist(),
                fit_count=len(fit_indices),
                evaluation_count=len(evaluation_indices),
                metrics={
                    "mmd_single": float(mmd_single),
                    "mmd_empirical_noise": float(mmd_noise),
                    "mmd_excess_ratio": float(mmd_single / (mmd_noise + 1e-12)),
                    "mmd_excess_ratio_bootstrap_95_ci": bootstrap_ci,
                    "mmd_bandwidth": bandwidth,
                    "mean_abs_offdiag_correlation": offdiag_correlation,
                    "mean_abs_excess_kurtosis": excess_kurtosis,
                    "label_mixture_nll_gain_per_dim": mixture_gain,
                },
            )
        )
    return summaries


def client_posterior_summary_payload(
    summaries: list[ClientGaussianSummary], config: ClientPosteriorAuditConfig
) -> dict[str, Any]:
    """Artifact JSON에 넣을 충분성 진단의 해석 metadata를 만든다."""
    return {
        "fit_fraction": config.fit_fraction,
        "mmd_policy": "median-bandwidth RBF MMD squared",
        "bootstrap_resamples": config.bootstrap_resamples,
        "interpretation": (
            "이 결과는 centralized diagnostic이다. 단일 (mu_k, sigma_k^2) summary의 "
            "채택 여부는 MMD/noise floor/correlation/kurtosis/mixture NLL을 함께 보고 판단한다."
        ),
        "clients": [summary.to_record() for summary in summaries],
        "config": dict(vars(config)),
    }
