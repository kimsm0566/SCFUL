"""연합 삭제 model을 재학습 기준 model과 비교하는 reconstruction metric."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.utils.fid import FeatureMoments, fid_from_moments
from src.utils.image_metrics import image_ssim, semantic_output_comparison


def _symmetric_posterior_kl(first, second):
    first_to_second = 0.5 * (
        second.logvar
        - first.logvar
        + (first.logvar.exp() + (first.mu - second.mu).square())
        / second.logvar.exp()
        - 1.0
    ).sum(dim=1)
    second_to_first = 0.5 * (
        first.logvar
        - second.logvar
        + (second.logvar.exp() + (second.mu - first.mu).square())
        / first.logvar.exp()
        - 1.0
    ).sum(dim=1)
    return 0.5 * (first_to_second + second_to_first)


def _record(model_name, group_name, group_id, metrics, indices, snr_db):
    indices = np.asarray(indices, dtype=np.int64)
    mse_input = float(metrics["mse_input"][indices].mean())
    mse_input_noiseless = float(metrics["mse_input_noiseless"][indices].mean())
    record = {
        "model": model_name,
        "group": group_name,
        "group_id": int(group_id),
        "samples": int(len(indices)),
        "evaluation_channel": "paired_awgn_posterior_mean",
        "evaluation_snr_db": float(snr_db),
        "mse_input": mse_input,
        "rmse_255": 255.0 * math.sqrt(max(mse_input, 0.0)),
        "psnr": 10.0 * math.log10(1.0 / max(mse_input, 1e-12)),
        "fg_output": float(metrics["fg_output"][indices].mean()),
        "mse_input_noiseless": mse_input_noiseless,
        "rmse_255_noiseless": 255.0
        * math.sqrt(max(mse_input_noiseless, 0.0)),
        "psnr_noiseless": 10.0
        * math.log10(1.0 / max(mse_input_noiseless, 1e-12)),
        "fg_output_noiseless": float(
            metrics["fg_output_noiseless"][indices].mean()
        ),
        "fg_mu": float(metrics["fg_mu"][indices].mean()),
        "fg_latent": float(metrics["fg_latent"][indices].mean()),
    }
    for key in ("ssim_input", "ssim_to_retrain", "class_agreement_to_retrain",
                "logit_mse_to_retrain", "probability_js_to_retrain", "semantic_accuracy"):
        if key in metrics:
            record[key] = float(metrics[key][indices].mean())
    if "ssim_to_retrain" in metrics:
        record["psnr_to_retrain_db"] = 10.0 * math.log10(1.0 / max(record["fg_output"], 1e-12))
        record["psnr_to_retrain_capped"] = record["fg_output"] <= 1e-12
    return record


def _evaluate_model_with_paired_awgn(model, batch, standard_noise, snr_db):
    """Posterior mean을 동일 AWGN realization으로 전송하고 noiseless 출력도 반환한다."""
    posterior = model(batch, mode="posterior")
    noiseless_reconstruction = model(batch, mode="mean_reconstruction").reconstruction
    reconstruction = model(
        batch, mode="paired_awgn", standard_noise=standard_noise, snr_db=snr_db
    ).reconstruction
    return posterior, reconstruction, noiseless_reconstruction


def evaluate_candidate_against_reference(
    model_factory,
    candidate_state,
    reference_state,
    test_data,
    batch_size,
    device,
    snr_db=20.0,
    seed=0,
):
    """Paired AWGN에서 candidate와 retrain reference의 평균 gap을 계산한다."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    images, _labels = test_data
    images = torch.as_tensor(images, dtype=torch.float32)
    if images.max() > 1.0:
        images = images / 255.0
    if len(images) == 0:
        raise ValueError("screening data must contain at least one sample")

    device = torch.device(device)
    candidate = model_factory().to(device).eval()
    candidate.load_state_dict(candidate_state)
    reference = model_factory().to(device).eval()
    reference.load_state_dict(reference_state)
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(seed))
    totals = {
        "mse_input": 0.0,
        "fg_output": 0.0,
        "mse_input_noiseless": 0.0,
        "fg_output_noiseless": 0.0,
        "fg_mu": 0.0,
        "fg_latent": 0.0,
    }

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            reference_posterior = reference.posterior(batch)
            standard_noise = torch.randn(
                reference_posterior.mu.shape,
                dtype=reference_posterior.mu.dtype,
                generator=noise_generator,
            )
            (
                reference_posterior,
                reference_reconstruction,
                reference_reconstruction_noiseless,
            ) = _evaluate_model_with_paired_awgn(
                reference, batch, standard_noise, snr_db
            )
            (
                candidate_posterior,
                candidate_reconstruction,
                candidate_reconstruction_noiseless,
            ) = _evaluate_model_with_paired_awgn(
                candidate, batch, standard_noise, snr_db
            )
            totals["mse_input"] += float(
                (candidate_reconstruction - batch)
                .flatten(start_dim=1)
                .square()
                .mean(dim=1)
                .sum()
                .cpu()
            )
            totals["fg_output"] += float(
                (candidate_reconstruction - reference_reconstruction)
                .flatten(start_dim=1)
                .square()
                .mean(dim=1)
                .sum()
                .cpu()
            )
            totals["mse_input_noiseless"] += float(
                (candidate_reconstruction_noiseless - batch)
                .flatten(start_dim=1)
                .square()
                .mean(dim=1)
                .sum()
                .cpu()
            )
            totals["fg_output_noiseless"] += float(
                (
                    candidate_reconstruction_noiseless
                    - reference_reconstruction_noiseless
                )
                .flatten(start_dim=1)
                .square()
                .mean(dim=1)
                .sum()
                .cpu()
            )
            totals["fg_mu"] += float(
                (candidate_posterior.mu - reference_posterior.mu)
                .square()
                .mean(dim=1)
                .sum()
                .cpu()
            )
            totals["fg_latent"] += float(
                _symmetric_posterior_kl(
                    candidate_posterior, reference_posterior
                )
                .sum()
                .cpu()
            )

    return {
        "samples": int(len(images)),
        "evaluation_channel": "paired_awgn_posterior_mean",
        "evaluation_snr_db": float(snr_db),
        **{name: total / len(images) for name, total in totals.items()},
    }


def evaluate_codec_states(
    model_factory,
    model_states,
    reference_name,
    test_data,
    client_test_indices,
    batch_size,
    device,
    snr_db=20.0,
    seed=0,
    fid_features=None,
    semantic_classifier=None,
    output_comparison=False,
):
    """같은 test sample과 paired AWGN에서 reconstruction 및 gap을 계산한다."""
    if reference_name not in model_states:
        raise ValueError("reference model state is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    images, labels = test_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if len(images) == 0 or (fid_features is not None and len(images) < 2):
        raise ValueError("evaluation needs samples; FID needs at least two")
    if images.max() > 1.0:
        images = images / 255.0
    if len(images) != len(labels):
        raise ValueError("test images and labels must have the same length")

    device = torch.device(device)
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    models = {}
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(seed))
    fid_moments = {name: FeatureMoments() for name in models} if fid_features is not None else {}

    metric_arrays = {
        name: {
            "mse_input": np.zeros(len(images), dtype=np.float64),
            "fg_output": np.zeros(len(images), dtype=np.float64),
            "mse_input_noiseless": np.zeros(len(images), dtype=np.float64),
            "fg_output_noiseless": np.zeros(len(images), dtype=np.float64),
            "fg_mu": np.zeros(len(images), dtype=np.float64),
            "fg_latent": np.zeros(len(images), dtype=np.float64),
        }
        for name in models
    }
    extra_keys = ["ssim_input", "ssim_to_retrain"] if output_comparison else []
    if semantic_classifier is not None:
        extra_keys += ["class_agreement_to_retrain", "logit_mse_to_retrain", "probability_js_to_retrain", "semantic_accuracy"]
    for arrays in metric_arrays.values():
        arrays.update({key: np.zeros(len(images), dtype=np.float64) for key in extra_keys})
    logit_chunks = {name: [] for name in models} if semantic_classifier is not None else {}

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch = images[start:end].to(device)
            reference_posterior = models[reference_name](batch, mode="posterior")
            standard_noise = torch.randn(
                reference_posterior.mu.shape,
                dtype=reference_posterior.mu.dtype,
                generator=noise_generator,
            )
            (
                reference_posterior,
                reference_reconstruction,
                reference_reconstruction_noiseless,
            ) = _evaluate_model_with_paired_awgn(
                models[reference_name], batch, standard_noise, snr_db
            )
            reference_logits = (
                semantic_classifier(reference_reconstruction.detach())
                if semantic_classifier is not None else None
            )
            for name, model in models.items():
                if name == reference_name:
                    posterior = reference_posterior
                    reconstruction = reference_reconstruction
                    reconstruction_noiseless = reference_reconstruction_noiseless
                else:
                    posterior, reconstruction, reconstruction_noiseless = _evaluate_model_with_paired_awgn(
                        model, batch, standard_noise, snr_db
                    )
                if output_comparison:
                    metric_arrays[name]["ssim_input"][start:end] = image_ssim(reconstruction, batch).cpu().numpy()
                    metric_arrays[name]["ssim_to_retrain"][start:end] = image_ssim(reconstruction, reference_reconstruction).cpu().numpy()
                if semantic_classifier is not None:
                    logits = reference_logits if name == reference_name else semantic_classifier(reconstruction.detach())
                    logit_chunks[name].append(logits.cpu())
                    comparisons = semantic_output_comparison(logits, reference_logits, labels[start:end].to(device))
                    for key, values in comparisons.items():
                        metric_arrays[name][key][start:end] = values.cpu().numpy()
                if fid_features is not None:
                    fid_moments[name].update(fid_features(reconstruction))
                metric_arrays[name]["mse_input"][start:end] = (
                    (reconstruction - batch)
                    .flatten(start_dim=1)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                metric_arrays[name]["fg_output"][start:end] = (
                    (reconstruction - reference_reconstruction)
                    .flatten(start_dim=1)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                metric_arrays[name]["mse_input_noiseless"][start:end] = (
                    (reconstruction_noiseless - batch)
                    .flatten(start_dim=1)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                metric_arrays[name]["fg_output_noiseless"][start:end] = (
                    (reconstruction_noiseless - reference_reconstruction_noiseless)
                    .flatten(start_dim=1)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                metric_arrays[name]["fg_mu"][start:end] = (
                    (posterior.mu - reference_posterior.mu)
                    .square()
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                metric_arrays[name]["fg_latent"][start:end] = (
                    _symmetric_posterior_kl(posterior, reference_posterior)
                    .cpu()
                    .numpy()
                )

    all_indices = np.arange(len(images), dtype=np.int64)
    label_array = labels.cpu().numpy()
    summary_records = []
    class_records = []
    client_records = []
    for name, metrics in metric_arrays.items():
        summary_records.append(
            _record(name, "all", -1, metrics, all_indices, snr_db)
        )
        if fid_features is not None:
            summary_records[-1].update({
                "fid_to_retrain": fid_from_moments(fid_moments[name], fid_moments[reference_name]),
                "fid_samples": fid_moments[name].count,
            })
        for label in np.unique(label_array):
            class_indices = np.flatnonzero(label_array == label)
            class_records.append(
                _record(name, "class", label, metrics, class_indices, snr_db)
            )
        for client_id, indices in enumerate(client_test_indices):
            if len(indices) == 0:
                continue
            client_records.append(
                _record(name, "client", client_id, metrics, indices, snr_db)
            )

    result = {
        "protocol": {
            "channel": "awgn",
            "snr_db": float(snr_db),
            "channel_noise": "paired_across_models",
            "posterior_path": "mean",
            "seed": int(seed),
            "canonical_metrics": ["rmse_255", "psnr", "fg_output"],
            "internal_reconstruction_metric": "mse_input",
            "supplementary_noiseless_metrics": [
                "mse_input_noiseless",
                "rmse_255_noiseless",
                "psnr_noiseless",
                "fg_output_noiseless",
            ],
        },
        "summary": summary_records,
        "per_class": class_records,
        "per_client": client_records,
    }
    if fid_features is not None:
        result["protocol"]["fid"] = {
            **fid_features.metadata,
            "reference_model": reference_name,
            "reference_images": "retrain reconstructions on the same inputs and AWGN draws, not ground-truth images",
            "samples_per_distribution": len(images),
            "scope": "entire supplied split; not averaged over batch/class/client FIDs",
            "direction": "lower is closer to retrain; retrain self-comparison is zero by construction",
            "rank_deficient_sample_count": len(images) <= fid_features.metadata["feature_dimensions"],
        }
        result["protocol"]["canonical_metrics"].append("fid_to_retrain")
    if extra_keys:
        result["per_sample"] = [
            {"model": name, "input_position": index, "label": int(labels[index]),
             "output_mse_to_retrain": float(arrays["fg_output"][index]),
             **{key: float(arrays[key][index]) for key in extra_keys}}
            for name, arrays in metric_arrays.items() for index in range(len(images))
        ]
        result["protocol"]["output_comparison"] = {
            "reference_model": reference_name, "shared_inputs_and_channel_noise": True,
            "ssim": "RGB channel mean; 11x11 Gaussian sigma=1.5; valid windows; population covariance; K1=.01 K2=.03; range=1; no downsampling",
            "psnr_to_retrain": "from pooled output MSE, peak=1; floor=1e-12 caps exact matches at 120 dB",
            "classifier": "one shared frozen observer on decoded RGB; no_grad + detached input",
            "probability_js": "sample mean Jensen-Shannon divergence; natural logs (nats); temperature=1",
            "class_agreement": "per-input argmax equality with retrain, NOT equality of accuracies",
            "logit_mse": "mean over samples and 10 class logits; supplementary shift-sensitive diagnostic",
        }
    if semantic_classifier is not None:
        result["semantic_logits"] = {name: torch.cat(chunks) for name, chunks in logit_chunks.items()}
    return result


def evaluate_retain_ufid_states(
    model_factory,
    model_states,
    retain_data,
    batch_size,
    device,
    snr_db,
    seed,
    fid_features,
):
    """Compute FID from retain reconstructions to the real retain inputs."""

    if fid_features is None:
        raise ValueError("retain u-FID requires a frozen FID feature extractor")
    if batch_size <= 0:
        raise ValueError("retain u-FID batch_size must be positive")
    images, labels = retain_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    if len(images) < 2 or len(images) != len(labels):
        raise ValueError("retain u-FID needs at least two aligned images and labels")
    device = torch.device(device)
    models = {}
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    real_moments = FeatureMoments()
    reconstruction_moments = {name: FeatureMoments() for name in models}
    noise_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)].to(device)
            real_moments.update(fid_features(batch))
            first_model = next(iter(models.values()))
            posterior = first_model(batch, mode="posterior")
            standard_noise = torch.randn(
                posterior.mu.shape,
                dtype=posterior.mu.dtype,
                generator=noise_generator,
            )
            for name, model in models.items():
                reconstruction = model(
                    batch,
                    mode="paired_awgn",
                    standard_noise=standard_noise,
                    snr_db=float(snr_db),
                ).reconstruction
                reconstruction_moments[name].update(
                    fid_features(reconstruction)
                )
    class_counts = {
        int(label): int((labels == label).sum())
        for label in torch.unique(labels, sorted=True)
    }
    rows = [
        {
            "model": name,
            "retain_ufid": fid_from_moments(moments, real_moments),
            "reference_distribution": "real_clean_remaining_client_images",
            "samples": int(real_moments.count),
            "feature_dimensions": int(
                fid_features.metadata["feature_dimensions"]
            ),
            "evaluation_channel": "paired_awgn_posterior_mean",
            "evaluation_snr_db": float(snr_db),
            "evaluation_seed": int(seed),
            "rank_deficient_sample_count": (
                real_moments.count
                <= int(fid_features.metadata["feature_dimensions"])
            ),
        }
        for name, moments in reconstruction_moments.items()
    ]
    protocol = {
        **fid_features.metadata,
        "metric": "retain_uFID",
        "candidate_images": "codec reconstructions of the selected clean remaining-client inputs",
        "reference_images": "the same real clean remaining-client inputs",
        "channel": "paired AWGN across models",
        "snr_db": float(snr_db),
        "seed": int(seed),
        "samples_per_distribution": int(real_moments.count),
        "class_counts": class_counts,
        "direction": "lower is better retain reconstruction distribution quality",
        "distinction_from_fid_to_retrain": "ground-truth retain inputs are the reference here; retrain outputs are the reference for FID-to-Retrain",
        "rank_deficient_sample_count": (
            real_moments.count
            <= int(fid_features.metadata["feature_dimensions"])
        ),
    }
    return rows, protocol
