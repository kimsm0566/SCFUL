"""Loss functions shared by semantic-codec training and FSCU."""

import math
import torch
import torch.nn.functional as functional

from src.data.data import augment_cifar10_contrastive_batch


def normalized_feature_distillation_loss(student_feature, teacher_feature):
    """Match feature-map direction without making channel count set the scale."""

    if student_feature.shape != teacher_feature.shape:
        raise ValueError("student and teacher feature maps must have the same shape")
    student_flat = functional.normalize(
        student_feature.flatten(start_dim=1), dim=1
    )
    teacher_flat = functional.normalize(
        teacher_feature.detach().flatten(start_dim=1), dim=1
    )
    return (1.0 - (student_flat * teacher_flat).sum(dim=1)).mean()


def posterior_parameter_distillation_losses(student_posterior, teacher_posterior):
    """Match retained-data posterior location and uncertainty separately."""

    if student_posterior.mu.shape != teacher_posterior.mu.shape:
        raise ValueError("student and teacher posterior means must have the same shape")
    if student_posterior.logvar.shape != teacher_posterior.logvar.shape:
        raise ValueError(
            "student and teacher posterior log-variances must have the same shape"
        )
    mean_loss = functional.mse_loss(
        student_posterior.mu,
        teacher_posterior.mu.detach(),
        reduction="mean",
    )
    logvar_loss = functional.mse_loss(
        student_posterior.logvar,
        teacher_posterior.logvar.detach(),
        reduction="mean",
    )
    return mean_loss, logvar_loss


def outside_marker_consistency_loss(
    probe_reconstruction,
    clean_reconstruction,
    marker_region,
):
    """Keep a masked marker probe from changing reconstruction elsewhere."""

    if probe_reconstruction.shape != clean_reconstruction.shape:
        raise ValueError("probe and clean reconstructions must have the same shape")
    if probe_reconstruction.ndim != 4:
        raise ValueError("marker consistency expects NCHW reconstructions")
    top, left, height, width = (int(value) for value in marker_region)
    image_height, image_width = probe_reconstruction.shape[-2:]
    if (
        top < 0
        or left < 0
        or height <= 0
        or width <= 0
        or top + height > image_height
        or left + width > image_width
    ):
        raise ValueError("marker region is outside the reconstruction bounds")
    outside_mask = torch.ones(
        (1, 1, image_height, image_width),
        dtype=probe_reconstruction.dtype,
        device=probe_reconstruction.device,
    )
    outside_mask[:, :, top : top + height, left : left + width] = 0.0
    squared_error = (
        probe_reconstruction - clean_reconstruction.detach()
    ).square() * outside_mask
    return squared_error.sum() / len(probe_reconstruction)


def _power_normalize(model, latent):
    normalize = getattr(model, "normalized_latent", None)
    if callable(normalize):
        return normalize(latent)
    return (latent.shape[1] ** 0.5) * functional.normalize(latent, dim=1)


def public_contrastive_loss(
    model,
    inputs,
    erased_buffer,
    temperature,
    first_generator,
    second_generator,
    positive_mode="deterministic_augmentation",
    crop_padding=4,
    horizontal_flip_probability=0.5,
    return_diagnostics=False,
):
    """Contrast public views against a canonical erased-representation buffer."""

    if positive_mode == "deterministic_augmentation":
        first_inputs = augment_cifar10_contrastive_batch(
            inputs,
            first_generator,
            crop_padding=crop_padding,
            horizontal_flip_probability=horizontal_flip_probability,
        )
        second_inputs = augment_cifar10_contrastive_batch(
            inputs,
            second_generator,
            crop_padding=crop_padding,
            horizontal_flip_probability=horizontal_flip_probability,
        )
        first_posterior = model(first_inputs, mode="posterior")
        second_posterior = model(second_inputs, mode="posterior")
        first_view = model.canonical_latent_from_posterior(first_posterior)
        second_view = model.canonical_latent_from_posterior(second_posterior)
    elif positive_mode == "stochastic_self":
        first_posterior = model(inputs, mode="posterior")
        second_posterior = first_posterior
        first_view = _power_normalize(
            model, model.sample_posterior(first_posterior, first_generator)
        )
        second_view = _power_normalize(
            model, model.sample_posterior(second_posterior, second_generator)
        )
    else:
        raise ValueError(f"unsupported public positive mode: {positive_mode}")

    anchors = functional.normalize(first_view, dim=1)
    positives = functional.normalize(second_view, dim=1)
    negatives = functional.normalize(erased_buffer, dim=1)
    positive_logits = (anchors * positives).sum(dim=1, keepdim=True)
    negative_logits = anchors @ negatives.transpose(0, 1)
    logits = torch.cat((positive_logits, negative_logits), dim=1) / temperature
    labels = torch.zeros(len(inputs), dtype=torch.long, device=inputs.device)
    loss = functional.cross_entropy(logits, labels)
    if not return_diagnostics:
        return loss

    logvar = 0.5 * (first_posterior.logvar + second_posterior.logvar)
    entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar)
    diagnostics = {
        "posterior_logvar_mean": logvar.mean(),
        "posterior_std_mean": logvar.mul(0.5).exp().mean(),
        "posterior_entropy_mean": entropy.mean(),
        "public_prior_kl": 0.5
        * (
            model.prior_kl(first_posterior)
            + model.prior_kl(second_posterior)
        ),
        "positive_cosine": positive_logits.mean(),
        "negative_cosine_mean": negative_logits.mean(),
        "negative_cosine_max": negative_logits.max(dim=1).values.mean(),
        "canonical_norm_mean": first_view.norm(dim=1).mean(),
    }
    return loss, diagnostics
