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
