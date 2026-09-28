"""Full-reference image comparison; RGB [0, 1], no learned feature extractor."""

import torch
import torch.nn.functional as functional


def image_ssim(first, second):
    """Per-image SSIM: 11x11 Gaussian (sigma 1.5), valid windows, RGB mean.

    Population covariance; K1=.01, K2=.03, data_range=1. No downsampling.
    This is SSIM, not MS-SSIM. Border pixels without a full window are excluded.
    """
    if first.shape != second.shape or first.ndim != 4 or min(first.shape[-2:]) < 11:
        raise ValueError("SSIM needs matching NCHW images of at least 11x11")
    positions = torch.arange(11, dtype=first.dtype, device=first.device) - 5
    gaussian = torch.exp(-positions.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).expand(first.shape[1], 1, 11, 11)
    def smooth(value):
        return functional.conv2d(value, window, groups=first.shape[1])
    mean_first, mean_second = smooth(first), smooth(second)
    variance_first = smooth(first.square()) - mean_first.square()
    variance_second = smooth(second.square()) - mean_second.square()
    covariance = smooth(first * second) - mean_first * mean_second
    value = ((2 * mean_first * mean_second + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_first.square() + mean_second.square() + 0.01**2)
        * (variance_first + variance_second + 0.03**2)
    )
    return value.flatten(1).mean(1)


def semantic_output_comparison(logits, reference_logits, labels):
    """Per-input class agreement, logit MSE, and probability JS (natural-log nats)."""
    if logits.shape != reference_logits.shape or logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("semantic comparison requires aligned NxC logits and N labels")
    log_p = functional.log_softmax(logits.double(), dim=1)
    log_q = functional.log_softmax(reference_logits.double(), dim=1)
    log_midpoint = torch.logaddexp(log_p, log_q) - torch.log(torch.tensor(2.0, device=logits.device, dtype=torch.float64))
    js = 0.5 * (log_p.exp() * (log_p - log_midpoint) + log_q.exp() * (log_q - log_midpoint)).sum(1)
    return {
        "class_agreement_to_retrain": (logits.argmax(1) == reference_logits.argmax(1)).double(),
        "logit_mse_to_retrain": (logits.double() - reference_logits.double()).square().mean(1),
        "probability_js_to_retrain": js.clamp_min(0),
        "semantic_accuracy": (logits.argmax(1) == labels).double(),
    }
