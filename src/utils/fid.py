"""FID-specific frozen Inception features and streaming, float64 statistics.

This measures distributions of reconstructed images, not classification accuracy.
The caller supplies the retrained codec's outputs as the reference distribution.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version

import numpy as np
import torch


class InceptionFIDFeatures(torch.nn.Module):
    """pytorch-fid's TensorFlow-compatible Inception, pool3 (2048 dimensions)."""

    def __init__(self, device, batch_size=32):
        super().__init__()
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("FID batch_size must be a positive integer")
        from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3

        self.batch_size = int(batch_size)
        self.network = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
            resize_input=True,
            normalize_input=True,
            requires_grad=False,
            use_fid_inception=True,
        ).to(device).eval()
        self.eval()
        state_hash = hashlib.sha256()
        for name, value in sorted(self.network.state_dict().items()):
            state_hash.update(name.encode("utf-8"))
            state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.metadata = {
            "implementation": "pytorch-fid",
            "version": version("pytorch-fid"),
            "feature_layer": "InceptionV3 final average pool (pool3)",
            "feature_dimensions": 2048,
            "weights_url": FID_WEIGHTS_URL,
            "feature_state_sha256": state_hash.hexdigest(),
            "input": "float32 RGB reconstruction in [0, 1]; no PNG/uint8 quantization",
            "resize": "bilinear 299x299, align_corners=False (pytorch-fid)",
            "normalization": "2*x-1 (pytorch-fid); not ImageNet mean/std",
            "feature_batch_size": self.batch_size,
            "statistics_dtype": "float64",
            "covariance": "unbiased sample covariance (N-1), streaming batch-merge",
            "distance": "pytorch_fid.fid_score.calculate_frechet_distance",
            "small_sample_warning": "N <= 2048 gives rank-deficient covariance; not a reliable paper estimate",
        }

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
            raise ValueError("FID expects a nonempty NCHW RGB image batch")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("FID images must be finite and in [0, 1]")
        self.network.eval()
        with torch.no_grad():
            return torch.cat([
                self.network(batch)[0].flatten(1).cpu()
                for batch in images.split(self.batch_size)
            ])


class FeatureMoments:
    """Accumulate sufficient statistics without retaining all decoded images."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.centered_sum = None

    def update(self, features):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or min(features.shape) == 0 or not np.isfinite(features).all():
            raise ValueError("FID features must be a nonempty finite N x D matrix")
        batch_mean = features.mean(axis=0)
        centered = features - batch_mean
        batch_sum = centered.T @ centered
        batch_count = len(features)
        if self.count == 0:
            self.mean = batch_mean
            self.centered_sum = batch_sum
        else:
            if batch_mean.shape != self.mean.shape:
                raise ValueError("FID feature dimension changed between batches")
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.centered_sum += batch_sum + np.outer(delta, delta) * (
                self.count * batch_count / total
            )
            self.mean += delta * (batch_count / total)
        self.count += batch_count

    def statistics(self):
        if self.count < 2:
            raise ValueError("FID needs at least two images per distribution")
        return self.mean, self.centered_sum / (self.count - 1)


def fid_from_moments(candidate, reference):
    """Standard FID; lower means closer to the supplied retrain distribution."""
    from pytorch_fid.fid_score import calculate_frechet_distance

    mean, covariance = candidate.statistics()
    reference_mean, reference_covariance = reference.statistics()
    if mean.shape != reference_mean.shape:
        raise ValueError("FID candidate and reference feature dimensions differ")
    # Avoid sqrtm roundoff for a true self comparison (especially singular smoke data).
    if np.array_equal(mean, reference_mean) and np.array_equal(covariance, reference_covariance):
        return 0.0
    value = float(calculate_frechet_distance(
        mean, covariance, reference_mean, reference_covariance
    ))
    tolerance = 1e-6 * max(1.0, float(np.trace(covariance) + np.trace(reference_covariance)))
    if not np.isfinite(value) or value < -tolerance:
        raise ValueError(f"FID is numerically invalid: {value}")
    return max(0.0, value)
