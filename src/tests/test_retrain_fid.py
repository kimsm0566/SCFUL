"""FID math tests plus real-CIFAR codec/feature/CSV/figure integration checks."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.data import load_cifar10_test_dataset, select_cifar10_client_data
from src.evaluation.federated_behavior import (
    evaluate_codec_states,
    evaluate_retain_ufid_states,
)
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.utils.artifacts import RunArtifacts
from src.utils.fid import FeatureMoments, InceptionFIDFeatures, fid_from_moments
from src.utils.plotting import save_retain_ufid_figure, save_retrain_fid_figure


ROOT = Path(__file__).resolve().parents[2]


def moments(values, batch_size=3):
    result = FeatureMoments()
    for start in range(0, len(values), batch_size):
        result.update(values[start:start + batch_size])
    return result


def real_cifar_data(count=6):
    dataset = load_cifar10_test_dataset(
        {"name": "CIFAR10", "data_root": "data/cifar10", "download": False}, ROOT
    )
    return select_cifar10_client_data(dataset, np.arange(count))


def test_streaming_moments_match_numpy_unbiased_covariance_with_uneven_batches():
    features = np.random.default_rng(17).normal(size=(11, 5))
    mean, covariance = moments(features).statistics()
    np.testing.assert_allclose(mean, features.mean(0), atol=1e-15)
    np.testing.assert_allclose(covariance, np.cov(features, rowvar=False), atol=1e-15)


def test_fid_matches_pytorch_fid_and_is_symmetric():
    from pytorch_fid.fid_score import calculate_frechet_distance

    generator = np.random.default_rng(3)
    first = generator.normal(size=(30, 5))
    second = generator.normal(size=(30, 5)) * 2 + 1
    expected = calculate_frechet_distance(
        first.mean(0), np.cov(first, rowvar=False),
        second.mean(0), np.cov(second, rowvar=False),
    )
    actual = fid_from_moments(moments(first), moments(second))
    assert actual == pytest.approx(expected, abs=1e-10)
    assert actual == pytest.approx(fid_from_moments(moments(second), moments(first)), abs=1e-10)


def test_fid_self_zero_and_mean_shift_has_known_distance():
    features = np.random.default_rng(7).normal(size=(20, 3))
    first = moments(features)
    assert fid_from_moments(first, first) == 0.0
    assert fid_from_moments(first, moments(features + [1, 2, 3])) == pytest.approx(14.0)


def test_fid_handles_singular_covariance_and_rejects_invalid_features():
    first = moments(np.ones((3, 5)))
    second = moments(np.ones((3, 5)) * 2)
    assert fid_from_moments(first, first) == 0.0
    assert fid_from_moments(first, second) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="at least two"):
        moments(np.ones((1, 5))).statistics()
    with pytest.raises(ValueError, match="finite"):
        FeatureMoments().update(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="dimension"):
        first.update(np.ones((2, 4)))


class DiagnosticFeatures:
    """Cheap non-Inception test double; never used for reported FID."""

    def __init__(self):
        self.metadata = {"feature_dimensions": 3, "implementation": "test-double-only"}
        self.seen = []

    def __call__(self, images):
        assert not torch.is_grad_enabled()
        self.seen.append(images.detach().cpu().clone())
        return images.mean(dim=(2, 3))


def test_codec_fid_uses_retrain_outputs_and_shared_awgn_without_changing_psnr():
    config = VariationalDeepJSCCConfig(latent_channels=1, latent_dim=64)
    model = VariationalDeepJSCC(config)
    state = model.state_dict()
    changed = {name: tensor.clone() for name, tensor in state.items()}
    changed["decoder.8.bias"] += 0.2
    data = real_cifar_data()
    features = DiagnosticFeatures()
    arguments = dict(
        model_factory=lambda: VariationalDeepJSCC(config),
        model_states={"retrain_without_client_0": state, "identical": state, "changed": changed},
        reference_name="retrain_without_client_0", test_data=data,
        client_test_indices=[np.arange(6)], batch_size=4, device="cpu", seed=3,
    )
    baseline = evaluate_codec_states(**arguments)
    result = evaluate_codec_states(**arguments, fid_features=features)
    by_name = {row["model"]: row for row in result["summary"]}
    assert by_name["retrain_without_client_0"]["fid_to_retrain"] == 0
    assert by_name["identical"]["fid_to_retrain"] == 0
    assert by_name["changed"]["fid_to_retrain"] > 0
    for before, after in zip(baseline["summary"], result["summary"]):
        assert before["psnr"] == after["psnr"]
        assert after["fid_samples"] == 6
    torch.testing.assert_close(features.seen[0], features.seen[1], rtol=0, atol=0)
    assert not torch.allclose(features.seen[0], torch.as_tensor(data[0][:4]).float())
    assert result["protocol"]["fid"]["reference_model"] == "retrain_without_client_0"
    assert "fid_to_retrain" not in result["per_class"][0]


def test_retain_ufid_uses_real_retain_inputs_as_reference(tmp_path):
    config = VariationalDeepJSCCConfig(latent_channels=1, latent_dim=64)
    model = VariationalDeepJSCC(config)
    state = model.state_dict()
    identical = {name: tensor.clone() for name, tensor in state.items()}
    features = DiagnosticFeatures()
    rows, protocol = evaluate_retain_ufid_states(
        model_factory=lambda: VariationalDeepJSCC(config),
        model_states={"original": state, "identical": identical},
        retain_data=real_cifar_data(6),
        batch_size=3,
        device="cpu",
        snr_db=20.0,
        seed=9,
        fid_features=features,
    )
    by_name = {row["model"]: row for row in rows}
    assert by_name["original"]["retain_ufid"] == pytest.approx(
        by_name["identical"]["retain_ufid"]
    )
    assert by_name["original"]["retain_ufid"] >= 0
    assert protocol["reference_images"] == (
        "the same real clean remaining-client inputs"
    )
    assert protocol["metric"] == "retain_uFID"
    save_retain_ufid_figure(rows, tmp_path)
    assert (tmp_path / "retain_ufid_comparison.png").stat().st_size > 0


def test_pretrained_inception_real_cifar_features_are_frozen_and_reproducible():
    extractor = InceptionFIDFeatures(device="cpu", batch_size=2)
    images = torch.as_tensor(real_cifar_data(2)[0]).float()
    if images.max() > 1:
        images = images / 255.0
    images.requires_grad_(True)
    first = extractor(images)
    second = extractor(images)
    assert first.shape == (2, 2048)
    assert torch.isfinite(first).all()
    assert not first.requires_grad
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    assert len(extractor.metadata["feature_state_sha256"]) == 64
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        extractor(images + 2)


def test_retrain_fid_figure_is_backed_by_nonempty_saved_metric_rows(tmp_path):
    artifacts = RunArtifacts(tmp_path)
    rows = [
        {"model": model, "split": split, "fid_to_retrain": value,
         "samples": 6, "evaluation_snr_db": 20, "smoke_test": True}
        for split in ("clean_test", "erased_train")
        for model, value in (("original", 3.0), ("retrain", 0.0), ("unlearned", 1.0))
    ]
    metric_path = artifacts.write_csv("metrics/retrain_fid.csv", rows)
    with metric_path.open() as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == 6
    save_retrain_fid_figure(saved_rows, artifacts.path("figures"))
    assert (tmp_path / "figures/retrain_fid_comparison.png").stat().st_size > 0
    with pytest.raises(ValueError, match="empty"):
        save_retrain_fid_figure([], artifacts.path("figures"))


def test_smoke_audit_cannot_overwrite_source_results():
    from src.experiments.reconstruction_channel_audit import run_reconstruction_channel_audit

    with pytest.raises(ValueError, match="not overwrite"):
        run_reconstruction_channel_audit("unused", None, seed=0, smoke_test=True, in_place=True)
