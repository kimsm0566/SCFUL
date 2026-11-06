"""Analytic Fisher checks and real-CIFAR artifact integration for dFIL."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.data import load_cifar10_test_dataset, select_cifar10_client_data
from src.evaluation.dfil import evaluate_encoder_dfil, posterior_dfil
from src.experiments.reconstruction_channel_audit import (
    _selected_core_metric_rows, _validate_core_artifacts,
)
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.utils.artifacts import RunArtifacts
from src.utils.plotting import save_encoder_dfil_figure, save_selected_core_metrics_figure


def real_images(count=2):
    dataset = load_cifar10_test_dataset(
        {"name": "CIFAR10", "data_root": "data/cifar10", "download": False},
        Path(__file__).resolve().parents[2],
    )
    images, _ = select_cifar10_client_data(dataset, np.arange(count))
    images = torch.as_tensor(images)
    return images.float() / 255 if images.dtype == torch.uint8 else images.float()


class DiagonalGaussian(torch.nn.Module):
    """Analytic independent-coordinate encoder, evaluated on real CIFAR images."""

    def __init__(self, gain=2.0, variance=4.0, scale_slope=0.0):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(float(gain)))
        self.variance = variance
        self.scale_slope = scale_slope

    def forward(self, inputs, mode):
        assert mode == "posterior"
        first = inputs.flatten(1)[:, :4]
        return SimpleNamespace(
            mu=self.gain * first,
            logvar=first * self.scale_slope + np.log(self.variance),
        )


def test_dfil_matches_analytic_mean_and_input_dependent_variance():
    x = real_images()
    for slope in (0.0, 0.7):
        model = DiagonalGaussian(scale_slope=slope).eval()
        model.gain.grad = torch.tensor(9.0)
        with torch.no_grad():
            result = posterior_dfil(model, x, probes=4, seed=13)
        variance = 4 * torch.exp(slope * x.flatten(1)[:, :4])
        expected = ((4 / variance).sum(1) + 0.5 * 4 * slope ** 2) / x[0].numel()
        torch.testing.assert_close(result["dfil"], expected.double(), rtol=1e-5, atol=1e-10)
        assert float(model.gain.grad) == 9
        assert x.grad is None
        assert result["mc_standard_error"].max() < 1e-10


def test_dfil_noise_scaling_and_zero_input_information():
    x = real_images()
    first = posterior_dfil(DiagonalGaussian(variance=1).eval(), x, probes=2)
    second = posterior_dfil(DiagonalGaussian(variance=4).eval(), x, probes=2)
    torch.testing.assert_close(first["dfil"], second["dfil"] * 4)
    constant = posterior_dfil(DiagonalGaussian(gain=0).eval(), x, probes=2)
    assert torch.equal(constant["dfil"], torch.zeros_like(constant["dfil"]))
    with pytest.raises(ValueError, match="probes"):
        posterior_dfil(DiagonalGaussian().eval(), x, probes=1)
    with pytest.raises(ValueError, match="eval-mode"):
        posterior_dfil(DiagonalGaussian(), x)


def test_real_codec_dfil_reproducible_artifacts_and_core_suite(tmp_path):
    torch.manual_seed(7)
    factory = lambda: VariationalDeepJSCC(VariationalDeepJSCCConfig())
    state = factory().state_dict()
    x = real_images()
    splits = {name: (x, np.arange(len(x)))
              for name in ("clean_test", "retain_train", "erased_train")}
    result = evaluate_encoder_dfil(
        factory, {"original": state, "retrain_without_client_0": state},
        splits, probes=2, max_examples=1, seed=7, smoke_test=True,
    )
    assert len(result["summary"]) == 6
    assert all(np.isfinite(row["dfil"]) and row["dfil"] >= 0
               for row in result["per_sample"])
    for split in splits:
        paired = [row for row in result["per_sample"] if row["split"] == split]
        assert paired[0]["dfil"] == paired[1]["dfil"]
        assert paired[0]["dataset_index"] == paired[1]["dataset_index"]
    artifacts = RunArtifacts(tmp_path)
    artifacts.write_csv("metrics/encoder_dfil.csv", result["summary"])
    artifacts.write_csv("metrics/encoder_dfil_per_sample.csv", result["per_sample"])
    artifacts.write_json("metrics/encoder_dfil_protocol.json", result["protocol"])
    save_encoder_dfil_figure(result["summary"], artifacts.path("figures"))
    extended = []
    for model in ("original", "retrain_without_client_0"):
        lookup = {row["split"]: row["dfil"] for row in result["summary"] if row["model"] == model}
        extended.append({
            "model": model, "retain_ufid": 1.0,
            "clean_psnr_db": 20.0, "erased_psnr_db": 19.0,
            "clean_ssim": 0.7, "erased_ssim": 0.6,
            "clean_dfil": lookup["clean_test"], "retain_dfil": lookup["retain_train"],
            "erased_dfil": lookup["erased_train"],
            "running_time_seconds": 1.0, "running_time_scope": "test-only",
        })
    core = _selected_core_metric_rows(extended, [])
    assert all(row["output_pul_status"] == "not_applicable_no_marker_protocol" for row in core)
    assert all(row["output_marker_positive_percent"] is None for row in core)
    assert "encoder_idi" not in core[0]
    artifacts.write_csv("metrics/selected_core_metrics.csv", core)
    save_selected_core_metrics_figure(core, artifacts.path("figures"))
    for name in ("encoder_dfil.png", "selected_core_metrics.png"):
        assert artifacts.path("figures/" + name).stat().st_size > 0
    with pytest.raises(RuntimeError, match="Incomplete core audit"):
        _validate_core_artifacts(tmp_path, [row["model"] for row in core])
    for name in ("selected_core_metrics_protocol", "retain_ufid_protocol"):
        artifacts.write_json("metrics/" + name + ".json", {"test_only": True})
    artifacts.write_csv("metrics/core_runtime.csv", core)
    artifacts.write_csv("metrics/retain_ufid.csv", core)
    _validate_core_artifacts(tmp_path, [row["model"] for row in core])
    with pytest.raises(RuntimeError, match="checkpoint rows"):
        _validate_core_artifacts(tmp_path, ["missing-model"])
    with pytest.raises(RuntimeError, match="reconstruction_marker_pul"):
        _validate_core_artifacts(tmp_path, [row["model"] for row in core], marker_required=True)
    marker_rows = [
        {
            "model": row["model"], "marker_positive_count": 0,
            "evaluation_samples": 500, "marker_positive_rate": 0.0,
            "marker_pul_percent": None,
            "marker_pul_status": "original_has_no_detectable_marker",
        }
        for row in core
    ]
    artifacts.write_csv("metrics/reconstruction_marker_pul.csv", marker_rows)
    artifacts.write_csv("metrics/reconstruction_marker_detection_per_sample.csv", marker_rows)
    artifacts.write_json("metrics/reconstruction_marker_pul_protocol.json", {"test_only": True})
    with pytest.raises(RuntimeError, match="missing absolute marker detection"):
        _validate_core_artifacts(tmp_path, [row["model"] for row in core], marker_required=True)
    core = _selected_core_metric_rows(extended, marker_rows)
    artifacts.write_csv("metrics/selected_core_metrics.csv", core)
    save_selected_core_metrics_figure(core, artifacts.path("figures"))
    _validate_core_artifacts(tmp_path, [row["model"] for row in core], marker_required=True)
    core[0]["output_marker_positive_percent"] = 1.0
    artifacts.write_csv("metrics/selected_core_metrics.csv", core)
    with pytest.raises(RuntimeError, match="inconsistent absolute marker detection"):
        _validate_core_artifacts(tmp_path, [row["model"] for row in core], marker_required=True)


def test_core_marker_rate_is_absolute_and_orders_all_three_models():
    records = [
        {
            "model": name, "retain_ufid": 20.0,
            "clean_psnr_db": 28.0, "erased_psnr_db": 27.0,
            "clean_ssim": 0.9, "erased_ssim": 0.89,
            "clean_dfil": 1.0, "retain_dfil": 1.0, "erased_dfil": 1.0,
            "running_time_seconds": 1.0, "running_time_scope": "test-only",
        }
        for name in ("FSCUv4", "retrain_without_client_0", "original")
    ]
    marker_rows = [
        {
            "model": row["model"], "marker_positive_count": count,
            "evaluation_samples": 500, "marker_positive_rate": count / 500,
            "marker_pul_percent": None,
            "marker_pul_status": "original_has_no_detectable_marker",
        }
        for row, count in zip(records, (2, 0, 0))
    ]
    response_rows = [
        {
            "model": row["model"], "samples": 1000,
            "marker_projection_score": value,
            "marker_projection_score_standard_error": 0.001,
            "marker_correlation": value,
            "marker_correlation_standard_error": 0.002,
        }
        for row, value in zip(records, (0.001, 0.0, 0.04))
    ]
    core = _selected_core_metric_rows(records, marker_rows, response_rows)
    assert [row["model"] for row in core] == [
        "original", "retrain_without_client_0", "FSCUv4"
    ]
    assert [row["output_marker_positive_percent"] for row in core] == [0.0, 0.0, 0.4]
    assert [row["output_marker_positive_count"] for row in core] == [0, 0, 2]
    assert all(row["output_marker_evaluation_samples"] == 500 for row in core)
    assert all(row["output_pul_percent"] is None for row in core)
    assert [row["output_marker_projection_mean"] for row in core] == [0.04, 0.0, 0.001]
    assert all(row["output_marker_response_samples"] == 1000 for row in core)
    assert all(row["output_marker_correlation_standard_error"] == 0.002 for row in core)
