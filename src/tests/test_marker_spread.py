"""Real-CIFAR controls for local amplitude, global matching and pixel-union coverage."""

from pathlib import Path

import numpy as np
import torch

from src.data.data import (
    apply_reconstruction_output_watermark, build_marker_spread_positive_controls,
    load_cifar10_test_dataset, reconstruction_marker_pattern, select_cifar10_client_data,
)
from src.evaluation.marker_spread import (
    evaluate_marker_spread, marker_spread_responses, marker_spread_statistics,
)
from src.utils.artifacts import RunArtifacts
from src.utils.plotting import save_marker_spread_figures


def marker_config():
    return {"protocol": "unconditional_output_watermark_v1", "output_watermark": {
        "top": 22, "left": 22, "height": 8, "width": 8, "cell_size": 4,
        "alpha": 1.0, "first_rgb": [1., 0., 1.], "second_rgb": [0., 1., 0.],
    }}


def real_inputs(count=20):
    root = Path(__file__).resolve().parents[2]
    dataset = load_cifar10_test_dataset({"name": "CIFAR10", "data_root": "data/cifar10", "download": False}, root)
    images, labels = select_cifar10_client_data(dataset, np.arange(count))
    images = torch.as_tensor(images, dtype=torch.float32)
    if images.max() > 1:
        images /= 255.
    return images, labels


def test_marker_amplitude_and_ncc_are_distinct_and_shift_is_detected():
    images, _ = real_inputs(2)
    config = marker_config()
    pattern = reconstruction_marker_pattern(config)
    weak = images.clone()
    weak[:, :, 22:30, 22:30] = .5 + .05 * (pattern - .5)
    response = marker_spread_responses(weak, config, {"sizes": [8], "blur_sigmas": [0.]})[0]
    assert torch.allclose(response["amplitude"][:, 22, 22], torch.full((2,), .05), atol=1e-5)
    assert torch.allclose(response["ncc"][:, 22, 22], torch.ones(2), atol=1e-3)
    assert not bool(response["outside"][21, 22])
    assert bool(response["outside"][14, 22])
    constant_patch = images.clone()
    constant_patch[:, :, 22:30, 22:30] = .5
    constant_response = marker_spread_responses(constant_patch, config, {"sizes": [8], "blur_sigmas": [0.]})[0]
    assert torch.equal(constant_response["ncc"][:, 22, 22], torch.zeros(2))
    shifted = build_marker_spread_positive_controls(images[:1], config)
    response = marker_spread_responses(shifted, config, {"sizes": [8], "blur_sigmas": [0.]})[0]
    assert response["outside"][0, 0]
    assert torch.allclose(response["amplitude"][:, 0, 0], torch.ones(1), atol=1e-5)


def test_marker_coverage_is_union_of_pixels_not_sum_of_overlapping_windows():
    amplitude = torch.zeros(1, 29, 29)
    amplitude[0, 0, :2] = 1
    responses = [{"amplitude": amplitude, "ncc": amplitude, "size": 4,
                  "outside": torch.ones(29, 29, dtype=torch.bool)}]
    result = marker_spread_statistics(responses, (32, 32), threshold=.5)
    assert int(result["coverage_map"].sum()) == 20
    assert int(result["outside_coverage_map"].sum()) == 20


def test_marker_spread_raw_calibration_artifacts_and_figures(tmp_path):
    images, labels = real_inputs(20)
    config = marker_config()
    marked = apply_reconstruction_output_watermark(images, config)
    outputs = {"original": marked, "retrain_without_client_0": images, "FSCUv4": images.clone()}
    options = {"sizes": [8], "blur_sigmas": [0.]}
    result = evaluate_marker_spread(outputs, images, labels, np.arange(20), np.arange(0, 20, 2), config, options=options)
    assert result["protocol"]["calibration_uses_codec_outputs"] is False
    assert not set(result["protocol"]["calibration_indices"]) & set(result["protocol"]["evaluation_indices"])
    assert result["summary"][0]["global_mpr_percent"] == 100.
    assert all(0 <= row["coverage_percent_mean"] <= 100 for row in result["summary"])
    changed = dict(outputs, original=images)
    second = evaluate_marker_spread(changed, images, labels, np.arange(20), np.arange(0, 20, 2), config, options=options)
    assert result["protocol"]["threshold"] == second["protocol"]["threshold"]
    artifacts = RunArtifacts(tmp_path)
    artifacts.write_csv("metrics/marker_spread_summary.csv", result["summary"])
    artifacts.write_csv("metrics/marker_spread_maps.csv", result["map_rows"])
    artifacts.write_csv("metrics/marker_spread_per_sample.csv", result["per_sample"])
    artifacts.write_json("metrics/marker_spread_protocol.json", result["protocol"])
    torch.save(result["maps"], artifacts.path("predictions/marker_spread_maps.pt"))
    save_marker_spread_figures(result, config, artifacts.path("figures"))
    for name in ("marker_spread_comparison.png", "marker_spread_maps.png", "marker_spread_examples.png"):
        assert artifacts.path("figures/" + name).stat().st_size > 0
    assert len(result["per_sample"]) == 30
    assert len(result["map_rows"]) == 3 * 32 * 32


def test_marker_spread_default_bank_on_available_cuda():
    if not torch.cuda.is_available():
        return
    images, labels = real_inputs(4)
    outputs = {"original": apply_reconstruction_output_watermark(images, marker_config())}
    result = evaluate_marker_spread(outputs, images, labels, np.arange(4), [0, 2], marker_config(), device="cuda:0")
    assert len(result["summary"]) == 1
    assert result["protocol"]["options"]["sizes"] == [4, 8, 12]
    assert np.isfinite(result["summary"][0]["global_peak_amplitude_mean"])
