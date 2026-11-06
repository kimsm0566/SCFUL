"""Tests of scalar reporting records (not replacement image datasets)."""

import csv

import numpy as np
import pytest

from src.evaluation.marker_spread import marker_response_report
from src.experiments.reconstruction_marker_audit import (
    _calibrate_marker_detector_threshold, evaluate_marker_pul,
)
from src.utils.artifacts import RunArtifacts
from src.utils.plotting import save_marker_response_figure


def scalar_records():
    models = ["original", "retrain_without_client_0", "FSCUv4"]
    local, detections, spread = [], [], []
    for model, amplitude, ncc in zip(models, [.6, .1, .3], [.8, .2, .5]):
        for index in range(3):
            local.append({"model": model, "sample_index": index,
                          "marker_projection_score": amplitude, "marker_correlation": ncc})
            if index < 2:
                detections.append({"model": model, "sample_index": index})
                spread.append({"model": model, "sample_index": 10 + index,
                               "global_peak_amplitude": {"original": .4, models[1]: .5, "FSCUv4": .3}[model],
                               "global_peak_ncc": ncc, "outside_peak_amplitude": amplitude,
                               "outside_peak_ncc": ncc})
    return local, detections, spread


def test_clean_max_threshold_does_not_depend_on_positive_control_strength():
    first = _calibrate_marker_detector_threshold([.01, .0734], [1., 1.])
    second = _calibrate_marker_detector_threshold([.01, .0734], [.08, .09])
    assert first["threshold"] == second["threshold"] == .0734
    assert first["balanced_accuracy"] == 1.
    with pytest.raises(ValueError, match="finite"):
        _calibrate_marker_detector_threshold([float("nan")], [1.])


def test_local_threshold_is_raw_only_and_holds_out_evaluation_inputs():
    rows = []
    for model, response in (("original", .1), ("retrain", 0.), ("FSCUv4", .06)):
        for index in range(40):
            rows.append({"model": model, "sample_index": index, "label": index % 10,
                         "marker_projection_score": response,
                         "detector_negative_reference_score": .01 + index * .001,
                         "detector_positive_reference_score": 1.})
    config = {"protocol": "unconditional_output_watermark_v1"}
    summary, detections, protocol = evaluate_marker_pul(rows, config, "original", "retrain", 8)
    calibration_ids = set(protocol["calibration_indices"])
    assert not calibration_ids & set(protocol["evaluation_indices"])
    assert protocol["threshold"] == max(.01 + index * .001 for index in calibration_ids)
    assert protocol["detector_version"] == 2
    assert protocol["calibration_uses_codec_outputs"] is False
    assert len(detections) == 60
    assert summary[0]["marker_positive_rate"] == 1.
    assert summary[1]["marker_positive_rate"] == 0.
    changed = [{**r, "marker_projection_score": -1.} for r in rows]
    _, _, other = evaluate_marker_pul(changed, config, "original", "retrain", 8)
    assert other["threshold"] == protocol["threshold"]


def test_raw_amplitude_ncc_and_same_heldout_inputs(tmp_path):
    local, detections, spread = scalar_records()
    result = marker_response_report(local, detections, spread, "retrain_without_client_0", [10, 11, 12])
    assert len(result["summary"]) == 18
    assert len(result["per_sample"]) == 18
    assert result["protocol"]["evaluation_indices"] == [10, 11]
    assert result["protocol"]["metrics"] == ["amplitude", "ncc"]
    assert result["protocol"]["reference_rescaling"] is False
    assert result["protocol"]["clipping"] is False
    selected = [r for r in result["summary"] if r["scope"] == "local" and r["metric"] == "amplitude"]
    assert [r["model"] for r in selected] == ["original", "retrain_without_client_0", "FSCUv4"]
    assert [r["mean"] for r in selected] == pytest.approx([.6, .1, .3])
    assert all("score" not in row and "score_status" not in row for row in result["summary"])
    assert all(r["samples"] == 2 for r in result["summary"])
    formerly_invalid = [r for r in result["summary"] if r["scope"] == "global" and r["metric"] == "amplitude"]
    assert [r["mean"] for r in formerly_invalid] == pytest.approx([.4, .5, .3])
    artifacts = RunArtifacts(tmp_path)
    artifacts.write_csv("metrics/marker_response_summary.csv", result["summary"])
    save_marker_response_figure(result["summary"], artifacts.path("figures"))
    assert artifacts.path("figures/marker_response_comparison.png").stat().st_size > 0
    with artifacts.path("metrics/marker_response_summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 18


def test_raw_sign_and_full_amplitude_range_are_preserved_and_indices_checked():
    local, detections, spread = scalar_records()
    for row in local:
        if row["model"] == "FSCUv4":
            row["marker_projection_score"] = -.1
            row["marker_correlation"] = 1.
    result = marker_response_report(local, detections, [], "retrain_without_client_0", [10, 11, 12])
    fscu = {r["metric"]: r for r in result["summary"] if r["model"] == "FSCUv4"}
    assert fscu["amplitude"]["mean"] == -.1
    assert fscu["ncc"]["mean"] == 1.
    for row in spread:
        if row["model"] == "FSCUv4":
            row["global_peak_amplitude"] = 1.5
    wide = marker_response_report(local, detections, spread, "retrain_without_client_0", [10, 11, 12])
    assert next(r for r in wide["summary"] if r["model"] == "FSCUv4" and r["scope"] == "global"
                and r["metric"] == "amplitude")["mean"] == pytest.approx(1.5)
    spread[0]["sample_index"] = 999
    with pytest.raises(ValueError, match="identical"):
        marker_response_report(local, detections, spread, "retrain_without_client_0", [10, 11, 12])


def test_equal_and_weak_references_do_not_hide_measurements():
    local, detections, _ = scalar_records()
    for row in local:
        if row["model"] == "original":
            row["marker_projection_score"] = [.3, -.09, .5][row["sample_index"]]
    result = marker_response_report(local, detections, [], "retrain_without_client_0", np.arange(3))
    selected = [r for r in result["summary"] if r["metric"] == "amplitude"]
    assert all(np.isfinite(r["mean"]) and np.isfinite(r["standard_error"]) for r in selected)
    for row in local:
        row["marker_projection_score"] = .1
    equal = marker_response_report(local, detections, [], "retrain_without_client_0", np.arange(3))
    assert [r["mean"] for r in equal["summary"] if r["metric"] == "amplitude"] == pytest.approx([.1, .1, .1])
