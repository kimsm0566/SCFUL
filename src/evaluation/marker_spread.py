"""Frozen, codec-output-blind whole-image marker response audit."""

import math

import numpy as np
import torch
import torch.nn.functional as functional

from src.data.data import build_marker_spread_positive_controls, reconstruction_marker_pattern


def marker_spread_options(overrides=None):
    options = {"sizes": [4, 8, 12], "blur_sigmas": [0.0, 1.0], "calibration_fpr": 0.01}
    options.update(overrides or {})
    if not options["sizes"] or any(int(size) < 4 or int(size) > 32 for size in options["sizes"]):
        raise ValueError("marker spread template sizes must be between 4 and 32")
    if not options["blur_sigmas"] or any(float(sigma) < 0 for sigma in options["blur_sigmas"]):
        raise ValueError("marker spread blur sigmas must be nonnegative")
    if not 0 < float(options["calibration_fpr"]) < 1:
        raise ValueError("marker spread calibration FPR must be in (0, 1)")
    return options


def marker_spread_responses(images, marker_config, options=None):
    """Return signed amplitude/NCC for every valid RGB template window."""
    options = marker_spread_options(options)
    images = images.float()
    template = reconstruction_marker_pattern(
        marker_config, dtype=images.dtype, device=images.device
    ).unsqueeze(0)
    region = marker_config["output_watermark"]
    responses = []
    for size_value in options["sizes"]:
        size = int(size_value)
        resized = functional.interpolate(template, size=(size, size), mode="bilinear", align_corners=False)
        mean = functional.avg_pool2d(images, size, stride=1)
        second = functional.avg_pool2d(images.square(), size, stride=1)
        patch_energy = ((second - mean.square()).sum(dim=1) * size * size).clamp_min(0)
        height, width = patch_energy.shape[-2:]
        y = torch.arange(height, device=images.device)[:, None]
        x = torch.arange(width, device=images.device)[None, :]
        outside = ((y + size <= int(region["top"]))
                   | (y >= int(region["top"]) + int(region["height"]))
                   | (x + size <= int(region["left"]))
                   | (x >= int(region["left"]) + int(region["width"])))
        for sigma_value in options["blur_sigmas"]:
            sigma = float(sigma_value)
            pattern = resized
            if sigma > 0:
                coordinates = torch.arange(-2, 3, device=images.device, dtype=images.dtype)
                gaussian = torch.exp(-coordinates.square() / (2 * sigma * sigma))
                kernel = gaussian[:, None] * gaussian[None, :]
                kernel = (kernel / kernel.sum()).expand(3, 1, 5, 5)
                pattern = functional.conv2d(functional.pad(pattern, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
            signature = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
            energy = signature.square().sum().clamp_min(1e-12)
            dot = functional.conv2d(images, signature).squeeze(1)
            responses.append({
                "size": size, "sigma": sigma, "outside": outside,
                "amplitude": dot / energy,
                "ncc": torch.where(
                    patch_energy > 1e-8,
                    (dot / (patch_energy * energy).sqrt().clamp_min(1e-12)).clamp(-1, 1),
                    torch.zeros_like(dot),
                ),
            })
    return responses


def marker_spread_statistics(responses, image_shape, threshold=None):
    """Aggregate a fixed template bank; coverage is a pixel union, not window sum."""
    height, width = image_shape
    first = responses[0]["amplitude"]
    count = len(first)
    amplitude_map = first.new_zeros((count, height, width))
    ncc_map = torch.zeros_like(amplitude_map)
    coverage = torch.zeros_like(amplitude_map, dtype=torch.bool)
    outside_coverage = torch.zeros_like(coverage)
    peaks, ncc_peaks, outside_peaks, outside_ncc_peaks = [], [], [], []
    outside_squares = first.new_zeros(count)
    outside_windows = 0
    for response in responses:
        amplitude = response["amplitude"].abs()
        ncc = response["ncc"].abs()
        size = response["size"]
        outside = response["outside"]
        peaks.append(amplitude.flatten(1).amax(dim=1))
        ncc_peaks.append(ncc.flatten(1).amax(dim=1))
        if outside.any():
            outside_peaks.append(amplitude[:, outside].amax(dim=1))
            outside_ncc_peaks.append(ncc[:, outside].amax(dim=1))
            outside_squares += amplitude[:, outside].square().sum(dim=1)
            outside_windows += int(outside.sum())
        offset = (size - 1) // 2
        sy, sx = amplitude.shape[-2:]
        target = amplitude_map[:, offset:offset + sy, offset:offset + sx]
        target.copy_(torch.maximum(target, amplitude))
        target = ncc_map[:, offset:offset + sy, offset:offset + sx]
        target.copy_(torch.maximum(target, ncc))
        if threshold is not None:
            detected = amplitude > float(threshold)
            kernel = first.new_ones((1, 1, size, size))
            coverage |= functional.conv_transpose2d(detected[:, None].float(), kernel).squeeze(1) > 0
            outside_coverage |= functional.conv_transpose2d(
                (detected & outside)[..., None, :, :].float(), kernel
            ).squeeze(1) > 0
    zeros = first.new_zeros(count)
    return {
        "global_peak_amplitude": torch.stack(peaks).amax(dim=0),
        "global_peak_ncc": torch.stack(ncc_peaks).amax(dim=0),
        "outside_peak_amplitude": torch.stack(outside_peaks).amax(dim=0) if outside_peaks else zeros,
        "outside_peak_ncc": torch.stack(outside_ncc_peaks).amax(dim=0) if outside_ncc_peaks else zeros,
        "outside_amplitude_rms": (outside_squares / max(outside_windows, 1)).sqrt(),
        "amplitude_map": amplitude_map, "ncc_map": ncc_map,
        "coverage_map": coverage, "outside_coverage_map": outside_coverage,
    }


def _mean_se(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0


def evaluate_marker_spread(outputs, clean_inputs, labels, dataset_indices, evaluation_indices,
                           marker_config, device="cpu", options=None):
    """Evaluate held-out outputs with a threshold fitted only on raw calibration inputs."""
    options = marker_spread_options(options)
    device = torch.device(device)
    images = clean_inputs.to(device).float()
    indices = np.asarray(dataset_indices, dtype=np.int64)
    evaluation_ids = set(int(index) for index in evaluation_indices)
    evaluation_mask = np.asarray([int(index) in evaluation_ids for index in indices])
    calibration_mask = ~evaluation_mask
    if not evaluation_mask.any() or not calibration_mask.any():
        raise ValueError("marker spread requires disjoint nonempty calibration and evaluation inputs")
    if evaluation_ids != set(int(index) for index in indices[evaluation_mask]):
        raise ValueError("marker spread evaluation index is outside the supplied pool")
    controls = marker_spread_statistics(marker_spread_responses(images, marker_config, options), images.shape[-2:])
    control_scores = controls["global_peak_amplitude"].cpu().numpy()
    threshold = float(np.quantile(
        control_scores[calibration_mask], 1 - float(options["calibration_fpr"]), method="higher"
    ))
    positive_images = build_marker_spread_positive_controls(images[evaluation_mask], marker_config)
    positives = marker_spread_statistics(marker_spread_responses(positive_images, marker_config, options), images.shape[-2:])
    positive_scores = positives["global_peak_amplitude"].cpu().numpy()
    region = marker_config["output_watermark"]
    outside_pixels = torch.ones(images.shape[-2:], dtype=torch.bool, device=device)
    top, left = int(region["top"]), int(region["left"])
    outside_pixels[top:top + int(region["height"]), left:left + int(region["width"])] = False
    palette_axis = images.new_tensor(region["first_rgb"]) - images.new_tensor(region["second_rgb"])
    palette_axis = functional.normalize(palette_axis, dim=0)
    selected = torch.as_tensor(np.flatnonzero(evaluation_mask), device=device)
    rows, summary, map_rows = [], [], []
    maps = {"dataset_indices": indices[evaluation_mask].tolist(), "models": {}}
    for name, values in outputs.items():
        decoded = values.to(device).float()
        if decoded.shape != images.shape:
            raise ValueError("marker spread outputs and inputs must align")
        result = marker_spread_statistics(marker_spread_responses(decoded, marker_config, options), images.shape[-2:], threshold)
        residual = decoded - images
        coverage_percent = 100 * result["coverage_map"].float().mean(dim=(-2, -1))
        outside_coverage_percent = 100 * result["outside_coverage_map"][:, outside_pixels].float().mean(dim=1)
        color_bias = (residual * palette_axis[None, :, None, None]).sum(dim=1)[:, outside_pixels].mean(dim=1)
        scalar_fields = [key for key in result if not key.endswith("_map")]
        for position in np.flatnonzero(evaluation_mask):
            row = {"model": name, "sample_index": int(indices[position]), "label": int(labels[position]),
                   **{key: float(result[key][position].cpu()) for key in scalar_fields},
                   "global_marker_detected": bool(result["global_peak_amplitude"][position] > threshold),
                   "coverage_percent": float(coverage_percent[position].cpu()),
                   "outside_coverage_percent": float(outside_coverage_percent[position].cpu()),
                   "outside_color_bias": float(color_bias[position].cpu()), "threshold": threshold}
            rows.append(row)
        model_rows = [row for row in rows if row["model"] == name]
        positives_count = sum(row["global_marker_detected"] for row in model_rows)
        aggregate = {"model": name, "samples": len(model_rows), "global_marker_positive_count": positives_count,
                     "global_mpr_percent": 100 * positives_count / len(model_rows), "threshold": threshold}
        for key in scalar_fields + ["coverage_percent", "outside_coverage_percent", "outside_color_bias"]:
            aggregate[key + "_mean"], aggregate[key + "_standard_error"] = _mean_se([row[key] for row in model_rows])
        summary.append(aggregate)
        mean_amplitude = result["amplitude_map"][selected].mean(dim=0).cpu()
        mean_ncc = result["ncc_map"][selected].mean(dim=0).cpu()
        coverage_frequency = result["coverage_map"][selected].float().mean(dim=0).cpu()
        mean_residual = residual[selected].mean(dim=0).cpu()
        for y in range(images.shape[-2]):
            for x in range(images.shape[-1]):
                map_rows.append({"model": name, "y": y, "x": x,
                                 "amplitude": float(mean_amplitude[y, x]), "ncc": float(mean_ncc[y, x]),
                                 "coverage_frequency": float(coverage_frequency[y, x]),
                                 **{f"residual_{channel}": float(mean_residual[c, y, x]) for c, channel in enumerate("rgb")}})
        maps["models"][name] = {
            "mean_amplitude": mean_amplitude, "mean_ncc": mean_ncc,
            "coverage_frequency": coverage_frequency, "mean_residual": mean_residual,
            "example_output": decoded[selected[0]].cpu(), "example_input": images[selected[0]].cpu(),
            "example_amplitude": result["amplitude_map"][selected[0]].cpu(),
            "example_coverage": result["coverage_map"][selected[0]].cpu(),
        }
    protocol = {
        "version": 2, "scope": "whole-image template-bank audit; not proof of client deletion",
        "active_protocol": marker_config["protocol"],
        "options": options, "threshold": threshold, "threshold_rule": "higher quantile of raw-clean calibration per-image maxima",
        "calibration_indices": indices[calibration_mask].tolist(), "evaluation_indices": indices[evaluation_mask].tolist(),
        "calibration_fpr": float(np.mean(control_scores[calibration_mask] > threshold)),
        "heldout_raw_clean_fpr": float(np.mean(control_scores[evaluation_mask] > threshold)),
        "heldout_shifted_raw_marker_tpr": float(np.mean(positive_scores > threshold)),
        "calibration_uses_codec_outputs": False, "model_order": list(outputs),
        "global_score": "maximum absolute centered projection across all positions, sizes, blur levels and both polarities",
        "ncc": "centered normalized correlation, absolute for global peaks; not an amplitude or probability",
        "outside": "only windows with zero overlap with the configured original marker region",
        "coverage": "union of pixels covered by windows above the global amplitude threshold; no double counting",
        "outside_coverage": "union from wholly outside positive windows divided by number of outside pixels",
        "maps": "absolute response at window centers; boundary positions without valid windows are zero",
        "color_bias": "signed residual along marker color-difference axis outside original region; nonspecific diagnostic",
        "limitations": "template-like spread only; arbitrary diffuse tint may escape matching. Natural texture and utility loss can affect scores. Compare Retrain and raw controls. Zero threshold crossings do not prove zero residual.",
        "positive_control": "known canonical marker inserted at cycling non-original locations in real held-out CIFAR-10 inputs",
    }
    control_rows = []
    shifted_by_index = dict(zip(indices[evaluation_mask].tolist(), positive_scores.tolist()))
    for index, score, is_evaluation in zip(indices, control_scores, evaluation_mask):
        shifted_score = shifted_by_index.get(int(index))
        control_rows.append({
            "sample_index": int(index), "partition": "evaluation" if is_evaluation else "calibration",
            "raw_clean_global_amplitude": float(score), "raw_clean_detected": bool(score > threshold),
            "shifted_marker_global_amplitude": shifted_score,
            "shifted_marker_detected": None if shifted_score is None else bool(shifted_score > threshold),
        })
    return {"summary": summary, "per_sample": rows, "map_rows": map_rows, "protocol": protocol,
            "maps": maps, "controls": control_rows}


def marker_response_report(local_rows, detection_rows, spread_rows, retrain_name,
                           dataset_indices, original_name="original"):
    """Report unscaled amplitude/NCC on the same held-out inputs for every model."""
    evaluation_ids = {int(row["sample_index"]) for row in detection_rows}
    if not evaluation_ids:
        raise ValueError("Marker responses need nonempty held-out detection indices")
    models = list(dict.fromkeys(row["model"] for row in local_rows))
    if original_name not in models or retrain_name not in models:
        raise ValueError("Marker responses require Original and Retrain")
    local = [{"model": row["model"], "scope": "local",
              "sample_index": int(dataset_indices[int(row["sample_index"])]),
              "amplitude": float(row["marker_projection_score"]),
              "ncc": float(row["marker_correlation"])}
             for row in local_rows if int(row["sample_index"]) in evaluation_ids]
    records = list(local)
    for scope in ("global", "outside"):
        for row in spread_rows:
            records.append({"model": row["model"], "scope": scope,
                            "sample_index": int(row["sample_index"]),
                            "amplitude": float(row[scope + "_peak_amplitude"]),
                            "ncc": float(row[scope + "_peak_ncc"])})
    expected_ids = {int(dataset_indices[index]) for index in evaluation_ids}
    summary = []
    for scope in dict.fromkeys(row["scope"] for row in records):
        by_model = {}
        for model in models:
            selected = [r for r in records if r["scope"] == scope and r["model"] == model]
            indexed = {r["sample_index"]: r for r in selected}
            if len(selected) != len(indexed) or set(indexed) != expected_ids:
                raise ValueError("Marker response models/scopes must share identical held-out inputs")
            by_model[model] = indexed
        indices = sorted(expected_ids)
        for metric in ("amplitude", "ncc"):
            arrays = {model: np.asarray([by_model[model][i][metric] for i in indices], dtype=np.float64)
                      for model in models}
            if not all(np.isfinite(values).all() for values in arrays.values()):
                raise ValueError("Marker responses must be finite")
            for model in models:
                mean, se = _mean_se(arrays[model])
                summary.append({"scope": scope, "metric": metric, "model": model,
                                "samples": len(indices), "mean": mean, "standard_error": se})
    protocol = {"version": 2, "model_order": models, "evaluation_indices": sorted(expected_ids),
                "samples": len(expected_ids), "metrics": ["amplitude", "ncc"],
                "reporting": "unscaled measurement means; no Original/Retrain score conversion",
                "reference_rescaling": False, "clipping": False,
                "scope_statistics": {"local": "signed fixed-region amplitude and NCC",
                                     "global": "whole-image maximum absolute amplitude and NCC",
                                     "outside": "maximum absolute response in wholly outside windows"},
                "uncertainty": "standard error of the mean across inputs",
                "interpretation": "measured response, not information remaining or deletion percentage"}
    return {"summary": summary, "per_sample": records, "protocol": protocol}
