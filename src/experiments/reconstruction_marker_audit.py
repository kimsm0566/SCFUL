"""Original-versus-Retrain marker-canary evaluation for reconstruction codecs."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.data.data import (
    apply_reconstruction_input_mask,
    apply_reconstruction_input_key,
    apply_reconstruction_output_watermark,
    reconstruction_marker_evaluation_scenario,
    reconstruction_marker_pattern,
)


MARKER_PUL_PROTOCOLS = frozenset(
    (
        "unconditional_output_watermark_v1",
        "masked_input_marker_reconstruction_v1",
    )
)


def marker_pul_supported(marker_config):
    """Return whether matched-filter Marker-PUL is defined for a protocol."""

    return str(marker_config["protocol"]) in MARKER_PUL_PROTOCOLS


def _balanced_indices(labels, maximum, seed):
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    per_class = maximum // len(classes)
    remainder = maximum % len(classes)
    generator = np.random.default_rng(int(seed))
    selected = []
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        count = per_class + (1 if position < remainder else 0)
        selected.extend(generator.permutation(candidates)[:count].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _region(tensor, marker_config):
    region = marker_config["output_watermark"]
    top = int(region["top"])
    left = int(region["left"])
    height = int(region["height"])
    width = int(region["width"])
    return tensor[:, :, top : top + height, left : left + width]


def _outside_region_squared_error(first, second, marker_config):
    squared = (first - second).square()
    region = marker_config["output_watermark"]
    top = int(region["top"])
    left = int(region["left"])
    height = int(region["height"])
    width = int(region["width"])
    mask = torch.ones_like(squared, dtype=torch.bool)
    mask[:, :, top : top + height, left : left + width] = False
    return squared[mask].view(len(first), -1).mean(dim=1)


def _mean_and_standard_error(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return mean, standard_error


def _psnr_from_mse(mse):
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _marker_detector_scores(images, marker_config):
    """Return frozen known-pattern detector scores in ideal-marker units."""

    observed_region = _region(images, marker_config)
    marker_template = reconstruction_marker_pattern(
        marker_config,
        dtype=observed_region.dtype,
        device=observed_region.device,
    ).unsqueeze(0)
    observed_signature = observed_region - observed_region.mean(
        dim=(-2, -1), keepdim=True
    )
    template_signature = marker_template - marker_template.mean(
        dim=(-2, -1), keepdim=True
    )
    observed_flat = observed_signature.flatten(start_dim=1)
    template_flat = template_signature.flatten(start_dim=1).expand(
        len(observed_region), -1
    )
    denominator = template_flat.square().sum(dim=1).clamp_min(1e-12)
    projection = (observed_flat * template_flat).sum(dim=1) / denominator
    correlation = (
        (observed_flat * template_flat).sum(dim=1)
        / (observed_flat.norm(dim=1) * template_flat.norm(dim=1)).clamp_min(1e-12)
    )
    return projection, correlation


def evaluate_reconstruction_marker_states(
    model_factory,
    model_states,
    test_data,
    marker_config,
    batch_size,
    snr_db,
    device,
    seed,
    collect_outputs=False,
):
    """Measure target-client marker response with paired inputs and AWGN draws."""

    images, labels = test_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    maximum = min(int(marker_config["evaluation_max_examples"]), len(images))
    indices = _balanced_indices(labels.numpy(), maximum, seed)
    images = images[indices]
    labels = labels[indices]
    desired = apply_reconstruction_output_watermark(images, marker_config)
    negative_reference_scores, _ = _marker_detector_scores(images, marker_config)
    positive_reference_scores, _ = _marker_detector_scores(desired, marker_config)
    protocol = str(marker_config["protocol"])
    scenario = reconstruction_marker_evaluation_scenario(protocol)
    if protocol == "keyed_output_watermark_v1":
        probe_inputs = apply_reconstruction_input_key(images, marker_config)
    elif protocol == "masked_input_marker_reconstruction_v1":
        probe_inputs = apply_reconstruction_input_mask(images, marker_config)
    else:
        probe_inputs = images.clone()
    input_conditioned_protocol = protocol in (
        "keyed_output_watermark_v1",
        "masked_input_marker_reconstruction_v1",
    )
    preview_positions = []
    for label in torch.unique(labels, sorted=True):
        position = torch.nonzero(labels == label, as_tuple=False).flatten()[0]
        preview_positions.append(int(position))
    preview_positions_tensor = torch.tensor(
        preview_positions, dtype=torch.long
    )
    previews = {
        "clean_inputs": images[preview_positions_tensor].clone(),
        "probe_inputs": probe_inputs[preview_positions_tensor].clone(),
        "desired_targets": desired[preview_positions_tensor].clone(),
        "labels": labels[preview_positions_tensor].clone(),
        "dataset_indices": torch.as_tensor(
            indices[preview_positions], dtype=torch.long
        ),
    }

    device = torch.device(device)
    models = {}
    for name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = []
    outputs_for_preview = {}
    clean_outputs_for_preview = {}
    complete_outputs = {name: [] for name in model_states} if collect_outputs else None
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            clean = images[start : start + int(batch_size)].to(device)
            probe = probe_inputs[start : start + int(batch_size)].to(device)
            posterior = next(iter(models.values()))(
                clean, mode="posterior"
            )
            standard_noise = torch.randn(
                posterior.mu.shape,
                dtype=posterior.mu.dtype,
                generator=generator,
            )
            for name, model in models.items():
                clean_output = model(
                    clean,
                    mode="paired_awgn",
                    standard_noise=standard_noise,
                    snr_db=float(snr_db),
                ).reconstruction
                if input_conditioned_protocol:
                    probe_output = model(
                        probe,
                        mode="paired_awgn",
                        standard_noise=standard_noise,
                        snr_db=float(snr_db),
                    ).reconstruction
                    clean_latent = model(clean, mode="canonical_latent")
                    probe_latent = model(probe, mode="canonical_latent")
                    input_change = (probe - clean).flatten(start_dim=1)
                    output_change = (probe_output - clean_output).flatten(start_dim=1)
                    latent_change = (probe_latent - clean_latent).flatten(start_dim=1)
                    input_norm = input_change.norm(dim=1).clamp_min(1e-12)
                    output_sensitivity = output_change.norm(dim=1) / input_norm
                    latent_sensitivity = latent_change.norm(dim=1) / input_norm
                else:
                    probe_output = clean_output
                    output_sensitivity = torch.full(
                        (len(clean),), float("nan"), device=device
                    )
                    latent_sensitivity = torch.full(
                        (len(clean),), float("nan"), device=device
                    )

                projection, correlation = _marker_detector_scores(
                    probe_output, marker_config
                )
                if collect_outputs:
                    complete_outputs[name].append(probe_output.detach().cpu())
                desired_probe_region = _region(
                    desired[start : start + len(clean)].to(device), marker_config
                )
                marker_region_mse = (
                    _region(probe_output, marker_config) - desired_probe_region
                ).square().flatten(start_dim=1).mean(dim=1)
                clean_mse = (clean_output - clean).square().flatten(
                    start_dim=1
                ).mean(dim=1)
                probe_mse = (probe_output - desired[start : start + len(clean)].to(device)).square().flatten(
                    start_dim=1
                ).mean(dim=1)
                outside_mse = _outside_region_squared_error(
                    probe_output, clean, marker_config
                )
                for offset in range(len(clean)):
                    rows.append(
                        {
                            "sample_index": int(indices[start + offset]),
                            "label": int(labels[start + offset]),
                            "model": name,
                            "protocol": protocol,
                            "evaluation_scenario_id": scenario["id"],
                            "evaluation_scenario_label_ko": scenario["label_ko"],
                            "marker_projection_score": float(projection[offset].cpu()),
                            "marker_correlation": float(correlation[offset].cpu()),
                            "detector_negative_reference_score": float(
                                negative_reference_scores[start + offset]
                            ),
                            "detector_positive_reference_score": float(
                                positive_reference_scores[start + offset]
                            ),
                            "marker_region_mse_to_desired_response": float(
                                marker_region_mse[offset].cpu()
                            ),
                            "clean_input_mse": float(clean_mse[offset].cpu()),
                            "probe_input_mse_to_desired": float(
                                probe_mse[offset].cpu()
                            ),
                            "outside_watermark_mse": float(outside_mse[offset].cpu()),
                            "output_probe_sensitivity": (
                                None
                                if protocol == "unconditional_output_watermark_v1"
                                else float(output_sensitivity[offset].cpu())
                            ),
                            "latent_probe_sensitivity": (
                                None
                                if protocol == "unconditional_output_watermark_v1"
                                else float(latent_sensitivity[offset].cpu())
                            ),
                            "evaluation_channel": "paired_awgn_posterior_mean",
                            "evaluation_snr_db": float(snr_db),
                        }
                    )
                if name not in outputs_for_preview:
                    outputs_for_preview[name] = torch.empty_like(
                        previews["clean_inputs"]
                    )
                    clean_outputs_for_preview[name] = torch.empty_like(
                        previews["clean_inputs"]
                    )
                for preview_slot, preview_position in enumerate(
                    preview_positions
                ):
                    if start <= preview_position < start + len(clean):
                        outputs_for_preview[name][preview_slot] = (
                            probe_output[preview_position - start].detach().cpu()
                        )
                        clean_outputs_for_preview[name][preview_slot] = (
                            clean_output[preview_position - start].detach().cpu()
                        )

    summary = []
    for name in model_states:
        model_rows = [row for row in rows if row["model"] == name]
        projection_mean, projection_se = _mean_and_standard_error(
            [row["marker_projection_score"] for row in model_rows]
        )
        correlation_mean, correlation_se = _mean_and_standard_error(
            [row["marker_correlation"] for row in model_rows]
        )
        marker_mse = float(
            np.mean(
                [row["marker_region_mse_to_desired_response"] for row in model_rows]
            )
        )
        clean_mse = float(np.mean([row["clean_input_mse"] for row in model_rows]))
        probe_mse = float(
            np.mean([row["probe_input_mse_to_desired"] for row in model_rows])
        )
        outside_mse = float(
            np.mean([row["outside_watermark_mse"] for row in model_rows])
        )
        output_sensitivity_mean = None
        output_sensitivity_se = None
        latent_sensitivity_mean = None
        latent_sensitivity_se = None
        if protocol != "unconditional_output_watermark_v1":
            output_sensitivity_mean, output_sensitivity_se = (
                _mean_and_standard_error(
                    [row["output_probe_sensitivity"] for row in model_rows]
                )
            )
            latent_sensitivity_mean, latent_sensitivity_se = (
                _mean_and_standard_error(
                    [row["latent_probe_sensitivity"] for row in model_rows]
                )
            )
        summary.append(
            {
                "model": name,
                "protocol": protocol,
                "evaluation_scenario_id": scenario["id"],
                "evaluation_scenario_label_ko": scenario["label_ko"],
                "samples": len(model_rows),
                "marker_projection_score": projection_mean,
                "marker_projection_score_standard_error": projection_se,
                "marker_correlation": correlation_mean,
                "marker_correlation_standard_error": correlation_se,
                "marker_region_mse_to_desired_response": marker_mse,
                "clean_rmse_255": 255.0 * math.sqrt(max(clean_mse, 0.0)),
                "clean_psnr_db": _psnr_from_mse(clean_mse),
                "probe_input_rmse_255": 255.0
                * math.sqrt(max(probe_mse, 0.0)),
                "probe_input_psnr_db": _psnr_from_mse(probe_mse),
                "outside_watermark_rmse_255": 255.0
                * math.sqrt(max(outside_mse, 0.0)),
                "output_probe_sensitivity": output_sensitivity_mean,
                "output_probe_sensitivity_standard_error": output_sensitivity_se,
                "latent_probe_sensitivity": latent_sensitivity_mean,
                "latent_probe_sensitivity_standard_error": latent_sensitivity_se,
                "evaluation_channel": "paired_awgn_posterior_mean",
                "evaluation_snr_db": float(snr_db),
            }
        )
    previews["model_outputs"] = outputs_for_preview
    previews["clean_model_outputs"] = clean_outputs_for_preview
    previews["evaluation_scenario"] = scenario
    if collect_outputs:
        previews["spread_inputs"] = images
        previews["spread_labels"] = labels
        previews["spread_indices"] = indices
        previews["spread_outputs"] = {
            name: torch.cat(batches) for name, batches in complete_outputs.items()
        }
    return summary, rows, previews


def marker_reference_gap(per_sample_rows, original_name, retrain_name):
    """Compute a paired first-model-minus-reference marker-score interval."""

    original = {
        int(row["sample_index"]): float(row["marker_projection_score"])
        for row in per_sample_rows
        if row["model"] == original_name
    }
    retrain = {
        int(row["sample_index"]): float(row["marker_projection_score"])
        for row in per_sample_rows
        if row["model"] == retrain_name
    }
    if set(original) != set(retrain) or not original:
        raise ValueError("Original and Retrain marker rows must share non-empty inputs")
    differences = np.asarray(
        [original[index] - retrain[index] for index in sorted(original)],
        dtype=np.float64,
    )
    mean, standard_error = _mean_and_standard_error(differences)
    return {
        "metric": "marker_projection_score",
        "comparison": f"{original_name}_minus_{retrain_name}",
        "samples": int(len(differences)),
        "mean_difference": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - 1.96 * standard_error,
        "ci95_upper": mean + 1.96 * standard_error,
        "fraction_original_greater": float(np.mean(differences > 0.0)),
        "interpretation": (
            f"positive means the marker is stronger in {original_name}"
        ),
    }


def _score_auc(positive_scores, negative_scores):
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("marker detector AUC requires both score groups")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _calibrate_marker_detector_threshold(negative_scores, positive_scores):
    """Freeze the raw-clean maximum; marked controls only measure sensitivity."""

    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if len(negative_scores) == 0 or len(positive_scores) == 0:
        raise ValueError("marker detector calibration needs both reference groups")
    if not np.isfinite(negative_scores).all() or not np.isfinite(positive_scores).all():
        raise ValueError("Marker calibration scores must be finite")
    threshold = float(negative_scores.max())
    true_negative_rate = float(np.mean(negative_scores <= threshold))
    true_positive_rate = float(np.mean(positive_scores > threshold))
    return {
        "threshold": float(threshold),
        "balanced_accuracy": 0.5 * (true_negative_rate + true_positive_rate),
        "true_negative_rate": float(true_negative_rate),
        "true_positive_rate": true_positive_rate,
        "roc_auc": _score_auc(positive_scores, negative_scores),
    }


def evaluate_marker_pul(
    per_sample_rows,
    marker_config,
    original_name,
    retrain_name,
    seed,
    calibration_fraction=0.5,
):
    """Compute PUL using a frozen detector calibrated without codec outputs."""

    if not marker_pul_supported(marker_config):
        raise ValueError(
            "Marker-PUL requires an unconditional or masked-input marker protocol"
        )
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("Marker-PUL calibration_fraction must be in (0, 1)")
    models = list(dict.fromkeys(row["model"] for row in per_sample_rows))
    if original_name not in models or retrain_name not in models:
        raise ValueError("Marker-PUL requires Original and Retrain rows")
    by_model = {
        model: {
            int(row["sample_index"]): row
            for row in per_sample_rows
            if row["model"] == model
        }
        for model in models
    }
    sample_ids = sorted(by_model[original_name])
    if not sample_ids or any(set(by_model[model]) != set(sample_ids) for model in models):
        raise ValueError("Marker-PUL models must share identical nonempty samples")
    generator = np.random.default_rng(int(seed))
    calibration_ids = []
    evaluation_ids = []
    singleton_ids = []
    original_rows = by_model[original_name]
    labels = sorted({int(original_rows[index]["label"]) for index in sample_ids})
    for label in labels:
        candidates = np.asarray(
            [index for index in sample_ids if int(original_rows[index]["label"]) == label],
            dtype=np.int64,
        )
        generator.shuffle(candidates)
        if len(candidates) == 1:
            singleton_ids.extend(candidates.tolist())
            continue
        calibration_count = max(
            1, min(len(candidates) - 1, int(round(len(candidates) * calibration_fraction)))
        )
        calibration_ids.extend(candidates[:calibration_count].tolist())
        evaluation_ids.extend(candidates[calibration_count:].tolist())
    generator.shuffle(singleton_ids)
    desired_calibration_count = max(
        1,
        min(
            len(sample_ids) - 1,
            int(round(len(sample_ids) * calibration_fraction)),
        ),
    )
    singleton_calibration_count = min(
        len(singleton_ids),
        max(0, desired_calibration_count - len(calibration_ids)),
    )
    calibration_ids.extend(singleton_ids[:singleton_calibration_count])
    evaluation_ids.extend(singleton_ids[singleton_calibration_count:])
    if not calibration_ids and evaluation_ids:
        calibration_ids.append(evaluation_ids.pop())
    if not evaluation_ids and calibration_ids:
        evaluation_ids.append(calibration_ids.pop())
    if not calibration_ids or not evaluation_ids:
        raise ValueError("Marker-PUL needs calibration and evaluation samples")
    detector_calibration = _calibrate_marker_detector_threshold(
        [
            original_rows[index]["detector_negative_reference_score"]
            for index in calibration_ids
        ],
        [
            original_rows[index]["detector_positive_reference_score"]
            for index in calibration_ids
        ],
    )
    threshold = detector_calibration["threshold"]
    detection_rows = []
    summary = []
    for model in models:
        model_scores = np.asarray(
            [
                by_model[model][index]["marker_projection_score"]
                for index in evaluation_ids
            ],
            dtype=np.float64,
        )
        detected = model_scores > threshold
        for index, score, is_detected in zip(
            evaluation_ids, model_scores, detected
        ):
            detection_rows.append(
                {
                    "model": model,
                    "sample_index": int(index),
                    "label": int(by_model[model][index]["label"]),
                    "marker_projection_score": float(score),
                    "detection_threshold": threshold,
                    "marker_detected": bool(is_detected),
                    "partition": "held_out_evaluation",
                }
            )
        summary.append(
            {
                "model": model,
                "protocol": str(marker_config["protocol"]),
                "marker_positive_count": int(detected.sum()),
                "evaluation_samples": int(len(detected)),
                "marker_positive_rate": float(detected.mean()),
                "marker_projection_threshold": threshold,
            }
        )
    by_summary = {row["model"]: row for row in summary}
    original_count = int(by_summary[original_name]["marker_positive_count"])
    original_rate = float(by_summary[original_name]["marker_positive_rate"])
    for row in summary:
        row["marker_pul_percent"] = (
            100.0
            * (original_count - int(row["marker_positive_count"]))
            / original_count
            if original_count > 0
            else None
        )
        row["marker_pul_status"] = (
            "ok" if original_count > 0 else "original_has_no_detectable_marker"
        )
        row["reference_model"] = retrain_name
        row["original_model"] = original_name
    detector_auc = _score_auc(
        [
            by_model[original_name][index]["marker_projection_score"]
            for index in evaluation_ids
        ],
        [
            by_model[retrain_name][index]["marker_projection_score"]
            for index in evaluation_ids
        ],
    )
    protocol = {
        "metric": "Marker-PUL",
        "detector_version": 2,
        "applicable_protocols": sorted(MARKER_PUL_PROTOCOLS),
        "active_protocol": str(marker_config["protocol"]),
        "probe_definition": (
            "clean input for the unconditional protocol; marker-region-masked "
            "input for the masked-input protocol"
        ),
        "detector": "frozen spatial-mean-centered known-pattern projection",
        "score": "output marker projection in ideal-marker units",
        "threshold_calibration": (
            "maximum raw-clean calibration score; strict >; marked raw controls "
            "measure sensitivity only; no codec output is used"
        ),
        "detector_calibration": detector_calibration,
        "calibration_fraction_per_class": float(calibration_fraction),
        "threshold": threshold,
        "calibration_samples": int(len(calibration_ids)),
        "evaluation_samples": int(len(evaluation_ids)),
        "calibration_indices": [int(index) for index in calibration_ids],
        "evaluation_indices": [int(index) for index in evaluation_ids],
        "calibration_uses_codec_outputs": False,
        "heldout_raw_clean_false_positive_count": sum(
            float(original_rows[index]["detector_negative_reference_score"]) > threshold
            for index in evaluation_ids
        ),
        "heldout_raw_clean_fpr": float(np.mean([
            float(original_rows[index]["detector_negative_reference_score"]) > threshold
            for index in evaluation_ids
        ])),
        "heldout_raw_marked_tpr": float(np.mean([
            float(original_rows[index]["detector_positive_reference_score"]) > threshold
            for index in evaluation_ids
        ])),
        "original_vs_retrain_heldout_score_auc": detector_auc,
        "seed": int(seed),
        "pul_definition": "100*(Original marker-positive count - model marker-positive count)/Original marker-positive count",
        "warning": "Marker-PUL is non-informative when Original has no held-out marker-positive outputs; utility and Retrain gap remain required",
    }
    return summary, detection_rows, protocol
