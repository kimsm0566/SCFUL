"""완료된 federated run을 paired-AWGN reconstruction protocol로 재평가한다."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data import (
    apply_reconstruction_input_key,
    apply_reconstruction_input_mask,
    build_scu_cifar10_backdoor_probe,
    dataset_labels,
    inject_scu_cifar10_backdoor,
    load_cifar10_test_dataset,
    load_cifar10_train_dataset,
    select_cifar10_client_data,
    select_class_balanced_subset,
    select_class_balanced_pool_subset,
)
from src.evaluation.backdoor import (
    evaluate_clean_semantic_states,
    evaluate_scu_backdoor_states,
)
from src.evaluation.federated_behavior import (
    evaluate_codec_states,
    evaluate_retain_ufid_states,
)
from src.evaluation.dfil import evaluate_encoder_dfil
from src.evaluation.membership_inference import (
    ReconstructionMIAConfig,
    evaluate_reconstruction_membership_inference,
)
from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    SUPPORTED_DEEPJSCC_ARCHITECTURES,
    VariationalDeepJSCC,
    variational_deepjscc_config_for_architecture,
)
from src.utils.artifacts import RunArtifacts, create_run_artifacts
from src.utils.config import load_json_config
from src.utils.fid import InceptionFIDFeatures
from src.utils.reproducibility import collect_environment, derive_seed, set_global_seed
from src.utils.plotting import (
    reconstruction_comparison_sort_key,
    save_encoder_dfil_figure,
    save_encoder_retrain_gap_figure,
    save_marker_pul_figure,
    save_reconstruction_exemplar_figure,
    save_reconstruction_marker_figures,
    save_retain_ufid_figure,
    save_retrain_fid_figure,
    save_selected_core_metrics_figure,
    save_target_information_rate_figure,
    save_unlearning_headline_figure,
    save_output_comparison_figure,
)
from src.utils.semantic_classifier import (
    classifier_state_sha256,
    make_semantic_classifier,
    load_remaining_classifier,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_METRIC_FIELDS = (
    "model",
    "clean_rmse_255",
    "erased_rmse_255",
    "clean_psnr_db",
    "erased_psnr_db",
    "clean_ssim",
    "erased_ssim",
    "retain_ufid",
    "model_accuracy",
    "clean_latent_mean_mse_to_retrain",
    "erased_latent_mean_mse_to_retrain",
    "clean_posterior_symmetric_kl_to_retrain",
    "erased_posterior_symmetric_kl_to_retrain",
    "clean_fid_to_retrain",
    "erased_fid_to_retrain",
    "clean_psnr_to_retrain_db",
    "erased_psnr_to_retrain_db",
    "clean_ssim_to_retrain",
    "erased_ssim_to_retrain",
    "backdoor_test_probe_asr",
    "mia_auc",
    "backdoor_accuracy",
    "running_time_seconds",
    "running_time_scope",
)


def _canonical_model_name(name):
    value = str(name)
    if value.lower() in {"origin", "original"}:
        return "original"
    return value


def _audit_checkpoint_sources(source_run, checkpoint_by_name, selected_names):
    """Validate immutable reference provenance and hash every evaluated checkpoint."""
    from src.utils.fscu_runtime import _canonical_sha256, _file_sha256
    provenance = {"source_run": str(source_run), "checkpoints": {}, "command": sys.argv}
    metadata_path = source_run / "reference_source.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata.get("reused"):
        reference_path = Path(metadata["source_run"])
        manifest = json.loads((reference_path / "reference_manifest.json").read_text())
        fingerprint = manifest.get("scenario_fingerprint_sha256")
        if _canonical_sha256(manifest["scenario"]) != fingerprint or fingerprint != metadata.get("scenario_fingerprint_sha256"):
            raise ValueError("audit reference scenario fingerprint mismatch")
        provenance["reference_fingerprint_sha256"] = fingerprint
    for name in selected_names:
        path = checkpoint_by_name[name]
        digest = _file_sha256(path)
        expected = metadata.get("checkpoint_sha256", {}).get(name)
        if expected is not None and digest != expected:
            raise ValueError(f"audit reference checkpoint hash mismatch: {name}")
        provenance["checkpoints"][name] = {
            "path": str(path.resolve()), "sha256": digest,
            "reference_hash_verified": expected is not None,
        }
    return provenance


def _validate_core_artifacts(run_dir, expected_models, marker_required=False, marker_spread_required=False,
                             marker_response_required=False):
    """Reject incomplete audit artifacts before recording successful completion."""
    root = Path(run_dir)
    required = [
        "metrics/selected_core_metrics.csv", "metrics/selected_core_metrics_protocol.json",
        "metrics/encoder_dfil.csv", "metrics/encoder_dfil_per_sample.csv",
        "metrics/encoder_dfil_protocol.json", "metrics/core_runtime.csv",
        "metrics/retain_ufid.csv", "metrics/retain_ufid_protocol.json",
        "figures/selected_core_metrics.png", "figures/encoder_dfil.png",
    ]
    if marker_required:
        required += ["metrics/reconstruction_marker_pul.csv",
                     "metrics/reconstruction_marker_pul_protocol.json",
                     "metrics/reconstruction_marker_detection_per_sample.csv"]
    if marker_response_required:
        required += ["metrics/marker_response_summary.csv", "metrics/marker_response_per_sample.csv",
                     "metrics/marker_response_protocol.json", "figures/marker_response_comparison.png"]
    if marker_spread_required:
        required += ["metrics/marker_spread_summary.csv", "metrics/marker_spread_per_sample.csv",
                     "metrics/marker_spread_maps.csv", "metrics/marker_spread_protocol.json",
                     "metrics/marker_spread_controls.csv",
                     "predictions/marker_spread_maps.pt", "figures/marker_spread_comparison.png",
                     "figures/marker_spread_maps.png", "figures/marker_spread_examples.png"]
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Incomplete core audit: missing or empty {name}")
    for name in required:
        path = root / name
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f"Incomplete core audit: no metric rows in {name}")
            if name == "metrics/selected_core_metrics.csv":
                if len(rows) != len(expected_models) or {r["model"] for r in rows} != set(expected_models):
                    raise RuntimeError("Incomplete core audit: checkpoint rows do not match")
                for row in rows:
                    for field in ("clean_dfil", "retain_dfil", "erased_dfil",
                                  "clean_psnr_db", "erased_psnr_db", "clean_ssim",
                                  "erased_ssim", "retain_fid_to_real"):
                        if row.get(field) in (None, "") or np.isnan(float(row[field])):
                            raise RuntimeError(f"Incomplete core audit: missing {field}")
                    if marker_required:
                        marker_fields = (
                            "output_marker_positive_percent",
                            "output_marker_positive_count",
                            "output_marker_evaluation_samples",
                        )
                        if any(row.get(field) in (None, "") for field in marker_fields):
                            raise RuntimeError("Incomplete core audit: missing absolute marker detection")
                        count = int(row["output_marker_positive_count"])
                        samples = int(row["output_marker_evaluation_samples"])
                        rate = float(row["output_marker_positive_percent"])
                        if samples <= 0 or not 0 <= count <= samples or not np.isclose(
                            rate, 100.0 * count / samples
                        ):
                            raise RuntimeError("Incomplete core audit: inconsistent absolute marker detection")
        elif path.suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                if not json.load(handle):
                    raise RuntimeError(f"Incomplete core audit: empty protocol {name}")


def _selected_core_metric_rows(extended_records, marker_pul_rows, marker_response_rows=None,
                               marker_spread_rows=None):
    """Select registered metrics plus retained diagnostics in comparison order."""

    pul_by_model = {
        _canonical_model_name(row["model"]): row for row in marker_pul_rows
    }
    response_by_model = {
        _canonical_model_name(row["model"]): row for row in (marker_response_rows or [])
    }
    spread_by_model = {_canonical_model_name(row["model"]): row for row in (marker_spread_rows or [])}
    rows = []
    for record in sorted(
        extended_records,
        key=lambda row: reconstruction_comparison_sort_key(row["model"]),
    ):
        model_name = _canonical_model_name(record["model"])
        pul = pul_by_model.get(model_name)
        response = response_by_model.get(model_name, {})
        spread = spread_by_model.get(model_name, {})
        rows.append(
            {
                "model": record["model"],
                "evaluation_channel": record.get("evaluation_channel"),
                "evaluation_snr_db": record.get("evaluation_snr_db"),
                "output_marker_positive_percent": (
                    None if pul is None else 100.0 * float(pul["marker_positive_rate"])
                ),
                "output_marker_positive_count": (
                    None if pul is None else int(pul["marker_positive_count"])
                ),
                "output_marker_evaluation_samples": (
                    None if pul is None else int(pul["evaluation_samples"])
                ),
                "output_marker_response_samples": response.get("samples"),
                "output_marker_projection_mean": response.get("marker_projection_score"),
                "output_marker_projection_standard_error": response.get(
                    "marker_projection_score_standard_error"
                ),
                "output_marker_correlation_mean": response.get("marker_correlation"),
                "output_marker_correlation_standard_error": response.get(
                    "marker_correlation_standard_error"
                ),
                "output_global_mpr_percent": spread.get("global_mpr_percent"),
                "output_global_marker_positive_count": spread.get("global_marker_positive_count"),
                "output_global_marker_samples": spread.get("samples"),
                "output_global_peak_amplitude_mean": spread.get("global_peak_amplitude_mean"),
                "output_outside_peak_amplitude_mean": spread.get("outside_peak_amplitude_mean"),
                "output_outside_amplitude_rms_mean": spread.get("outside_amplitude_rms_mean"),
                "output_global_peak_ncc_mean": spread.get("global_peak_ncc_mean"),
                "output_outside_peak_ncc_mean": spread.get("outside_peak_ncc_mean"),
                "output_marker_coverage_percent": spread.get("coverage_percent_mean"),
                "output_outside_marker_coverage_percent": spread.get("outside_coverage_percent_mean"),
                "output_pul_percent": (
                    None if pul is None else pul.get("marker_pul_percent")
                ),
                "output_pul_status": (
                    "not_applicable_no_marker_protocol"
                    if pul is None
                    else pul.get("marker_pul_status")
                ),
                "retain_fid_to_real": record["retain_ufid"],
                "clean_psnr_db": record["clean_psnr_db"],
                "erased_psnr_db": record["erased_psnr_db"],
                "clean_ssim": record["clean_ssim"],
                "erased_ssim": record["erased_ssim"],
                **{f"{split}_dfil": record[f"{split}_dfil"]
                   for split in ("clean", "retain", "erased")},
                "running_time_seconds": record["running_time_seconds"],
                "running_time_scope": record["running_time_scope"],
            }
        )
    return rows


def _headline_checkpoint_names(
    available_names,
    reference_name,
    config,
    smoke_test=False,
    reference_only=False,
):
    """Keep FSCUv4 audits to Original, Retrain, and the final method state."""

    available_names = list(available_names)
    if bool(reference_only) or bool(config.get("reference_only", False)):
        selected = ["original", reference_name]
        missing = [name for name in selected if name not in available_names]
        if missing:
            raise ValueError(
                "reference-only audit requires Original and Retrain checkpoints; "
                f"missing={missing}"
            )
        return selected
    unlearning = config.get("unlearning", {})
    method_label = str(unlearning.get("method_label", "FSCUv4"))
    is_fscuv4 = (
        config.get("algorithm") == "FSCUv4"
        or unlearning.get("method_label") == "FSCUv4"
    )
    if not is_fscuv4:
        return available_names
    selected = ["original", reference_name, method_label]
    missing = [name for name in selected if name not in available_names]
    if missing:
        raise ValueError(
            "FSCUv4 audit requires Original, Retrain, and final "
            f"checkpoints; missing={missing}"
        )
    return selected


def _reconstruction_visualization_state_names(states, reference_name, config):
    """Select the paper-facing Original/Retrain/primary-unlearned columns."""

    names = []
    for name in ("original", reference_name):
        if name in states and name not in names:
            names.append(name)
    unlearning = config.get("unlearning", {})
    candidates = (
        unlearning.get("method_label"),
        config.get("algorithm"),
        "FSCUv4",
        "FSCU",
        "scu_self_sharing",
        "scu_forget_only",
    )
    for name in candidates:
        if name in states and name not in names:
            names.append(name)
            break
    if len(names) < 3:
        internal = {
            "initial",
            "original",
            reference_name,
            "client_joint_unlearning",
            "FSCU_last_joint_step",
        }
        for name in states:
            if name not in internal and name not in names and "_R" not in name:
                names.append(name)
                break
    return names


def _evaluate_reconstruction_exemplars(
    model_factory,
    model_states,
    test_data,
    dataset_indices,
    device,
    snr_db,
    seed,
):
    """Decode one deterministic class-balanced row with paired AWGN per model."""

    images, labels = test_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    positions = []
    for label in torch.unique(labels, sorted=True):
        position = torch.nonzero(labels == label, as_tuple=False).flatten()[0]
        positions.append(int(position))
    positions_tensor = torch.tensor(positions, dtype=torch.long)
    selected_images = images[positions_tensor]
    selected_labels = labels[positions_tensor]
    selected_dataset_indices = torch.as_tensor(
        np.asarray(dataset_indices, dtype=np.int64)[positions], dtype=torch.long
    )
    device = torch.device(device)
    model_names = list(model_states)
    first_model = model_factory().to(device)
    first_model.load_state_dict(model_states[model_names[0]])
    first_model.eval()
    batch = selected_images.to(device)
    with torch.no_grad():
        posterior = first_model(batch, mode="posterior")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    standard_noise = torch.randn(
        posterior.mu.shape, dtype=posterior.mu.dtype, generator=generator
    )
    outputs = {}
    for position, name in enumerate(model_names):
        model = first_model if position == 0 else model_factory().to(device)
        if position > 0:
            model.load_state_dict(model_states[name])
            model.eval()
        with torch.no_grad():
            outputs[name] = model(
                batch,
                mode="paired_awgn",
                standard_noise=standard_noise,
                snr_db=float(snr_db),
            ).reconstruction.detach().cpu()
        if position > 0:
            del model
    del first_model
    return {
        "inputs": selected_images.cpu(),
        "labels": selected_labels.cpu(),
        "dataset_indices": selected_dataset_indices,
        "outputs": outputs,
        "protocol": {
            "selection": "first supplied example per represented CIFAR-10 class",
            "channel": "paired_awgn_posterior_mean",
            "snr_db": float(snr_db),
            "seed": int(seed),
            "model_order": model_names,
        },
    }


def _select_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_codec(config):
    codec_values = dict(config["codec"])
    architecture = codec_values.pop("architecture", DEEPJSCC_ARCHITECTURE)
    if architecture not in SUPPORTED_DEEPJSCC_ARCHITECTURES:
        raise ValueError(f"unsupported codec.architecture: {architecture}")
    model_config = variational_deepjscc_config_for_architecture(
        architecture, **codec_values
    )
    return VariationalDeepJSCC(model_config)


def _load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _evaluate_target_information_rates(
    model_factory,
    model_states,
    target_data,
    batch_size,
    device,
    reference_name,
):
    images, labels = target_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    records = []
    for model_name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        values = []
        with torch.no_grad():
            for batch_images, _ in loader:
                posterior = model(batch_images.to(device), mode="posterior")
                per_sample_kl = 0.5 * (
                    posterior.mu.square()
                    + posterior.logvar.exp()
                    - 1.0
                    - posterior.logvar
                ).sum(dim=1)
                values.append(per_sample_kl.detach().cpu())
        per_sample = torch.cat(values)
        records.append(
            {
                "model": model_name,
                "target_prior_kl_mean": float(per_sample.mean()),
                "target_prior_kl_p95": float(torch.quantile(per_sample, 0.95)),
                "target_prior_kl_mean_per_latent": float(
                    per_sample.mean() / model.config.latent_dim
                ),
                "target_examples": int(per_sample.numel()),
            }
        )
        del model
    by_name = {record["model"]: record for record in records}
    original_mean = by_name["original"]["target_prior_kl_mean"]
    retrain_mean = by_name[reference_name]["target_prior_kl_mean"]
    for record in records:
        mean_value = record["target_prior_kl_mean"]
        record["target_prior_kl_ratio_to_original"] = (
            mean_value / max(original_mean, 1e-12)
        )
        record["target_prior_kl_gap_to_retrain"] = abs(
            mean_value - retrain_mean
        )
    return records


def run_reconstruction_channel_audit(
    source_run,
    output_root,
    seed,
    device="auto",
    snr_db=20.0,
    batch_size=None,
    in_place=False,
    smoke_test=False,
    observer_run=None,
    reference_only=False,
):
    """학습을 반복하지 않고 저장된 checkpoint를 CIFAR-10 test set에서 평가한다."""
    if smoke_test and in_place:
        raise ValueError("smoke audit must use a new output root, not overwrite source metrics")
    if observer_run is not None and in_place:
        raise ValueError("changing observer requires a new output root; source metrics are immutable")
    source_run = Path(source_run).resolve()
    source_config = load_json_config(source_run / "config.json")
    if source_config["dataset"]["name"] != "CIFAR10":
        raise ValueError("reconstruction channel audit currently supports CIFAR-10")
    manifest = json.loads(
        (source_run / "split_manifest.json").read_text(encoding="utf-8")
    )
    split_seed = int(manifest["seed"])
    target_client_id = int(source_config["unlearning"]["target_client_id"])
    train_dataset = load_cifar10_train_dataset(
        source_config["dataset"], WORKSPACE_ROOT
    )
    test_dataset = load_cifar10_test_dataset(
        source_config["dataset"], WORKSPACE_ROOT
    )
    test_indices = np.arange(len(test_dataset), dtype=np.int64)
    if smoke_test:
        # Two examples per class are the minimum needed to form independent
        # calibration and evaluation halves for the class-matched MIA.
        _, _, test_indices = select_class_balanced_subset(test_dataset, 2, seed)
    test_data = select_cifar10_client_data(test_dataset, test_indices)
    client_test_indices = [
        np.flatnonzero(np.isin(test_indices, client["test_indices"]))
        for client in manifest["clients"]
    ]
    target_train_indices = np.asarray(
        manifest["clients"][target_client_id]["train_indices"], dtype=np.int64
    )
    erased_data = select_cifar10_client_data(train_dataset, target_train_indices)
    erased_data_protocol = "target_client_full_local_training_data"
    remaining_train_indices = np.asarray(
        [
            index
            for client in manifest["clients"]
            if int(client["client_id"]) != target_client_id
            for index in client["train_indices"]
        ],
        dtype=np.int64,
    )
    if len(remaining_train_indices) == 0:
        raise ValueError("reconstruction audit needs remaining-client training data")
    evaluation_config = source_config.get("evaluation", {})
    retain_ufid_max_examples = int(
        evaluation_config.get("retain_ufid_max_examples", 10000)
    )
    if smoke_test:
        retain_ufid_max_examples = min(retain_ufid_max_examples, 100)
    remaining_labels = dataset_labels(train_dataset)[remaining_train_indices]
    remaining_classes, remaining_class_counts = np.unique(
        remaining_labels, return_counts=True
    )
    retain_examples_per_class = min(
        int(remaining_class_counts.min()),
        int(retain_ufid_max_examples) // len(remaining_classes),
    )
    if retain_examples_per_class <= 0:
        raise ValueError(
            "retain u-FID needs at least one remaining-client example per "
            "represented class"
        )
    retain_ufid_count = int(
        retain_examples_per_class * len(remaining_classes)
    )
    retain_ufid_indices, _ = select_class_balanced_pool_subset(
        dataset_labels(train_dataset),
        remaining_train_indices,
        retain_ufid_count,
        derive_seed(seed, "retain-ufid-index-selection"),
    )
    retain_ufid_data = select_cifar10_client_data(
        train_dataset, retain_ufid_indices
    )
    marker_config = source_config.get("reconstruction_marker")
    marker_metadata = manifest.get("target_client_reconstruction_marker")
    recorded_backdoor = None
    if source_config.get("backdoor") is not None:
        backdoor = source_config["backdoor"]
        recorded_backdoor = manifest.get("target_client_backdoor")
        if recorded_backdoor is None:
            raise ValueError("backdoored source run is missing target_client_backdoor")
        erased_data, reconstructed_backdoor = inject_scu_cifar10_backdoor(
            erased_data,
            poison_ratio=backdoor["poison_ratio"],
            base_label=backdoor["base_label"],
            trigger_label=backdoor["trigger_label"],
            seed=int(recorded_backdoor["seed"]),
        )
        if reconstructed_backdoor["target_local_positions"] != recorded_backdoor[
            "target_local_positions"
        ]:
            raise RuntimeError("could not reproduce target-client backdoor injection")
        erased_data_protocol += "_with_recorded_scu_backdoor_injection"
    checkpoint_paths = sorted((source_run / "checkpoints").glob("*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"source run has no checkpoints: {source_run}")
    checkpoint_by_name = {path.stem: path for path in checkpoint_paths}
    reference_names = [
        name
        for name in checkpoint_by_name
        if name.startswith("retrain_without_client_")
    ]
    if len(reference_names) != 1:
        raise ValueError(
            "source run must contain exactly one retrain_without_client checkpoint"
        )
    selected_names = _headline_checkpoint_names(
        checkpoint_by_name,
        reference_names[0],
        source_config,
        smoke_test=smoke_test,
        reference_only=reference_only,
    )
    checkpoint_sources = _audit_checkpoint_sources(source_run, checkpoint_by_name, selected_names)
    states = {
        name: _load_state(checkpoint_by_name[name])
        for name in selected_names
    }

    device = _select_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(seed, deterministic=True)
    classifier_path = source_run / "semantic_classifier" / "classifier.pt"
    classifier_metadata_path = (
        source_run / "semantic_classifier" / "metadata.json"
    )
    if observer_run is None and not classifier_path.is_file():
        raise ValueError(
            "source run has no separate semantic classifier; legacy joint-head "
            "artifacts must be reported with their original code"
        )
    if observer_run is None and not classifier_metadata_path.is_file():
        raise ValueError("source run has no semantic classifier metadata")
    if observer_run is not None:
        semantic_classifier, classifier_metadata = load_remaining_classifier(
            observer_run, source_config, manifest, device=device, allow_smoke=smoke_test,
        )
    else:
        classifier_metadata = json.loads(classifier_metadata_path.read_text(encoding="utf-8"))
        classifier_state = _load_state(classifier_path)
        if classifier_state_sha256(classifier_state) != classifier_metadata.get("state_sha256"):
            raise ValueError("semantic classifier state SHA-256 mismatch")
        semantic_classifier = make_semantic_classifier(
            {"semantic_classifier": classifier_metadata.get("architecture_config", source_config["semantic_classifier"])},
            state=classifier_state, device=device, frozen=True,
        )
    evaluation_seed = derive_seed(seed, "paired-awgn-reconstruction-evaluation")
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else source_config.get("evaluation", {}).get("batch_size", 100)
    )
    target_information_records = _evaluate_target_information_rates(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        target_data=erased_data,
        batch_size=resolved_batch_size,
        device=device,
        reference_name=reference_names[0],
    )
    target_information_by_model = {
        record["model"]: record for record in target_information_records
    }
    # Audit the actual encoder input, including the recorded masked/keyed rows.
    dfil_erased_images = torch.as_tensor(erased_data[0]).clone()
    if marker_config is not None and marker_metadata is not None:
        positions = marker_metadata.get("marked_local_positions", [])
        if positions:
            if dfil_erased_images.dtype == torch.uint8:
                dfil_erased_images = dfil_erased_images.float() / 255.0
            protocol = marker_config["protocol"]
            if protocol == "masked_input_marker_reconstruction_v1":
                dfil_erased_images[positions] = apply_reconstruction_input_mask(
                    dfil_erased_images[positions], marker_config
                )
            elif protocol == "keyed_output_watermark_v1":
                dfil_erased_images[positions] = apply_reconstruction_input_key(
                    dfil_erased_images[positions], marker_config
                )
    print("Evaluating Gaussian-posterior input dFIL", flush=True)
    dfil_result = evaluate_encoder_dfil(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        split_data={
            "clean_test": (test_data[0], test_indices),
            "retain_train": (retain_ufid_data[0], retain_ufid_indices),
            "erased_train": (dfil_erased_images, target_train_indices),
        },
        device=device,
        probes=2 if smoke_test else evaluation_config.get("dfil_probes", 32),
        max_examples=2 if smoke_test else evaluation_config.get("dfil_max_examples", 128),
        seed=derive_seed(seed, "encoder-dfil-evaluation"),
        smoke_test=smoke_test,
    )
    dfil_result["protocol"]["erased_input"] = (
        "actual recorded masked/keyed target inputs when configured; otherwise erased audit inputs"
    )
    dfil_result["protocol"]["retain_pool"] = (
        "same fixed class-balanced remaining-client clean pool as retain FID"
    )
    dfil_by_model_split = {
        (row["model"], row["split"]): row for row in dfil_result["summary"]
    }
    fid_start = time.perf_counter()
    fid_features = InceptionFIDFeatures(
        device=device,
        batch_size=source_config.get("evaluation", {}).get("fid_batch_size", 32),
    )
    print("Evaluating clean reconstruction metrics + retrain-reference FID", flush=True)
    metrics = evaluate_codec_states(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        reference_name=reference_names[0],
        test_data=test_data,
        client_test_indices=client_test_indices,
        batch_size=resolved_batch_size,
        device=device,
        snr_db=float(snr_db),
        seed=evaluation_seed,
        fid_features=fid_features,
        semantic_classifier=semantic_classifier,
        output_comparison=True,
    )
    print("Evaluating erased reconstruction metrics + retrain-reference FID", flush=True)
    erased_metrics = evaluate_codec_states(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        reference_name=reference_names[0],
        test_data=erased_data,
        client_test_indices=[np.arange(len(erased_data[0]), dtype=np.int64)],
        batch_size=resolved_batch_size,
        device=device,
        snr_db=float(snr_db),
        seed=derive_seed(seed, "paired-awgn-erased-reconstruction-evaluation"),
        fid_features=fid_features,
        semantic_classifier=semantic_classifier,
        output_comparison=True,
    )
    print("Evaluating retain u-FID against real remaining-client images", flush=True)
    retain_ufid_records, retain_ufid_protocol = evaluate_retain_ufid_states(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        retain_data=retain_ufid_data,
        batch_size=resolved_batch_size,
        device=device,
        snr_db=float(snr_db),
        seed=derive_seed(seed, "paired-awgn-retain-ufid-evaluation"),
        fid_features=fid_features,
    )
    retain_ufid_by_model = {
        row["model"]: row for row in retain_ufid_records
    }
    visualization_names = _reconstruction_visualization_state_names(
        states, reference_names[0], source_config
    )
    exemplar_seed = derive_seed(seed, "clean-reconstruction-exemplars")
    reconstruction_exemplars = _evaluate_reconstruction_exemplars(
        model_factory=lambda: _make_codec(source_config),
        model_states={name: states[name] for name in visualization_names},
        test_data=test_data,
        dataset_indices=test_indices,
        device=device,
        snr_db=float(snr_db),
        seed=exemplar_seed,
    )
    reconstruction_fid_wall_seconds = time.perf_counter() - fid_start
    del fid_features
    clean_semantic_records = evaluate_clean_semantic_states(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        semantic_classifier=semantic_classifier,
        clean_test_data=test_data,
        batch_size=resolved_batch_size,
        snr_db=float(snr_db),
        device=device,
        seed=derive_seed(seed, "clean-semantic-evaluation"),
    )

    backdoor_records = []
    backdoor_test_probe_records = []
    if source_config.get("backdoor") is not None:
        backdoor = source_config["backdoor"]
        backdoor_test_probe = build_scu_cifar10_backdoor_probe(
            test_data,
            base_label=backdoor["base_label"],
            trigger_label=backdoor["trigger_label"],
            max_examples=backdoor.get("probe_max_examples"),
        )
        backdoor_test_probe_records = evaluate_scu_backdoor_states(
            model_factory=lambda: _make_codec(source_config),
            model_states=states,
            semantic_classifier=semantic_classifier,
            backdoor_probe=backdoor_test_probe,
            clean_test_data=test_data,
            batch_size=resolved_batch_size,
            snr_db=float(snr_db),
            device=device,
            seed=derive_seed(seed, "scu-backdoor-evaluation"),
        )
        poison_positions = np.asarray(
            recorded_backdoor["target_local_positions"], dtype=np.int64
        )
        erased_images, erased_labels = erased_data
        erased_backdoor_probe = (
            torch.as_tensor(erased_images)[poison_positions],
            torch.as_tensor(erased_labels)[poison_positions],
        )
        backdoor_records = evaluate_scu_backdoor_states(
            model_factory=lambda: _make_codec(source_config),
            model_states=states,
            semantic_classifier=semantic_classifier,
            backdoor_probe=erased_backdoor_probe,
            clean_test_data=test_data,
            batch_size=resolved_batch_size,
            snr_db=float(snr_db),
            device=device,
            seed=derive_seed(seed, "scu-erased-backdoor-evaluation"),
        )

    mia_config = ReconstructionMIAConfig(
        min_examples_per_class=int(
            2
            if smoke_test
            else source_config.get("evaluation", {}).get(
                "mia_min_examples_per_class", 20
            )
        ),
        device=str(device),
    )
    mia_records, mia_per_class, mia_metadata = (
        evaluate_reconstruction_membership_inference(
            model_factory=lambda: _make_codec(source_config),
            model_states=states,
            member_data=erased_data,
            nonmember_data=test_data,
            config=mia_config,
            seed=derive_seed(seed, "scu-extended-reconstruction-mia"),
        )
    )

    clean_by_model = {record["model"]: record for record in metrics["summary"]}
    erased_by_model = {
        record["model"]: record for record in erased_metrics["summary"]
    }
    backdoor_by_model = {record["model"]: record for record in backdoor_records}
    backdoor_test_by_model = {
        record["model"]: record for record in backdoor_test_probe_records
    }
    clean_semantic_by_model = {
        record["model"]: record for record in clean_semantic_records
    }
    mia_by_model = {record["model"]: record for record in mia_records}
    source_efficiency_path = source_run / "metrics/efficiency.json"
    source_efficiency = (
        json.loads(source_efficiency_path.read_text(encoding="utf-8"))
        if source_efficiency_path.exists()
        else {}
    )
    phase_seconds = source_efficiency.get("phase_wall_seconds", {})
    timing_by_model = {
        "original": phase_seconds.get("original_training"),
        reference_names[0]: phase_seconds.get("retrain_without_target"),
        "client_joint_unlearning": phase_seconds.get(
            "target_client_joint_unlearning"
        ),
        "FSCU": source_efficiency.get("proposed_online_wall_seconds"),
        "FSCUv4": source_efficiency.get("proposed_online_wall_seconds"),
        "FSCUv4_NormalRepair": (
            phase_seconds.get("target_client_joint_unlearning", 0.0)
            + phase_seconds.get("retained_repair_normal_repair", 0.0)
        ),
        "FSCUv4_OutputKD": (
            phase_seconds.get("target_client_joint_unlearning", 0.0)
            + phase_seconds.get("retained_repair_output_kd", 0.0)
        ),
        "FedSCU_PubCC": source_efficiency.get("proposed_online_wall_seconds"),
    }
    timing_scope_by_model = {
        "original": "full_original_federated_training_wall_time",
        reference_names[0]: "full_retraining_without_target_wall_time",
        "client_joint_unlearning": "target_client_joint_unlearning_wall_time",
        "FSCU": "joint_unlearning_plus_server_public_compensation_wall_time",
        "FSCUv4": (
            "target_joint_unlearning_plus_conflict_aware_retained_feature_kd_wall_time"
        ),
        "FSCUv4_NormalRepair": (
            "target_joint_unlearning_plus_normal_retained_repair_wall_time"
        ),
        "FSCUv4_OutputKD": (
            "target_joint_unlearning_plus_retained_output_kd_wall_time"
        ),
        "FedSCU_PubCC": "joint_unlearning_plus_server_compensation_wall_time",
    }
    for model_name, seconds in source_efficiency.get(
        "repair_checkpoint_wall_seconds", {}
    ).items():
        if model_name in timing_by_model:
            continue
        timing_by_model[model_name] = seconds
        timing_scope_by_model[model_name] = (
            "target_joint_unlearning_plus_retained_fedavg_repair_wall_time"
        )
    for model_name in states:
        if model_name in timing_by_model:
            continue
        if model_name == "initial":
            timing_by_model[model_name] = 0.0
            timing_scope_by_model[model_name] = "model_initialization_reference"
            continue
        timing_by_model[model_name] = phase_seconds.get("unlearning_path_total")
        timing_scope_by_model[model_name] = (
            "complete_unlearning_branch_wall_time_including_coevaluated_ablations"
        )
    extended_records = []
    for model_name, clean in clean_by_model.items():
        erased = erased_by_model[model_name]
        semantic = backdoor_by_model.get(model_name, {})
        semantic_test = backdoor_test_by_model.get(model_name, {})
        clean_semantic = clean_semantic_by_model[model_name]
        mia = mia_by_model[model_name]
        target_information = target_information_by_model[model_name]
        retain_ufid = retain_ufid_by_model[model_name]
        extended_records.append(
            {
                "model": model_name,
                "clean_rmse_255": clean["rmse_255"],
                "erased_rmse_255": erased["rmse_255"],
                "clean_mse": clean["mse_input"],
                "clean_psnr_db": clean["psnr"],
                "clean_ssim": clean["ssim_input"],
                "erased_mse": erased["mse_input"],
                "erased_psnr_db": erased["psnr"],
                "erased_ssim": erased["ssim_input"],
                "clean_fid_to_retrain": clean["fid_to_retrain"],
                "erased_fid_to_retrain": erased["fid_to_retrain"],
                "retain_ufid": retain_ufid["retain_ufid"],
                **{f"{short}_dfil": dfil_by_model_split[(model_name, split)]["dfil"]
                   for short, split in (("clean", "clean_test"), ("retain", "retain_train"),
                                        ("erased", "erased_train"))},
                "clean_retrain_output_gap": clean["fg_output"],
                "erased_retrain_output_gap": erased["fg_output"],
                "clean_latent_mean_mse_to_retrain": clean["fg_mu"],
                "erased_latent_mean_mse_to_retrain": erased["fg_mu"],
                "clean_posterior_symmetric_kl_to_retrain": clean["fg_latent"],
                "erased_posterior_symmetric_kl_to_retrain": erased["fg_latent"],
                "clean_psnr_to_retrain_db": clean["psnr_to_retrain_db"],
                "erased_psnr_to_retrain_db": erased["psnr_to_retrain_db"],
                "clean_ssim_to_retrain": clean["ssim_to_retrain"],
                "erased_ssim_to_retrain": erased["ssim_to_retrain"],
                "model_accuracy": clean_semantic["clean_semantic_accuracy"],
                "classifier_raw_test_accuracy": classifier_metadata[
                    "raw_cifar10_test_accuracy"
                ],
                # BA and ASR are intentionally different headline metrics.
                # BA uses the target client's actual poisoned erased samples;
                # ASR uses newly triggered clean base-class test samples.
                "backdoor_accuracy": semantic.get("backdoor_asr"),
                "backdoor_test_probe_asr": semantic_test.get("backdoor_asr"),
                "running_time_seconds": timing_by_model.get(model_name),
                "running_time_scope": timing_scope_by_model.get(model_name),
                "mia_auc": mia["mia_auc_low_loss_member"],
                "mia_auc_best_direction": mia["mia_auc_best_direction"],
                "mia_balanced_accuracy": mia["attack_balanced_accuracy"],
                "evaluation_channel": "paired_awgn_posterior_mean",
                "evaluation_snr_db": float(snr_db),
                "mia_score_protocol": "noiseless_posterior_mean_reconstruction_mse",
                "target_prior_kl_mean": target_information[
                    "target_prior_kl_mean"
                ],
                "target_prior_kl_p95": target_information[
                    "target_prior_kl_p95"
                ],
                "target_prior_kl_ratio_to_original": target_information[
                    "target_prior_kl_ratio_to_original"
                ],
                "target_prior_kl_gap_to_retrain": target_information[
                    "target_prior_kl_gap_to_retrain"
                ],
            }
        )
    headline_records = [
        {field: record[field] for field in HEADLINE_METRIC_FIELDS}
        for record in extended_records
    ]
    fid_records = [
        {
            "model": record["model"],
            "split": split_name,
            "fid_to_retrain": record["fid_to_retrain"],
            "reference_model": reference_names[0],
            "samples": record["fid_samples"],
            "feature_dimensions": split_metrics["protocol"]["fid"]["feature_dimensions"],
            "evaluation_channel": "paired_awgn_posterior_mean",
            "evaluation_snr_db": float(snr_db),
            "evaluation_seed": split_metrics["protocol"]["seed"],
            "rank_deficient_sample_count": split_metrics["protocol"]["fid"]["rank_deficient_sample_count"],
            "smoke_test": bool(smoke_test),
        }
        for split_name, split_metrics in (("clean_test", metrics), ("erased_train", erased_metrics))
        for record in split_metrics["summary"]
    ]
    output_records = [
        {"split": split_name, "model": row["model"], "samples": row["samples"],
         "reference_model": reference_names[0], "evaluation_snr_db": float(snr_db),
         "evaluation_channel": "paired_awgn_posterior_mean",
         **{key: row[key] for key in ("ssim_input", "psnr_to_retrain_db", "psnr_to_retrain_capped",
                                     "ssim_to_retrain", "class_agreement_to_retrain",
                                     "probability_js_to_retrain", "logit_mse_to_retrain", "semantic_accuracy")}}
        for split_name, split_metrics in (("clean_test", metrics), ("erased_train", erased_metrics))
        for row in split_metrics["summary"]
    ]

    marker_summary = []
    marker_per_sample = []
    marker_previews = None
    marker_pul_rows = []
    marker_detection_rows = []
    marker_pul_protocol = None
    marker_spread = None
    marker_response = None
    spread_options = dict(evaluation_config.get("marker_spread", {}))
    spread_enabled = bool(
        marker_config is not None
        and marker_config["protocol"] in (
            "unconditional_output_watermark_v1", "masked_input_marker_reconstruction_v1"
        )
        and spread_options.pop("enabled", True)
    )
    if marker_config is not None:
        from src.experiments.reconstruction_marker_audit import (
            evaluate_marker_pul,
            evaluate_reconstruction_marker_states,
            marker_pul_supported,
        )

        marker_summary, marker_per_sample, marker_previews = (
            evaluate_reconstruction_marker_states(
                lambda: _make_codec(source_config),
                states,
                test_data,
                marker_config,
                resolved_batch_size,
                float(snr_db),
                device,
                derive_seed(seed, "registered-marker-pul-evaluation"),
                collect_outputs=spread_enabled,
            )
        )
        if marker_pul_supported(marker_config):
            marker_pul_rows, marker_detection_rows, marker_pul_protocol = (
                evaluate_marker_pul(
                    marker_per_sample,
                    marker_config,
                    "original",
                    reference_names[0],
                    derive_seed(seed, "registered-marker-pul-threshold"),
                )
            )
        if spread_enabled:
            from src.evaluation.marker_spread import evaluate_marker_spread
            print("Evaluating whole-image marker spread (raw-calibrated frozen detector)", flush=True)
            spread_start = time.perf_counter()
            marker_spread = evaluate_marker_spread(
                marker_previews.pop("spread_outputs"), marker_previews.pop("spread_inputs"),
                marker_previews.pop("spread_labels"), test_indices[marker_previews.pop("spread_indices")],
                {int(test_indices[row["sample_index"]]) for row in marker_detection_rows}, marker_config,
                device=device, options=spread_options,
            )
            marker_spread["protocol"].update({
                "channel": "paired_awgn_posterior_mean", "snr_db": float(snr_db),
                "seed": int(seed), "smoke_test": bool(smoke_test),
                "wall_seconds": time.perf_counter() - spread_start,
                "runtime_scope": "offline evaluation, excluded from algorithm Runtime",
            })
        if marker_detection_rows:
            from src.evaluation.marker_spread import marker_response_report
            marker_response = marker_response_report(
                marker_per_sample, marker_detection_rows,
                [] if marker_spread is None else marker_spread["per_sample"],
                reference_names[0], test_indices,
            )
    selected_core_records = _selected_core_metric_rows(
        extended_records, marker_pul_rows, marker_summary,
        None if marker_spread is None else marker_spread["summary"],
    )

    if in_place:
        artifacts = RunArtifacts(source_run)
        config_output = "metrics/headline_metric_config.json"
        summary_output = "metrics/headline_metric_summary.json"
        per_class_output = "metrics/headline_per_class.csv"
        per_client_output = "metrics/headline_per_client.csv"
        erased_per_class_output = "metrics/headline_erased_per_class.csv"
        extended_output = "metrics/headline_extended_metrics.csv"
        actual_backdoor_output = "metrics/scu_backdoor_actual_erased.csv"
        environment_output = "metrics/headline_metric_environment.json"
    else:
        if output_root is None:
            raise ValueError("output_root is required unless in_place=True")
        artifacts = create_run_artifacts(Path(output_root), seed)
        config_output = "config.json"
        summary_output = "metrics/summary.json"
        per_class_output = "metrics/per_class.csv"
        per_client_output = "metrics/per_client.csv"
        erased_per_class_output = "metrics/erased_per_class.csv"
        extended_output = "metrics/scu_extended_metrics.csv"
        actual_backdoor_output = "metrics/scu_backdoor.csv"
        environment_output = "environment.json"
    artifacts.write_json("metrics/audit_checkpoint_sources.json", checkpoint_sources)
    artifacts.write_json(
        config_output,
        {
            "experiment": "reconstruction_channel_audit",
            "observer_run": classifier_metadata.get("observer_run"),
            "source_run": str(source_run),
            "source_config": source_config,
            "marker_local_detector_version": 2,
            "marker_local_threshold_rule": "maximum raw-clean calibration amplitude; strict >",
            "seed": int(seed),
            "batch_size": resolved_batch_size,
            "evaluation_protocol": metrics["protocol"],
            "test_dataset": "class-balanced real CIFAR-10 smoke subset" if smoke_test else "official CIFAR-10 test split",
            "smoke_test": bool(smoke_test),
            "test_indices": test_indices.tolist(),
            "test_examples": int(len(test_data[0])),
            "erased_dataset": erased_data_protocol,
            "erased_examples": int(len(erased_data[0])),
            "target_client_id": target_client_id,
        },
    )
    artifacts.write_json(
        summary_output,
        {
            "source_run": str(source_run),
            "reference_model": reference_names[0],
            "evaluation_protocol": metrics["protocol"],
            "summary": metrics["summary"],
            "erased_summary": erased_metrics["summary"],
            "scu_extended_metrics": extended_records,
            "mse_definition": {
                "input_range": "[0, 1]",
                "reduction": "mean over samples, RGB channels, height, and width",
                "formula_plain_text": "average of (reconstruction - input)^2",
                "headline_rmse_255": "255 * sqrt(normalized MSE)",
                "psnr_peak": 1.0,
                "psnr_from_mse": "10 * log10(1 / MSE)",
            },
            "required_headline_metric_contract": {
                "clean_rmse_255": "0-255 scale RMSE on clean CIFAR-10 test data",
                "erased_rmse_255": "0-255 scale RMSE on the target-client erased data",
                "clean_psnr_db": "PSNR on clean CIFAR-10 test data",
                "erased_psnr_db": "PSNR on the target-client erased data",
                "clean_ssim": "SSIM between clean-test reconstructions and their inputs",
                "erased_ssim": "SSIM between erased-set reconstructions and their inputs",
                "retain_ufid": "FID from remaining-client reconstructions to the corresponding real clean retain inputs; lower is better",
                "model_accuracy": "clean semantic accuracy on decoded test samples",
                "clean_latent_mean_mse_to_retrain": "posterior-mean MSE to Retrain on clean test inputs; lower is closer",
                "erased_latent_mean_mse_to_retrain": "posterior-mean MSE to Retrain on erased inputs; lower is closer",
                "clean_posterior_symmetric_kl_to_retrain": "symmetric diagonal-Gaussian posterior KL to Retrain on clean test inputs; lower is closer",
                "erased_posterior_symmetric_kl_to_retrain": "symmetric diagonal-Gaussian posterior KL to Retrain on erased inputs; lower is closer",
                "clean_fid_to_retrain": "FID to retrain reconstructions on the same clean test inputs; lower is closer",
                "erased_fid_to_retrain": "FID to retrain reconstructions on the same erased inputs; lower is closer",
                "clean_psnr_to_retrain_db": "PSNR between model and Retrain outputs on paired clean inputs; higher is closer",
                "erased_psnr_to_retrain_db": "PSNR between model and Retrain outputs on paired erased inputs; higher is closer",
                "clean_ssim_to_retrain": "SSIM between model and Retrain outputs on paired clean inputs; higher is closer",
                "erased_ssim_to_retrain": "SSIM between model and Retrain outputs on paired erased inputs; higher is closer",
                "backdoor_test_probe_asr": "ASR on newly triggered clean base-class test samples",
                "mia_auc": "reconstruction-loss membership-inference ROC-AUC",
                "backdoor_accuracy": "BA on the target client's actual poisoned erased samples",
                "running_time_seconds": "measured wall time for the named model or unlearning path",
            },
            "scu_public_code_compatibility_note": {
                "paper_term": "mean squared error",
                "paper_clean_data": "held-out clean test dataset",
                "paper_erased_data": "erased training dataset",
                "released_cifar_reporting_reduction": "MSELoss(size_average=False), which is a batch sum",
                "released_cifar_batch_size": 16,
                "decision": "use literal normalized MSE as canonical; do not label batch-dependent SSE as MSE",
            },
            "membership_inference": mia_metadata,
            "semantic_classifier": classifier_metadata,
            "target_information_rate": target_information_records,
            "encoder_dfil": dfil_result["summary"],
            "retain_ufid": retain_ufid_records,
        },
    )
    artifacts.write_csv(per_class_output, metrics["per_class"])
    if metrics["per_client"]:
        artifacts.write_csv(per_client_output, metrics["per_client"])
    artifacts.write_csv(erased_per_class_output, erased_metrics["per_class"])
    artifacts.write_csv("metrics/headline_metrics.csv", headline_records)
    artifacts.write_csv("metrics/selected_core_metrics.csv", selected_core_records)
    artifacts.write_json(
        "metrics/selected_core_metrics_protocol.json",
        {
            "registered_metrics": {
                "reconstruction_utility": ["PSNR", "SSIM", "FID"],
                "marker_removal": ["MPR", "PUL"],
                "efficiency": ["Runtime"],
            },
            "supplementary_metrics": {
                "encoder_input_sensitivity": ["dFIL"],
                "continuous_marker_response": ["marker projection", "marker correlation"],
            },
            "schema_version": 4,
            "comparison_order": [row["model"] for row in selected_core_records],
            "smoke_test": bool(smoke_test),
            "output_evaluation_channel": "paired_awgn_posterior_mean",
            "output_evaluation_snr_db": float(snr_db),
            "encoder_evaluation_channel": "pre_normalization_pre_channel",
            "excluded_from_core": [
                "IDI", "MIA",
                "CKA",
                "client-ID probe",
                "latent mean gap",
                "latent posterior gap",
                "marker projection",
                "latent marker sensitivity",
                "output marker sensitivity",
                "FID-to-Retrain",
                "PSNR-to-Retrain",
                "SSIM-to-Retrain",
            ],
            "fid_definition": (
                "frozen pytorch-fid Inception pool3 FID between remaining-client "
                "reconstructions and real clean remaining-client inputs"
            ),
            "psnr_ssim_definition": (
                "reconstruction versus its ground-truth input, reported separately "
                "on clean test and erased target-client data"
            ),
            "pul_definition": (
                "100 * (Original marker-positive count - model marker-positive "
                "count) / Original marker-positive count"
            ),
            "marker_positive_rate_definition": (
                "MPR (%) = 100 * marker-positive decoded outputs / held-out evaluation inputs; "
                "report counts and sample size for Original, Retrain, and Unlearned"
            ),
            "marker_detection_scope": (
                "MPR is the v2 fixed-region raw-clean-maximum detector. S1/S2 also have independent "
                "whole-image template-bank metrics in marker_spread_protocol.json."
            ),
            "marker_spread_protocol": None if marker_spread is None else "metrics/marker_spread_protocol.json",
            "marker_response_protocol": None if marker_response is None else "metrics/marker_response_protocol.json",
            "continuous_marker_response_definition": (
                "Mean spatially centered projection in ideal-marker units and centered normalized "
                "correlation in the fixed marker region; between-input standard errors of means. Uses all "
                "output_marker_response_samples, unlike MPR's held-out detection subset. "
                "Report Original, Retrain, and Unlearned even if PUL is undefined."
            ),
            "marker_positive_rate_interpretation": (
                "Defined even when Original has zero positives, unlike PUL. Both S1 and S2 "
                "are supported; zero threshold crossings do not establish absence of a weak marker. "
                "This is marker occurrence, not downstream-classifier backdoor accuracy."
            ),
            "dfil_definition": dfil_result["protocol"],
            "interpretation": (
                "PSNR, SSIM, FID, MPR, and PUL observe the full codec output. "
                "dFIL measures input reconstructability of the pre-normalization "
                "Gaussian latent, not membership or proof of client deletion."
            ),
            "cka_used": False,
        },
    )
    artifacts.write_csv("metrics/retrain_fid.csv", fid_records)
    artifacts.write_csv("metrics/retain_ufid.csv", retain_ufid_records)
    artifacts.write_csv(
        "metrics/retain_ufid_indices.csv",
        [
            {
                "dataset_index": int(index),
                "class_id": int(label),
                "source": "remaining_client_clean_train",
            }
            for index, label in zip(retain_ufid_indices, retain_ufid_data[1])
        ],
    )
    artifacts.write_json(
        "metrics/retain_ufid_protocol.json", retain_ufid_protocol
    )
    if marker_summary:
        artifacts.write_csv(
            "metrics/reconstruction_marker_metrics.csv", marker_summary
        )
        artifacts.write_csv(
            "metrics/reconstruction_marker_per_sample.csv", marker_per_sample
        )
        torch.save(
            marker_previews,
            artifacts.path("predictions/reconstruction_marker_previews.pt"),
        )
        save_reconstruction_marker_figures(
            marker_summary,
            marker_previews,
            marker_config,
            artifacts.path("figures"),
        )
    if marker_pul_rows:
        artifacts.write_csv(
            "metrics/reconstruction_marker_pul.csv", marker_pul_rows
        )
        artifacts.write_csv(
            "metrics/reconstruction_marker_detection_per_sample.csv",
            marker_detection_rows,
        )
        artifacts.write_json(
            "metrics/reconstruction_marker_pul_protocol.json",
            marker_pul_protocol,
        )
        save_marker_pul_figure(marker_pul_rows, artifacts.path("figures"))
    if marker_spread is not None:
        from src.utils.plotting import save_marker_spread_figures
        for key, filename in (("summary", "marker_spread_summary.csv"),
                              ("per_sample", "marker_spread_per_sample.csv"),
                              ("map_rows", "marker_spread_maps.csv"),
                              ("controls", "marker_spread_controls.csv")):
            artifacts.write_csv("metrics/" + filename, marker_spread[key])
        artifacts.write_json("metrics/marker_spread_protocol.json", marker_spread["protocol"])
        torch.save(marker_spread["maps"], artifacts.path("predictions/marker_spread_maps.pt"))
        save_marker_spread_figures(marker_spread, marker_config, artifacts.path("figures"))
    if marker_response is not None:
        from src.utils.plotting import save_marker_response_figure
        artifacts.write_csv("metrics/marker_response_summary.csv", marker_response["summary"])
        artifacts.write_csv("metrics/marker_response_per_sample.csv", marker_response["per_sample"])
        marker_response["protocol"].update({"channel": "paired_awgn_posterior_mean", "snr_db": float(snr_db),
                                           "seed": int(seed), "smoke_test": bool(smoke_test)})
        artifacts.write_json("metrics/marker_response_protocol.json", marker_response["protocol"])
        save_marker_response_figure(marker_response["summary"], artifacts.path("figures"))
    artifacts.write_csv("metrics/encoder_dfil.csv", dfil_result["summary"])
    artifacts.write_csv("metrics/encoder_dfil_per_sample.csv", dfil_result["per_sample"])
    artifacts.write_json("metrics/encoder_dfil_protocol.json", dfil_result["protocol"])
    artifacts.write_csv("metrics/core_runtime.csv", [
        {key: row[key] for key in ("model", "running_time_seconds", "running_time_scope")}
        for row in selected_core_records
    ])
    save_encoder_dfil_figure(dfil_result["summary"], artifacts.path("figures"))
    artifacts.write_csv("metrics/retrain_output_semantic_comparison.csv", output_records)
    for split_name, split_metrics, input_indices in (
        ("clean_test", metrics, test_indices), ("erased_train", erased_metrics, target_train_indices),
    ):
        rows = [{**row, "dataset_index": int(input_indices[row["input_position"]])} for row in split_metrics["per_sample"]]
        artifacts.write_csv(f"metrics/{split_name}_output_comparison_per_sample.csv", rows)
        torch.save({"reference_model": reference_names[0], "dataset_indices": input_indices.tolist(),
                    "logits": split_metrics["semantic_logits"]},
                   artifacts.path(f"predictions/{split_name}_semantic_logits.pt"))
    artifacts.write_json("metrics/output_comparison_protocol.json", {
        "clean_test": metrics["protocol"], "erased_train": erased_metrics["protocol"],
        "observer": classifier_metadata, "smoke_test": bool(smoke_test),
        "erased_data_protocol": erased_data_protocol,
        "interpretation": "Output/semantic agreement, not proof of deletion; compare Original-to-Retrain too. Poisoned erased labels are trigger labels, not clean semantic ground truth.",
    })
    torch.save(
        reconstruction_exemplars,
        artifacts.path("predictions/clean_test_reconstruction_exemplars.pt"),
    )
    artifacts.write_json(
        "metrics/reconstruction_exemplar_protocol.json",
        {
            **reconstruction_exemplars["protocol"],
            "dataset_indices": reconstruction_exemplars[
                "dataset_indices"
            ].tolist(),
            "labels": reconstruction_exemplars["labels"].tolist(),
            "tensor_artifact": (
                "predictions/clean_test_reconstruction_exemplars.pt"
            ),
            "figure": "figures/clean_test_reconstruction_exemplars.png",
        },
    )
    save_reconstruction_exemplar_figure(
        reconstruction_exemplars["inputs"],
        reconstruction_exemplars["labels"],
        reconstruction_exemplars["outputs"],
        artifacts.path("figures"),
        filename="clean_test_reconstruction_exemplars.png",
        title=(
            f"Clean CIFAR-10 reconstruction exemplars | paired AWGN "
            f"{float(snr_db):g} dB"
        ),
    )
    save_output_comparison_figure(output_records, artifacts.path("figures"))
    artifacts.write_json("metrics/retrain_fid_protocol.json", {
        "source_run": str(source_run),
        "smoke_test": bool(smoke_test),
        "clean_test": metrics["protocol"],
        "erased_train": erased_metrics["protocol"],
        "erased_data_protocol": erased_data_protocol,
        "wall_seconds": reconstruction_fid_wall_seconds,
        "runtime_scope": "feature-model load + clean/erased reconstruction and FID evaluation; excluded from algorithm Runtime",
        "interpretation": "distributional similarity, not per-image equivalence or proof of deletion; compare Original against Retrain too",
    })
    save_retrain_fid_figure(fid_records, artifacts.path("figures"))
    save_retain_ufid_figure(retain_ufid_records, artifacts.path("figures"))
    save_selected_core_metrics_figure(
        selected_core_records, artifacts.path("figures")
    )
    save_unlearning_headline_figure(
        headline_records, artifacts.path("figures"),
        source_config.get("unlearning", {}).get("method_label", "Reconstruction audit"),
    )
    save_encoder_retrain_gap_figure(
        headline_records, artifacts.path("figures")
    )
    artifacts.write_csv(extended_output, extended_records)
    artifacts.write_csv("metrics/reconstruction_mia_summary.csv", mia_records)
    artifacts.write_csv("metrics/reconstruction_mia_per_class.csv", mia_per_class)
    artifacts.write_csv(
        "metrics/target_information_rate.csv", target_information_records
    )
    save_target_information_rate_figure(
        target_information_records, artifacts.path("figures")
    )
    if backdoor_records:
        artifacts.write_csv(actual_backdoor_output, backdoor_records)
        artifacts.write_csv(
            "metrics/scu_backdoor_test_probe.csv", backdoor_test_probe_records
        )
    artifacts.write_json(
        environment_output, collect_environment(device, WORKSPACE_ROOT)
    )
    artifacts.append_log(f"source_run={source_run}")
    artifacts.append_log(f"smoke_test={bool(smoke_test)}")
    artifacts.append_log(f"test_examples={len(test_data[0])}")
    artifacts.append_log(f"erased_examples={len(erased_data[0])}")
    artifacts.append_log(f"snr_db={float(snr_db)}")
    artifacts.append_log(f"evaluation_seed={evaluation_seed}")
    _validate_core_artifacts(artifacts.run_dir, states, marker_required=bool(marker_pul_rows),
                             marker_spread_required=marker_spread is not None,
                             marker_response_required=marker_response is not None)
    artifacts.append_log("completed")
    return artifacts.run_dir
