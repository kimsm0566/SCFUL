"""완료된 federated run을 paired-AWGN reconstruction protocol로 재평가한다."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data import (
    build_scu_cifar10_backdoor_probe,
    inject_scu_cifar10_backdoor,
    load_cifar10_test_dataset,
    load_cifar10_train_dataset,
    select_cifar10_client_data,
    select_class_balanced_subset,
)
from src.evaluation.backdoor import (
    evaluate_clean_semantic_states,
    evaluate_scu_backdoor_states,
)
from src.evaluation.federated_behavior import evaluate_codec_states
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
    save_retrain_fid_figure,
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
    "clean_fid_to_retrain",
    "erased_fid_to_retrain",
    "model_accuracy",
    "backdoor_test_probe_asr",
    "mia_auc",
    "backdoor_accuracy",
    "running_time_seconds",
    "running_time_scope",
)


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
        _, _, test_indices = select_class_balanced_subset(test_dataset, 10, seed)
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
    states = {path.stem: _load_state(path) for path in checkpoint_paths}
    reference_names = [name for name in states if name.startswith("retrain_without_client_")]
    if len(reference_names) != 1:
        raise ValueError(
            "source run must contain exactly one retrain_without_client checkpoint"
        )

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
            source_config.get("evaluation", {}).get(
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
        extended_records.append(
            {
                "model": model_name,
                "clean_rmse_255": clean["rmse_255"],
                "erased_rmse_255": erased["rmse_255"],
                "clean_mse": clean["mse_input"],
                "clean_psnr_db": clean["psnr"],
                "erased_mse": erased["mse_input"],
                "erased_psnr_db": erased["psnr"],
                "clean_fid_to_retrain": clean["fid_to_retrain"],
                "erased_fid_to_retrain": erased["fid_to_retrain"],
                "clean_retrain_output_gap": clean["fg_output"],
                "erased_retrain_output_gap": erased["fg_output"],
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
    artifacts.write_json(
        config_output,
        {
            "experiment": "reconstruction_channel_audit",
            "observer_run": classifier_metadata.get("observer_run"),
            "source_run": str(source_run),
            "source_config": source_config,
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
                "clean_fid_to_retrain": "FID to retrain reconstructions on the same clean test inputs; lower is closer",
                "erased_fid_to_retrain": "FID to retrain reconstructions on the same erased inputs; lower is closer",
                "model_accuracy": "clean semantic accuracy on decoded test samples",
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
        },
    )
    artifacts.write_csv(per_class_output, metrics["per_class"])
    if metrics["per_client"]:
        artifacts.write_csv(per_client_output, metrics["per_client"])
    artifacts.write_csv(erased_per_class_output, erased_metrics["per_class"])
    artifacts.write_csv("metrics/headline_metrics.csv", headline_records)
    artifacts.write_csv("metrics/retrain_fid.csv", fid_records)
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
    save_unlearning_headline_figure(
        headline_records, artifacts.path("figures"),
        source_config.get("unlearning", {}).get("method_label", "Reconstruction audit"),
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
    artifacts.append_log("completed")
    return artifacts.run_dir
