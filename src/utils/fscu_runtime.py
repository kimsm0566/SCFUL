"""MPI building blocks used directly by Trainer's public-data FSCU workflow."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    SUPPORTED_DEEPJSCC_ARCHITECTURES,
    VariationalDeepJSCC,
    variational_deepjscc_config_for_architecture,
)
from src.models.semantic_classifier import (
    CIFAR10_CLASSIFIER_ARCHITECTURE,
    Cifar10Classifier,
    Cifar10ClassifierConfig,
)
from src.utils.federated import (
    LocalClassificationConfig,
    LocalReconstructionConfig,
    RetainedFeatureKDConfig,
    average_model_states,
    conflict_aware_average_updates,
    train_local_classification,
    train_local_reconstruction,
    train_local_retained_feature_kd,
    train_local_reconstruction_with_optimizer_state,
)
from src.utils.fscu import (
    extract_erased_representation_buffer,
    forget_target_client,
    module_wise_budgeted_forget_target_client,
)
from src.utils.reproducibility import derive_seed


REFERENCE_FEDERATION_FIELDS = (
    "n_clients",
    "samples_per_client",
    "rounds",
    "local_epochs",
    "persist_client_optimizer_state",
    "serialize_cuda_clients",
    "cuda_client_group_size",
)


def make_fscu_codec(config):
    """Construct the repository's single supported semantic codec."""
    codec_values = dict(config["codec"])
    architecture = codec_values.pop("architecture", DEEPJSCC_ARCHITECTURE)
    if architecture not in SUPPORTED_DEEPJSCC_ARCHITECTURES:
        raise ValueError(
            f"unsupported codec.architecture: {architecture}; supported: "
            f"{', '.join(SUPPORTED_DEEPJSCC_ARCHITECTURES)}"
        )
    model_config = variational_deepjscc_config_for_architecture(
        architecture,
        **codec_values,
    )
    return VariationalDeepJSCC(model_config)


def make_initial_fscu_state(config, seed):
    """Create the rank-0 initial state without consuming the caller's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(seed, "federated-rank-0"))
        return {
            name: value.detach().cpu().clone()
            for name, value in make_fscu_codec(config).state_dict().items()
        }


def make_federated_classifier(config):
    """Construct the classifier selected by an independent training-curve task."""

    values = dict(config["model"])
    kind = values.pop("kind")
    architecture = values.pop(
        "architecture", CIFAR10_CLASSIFIER_ARCHITECTURE
    )
    if kind != "cifar10_classifier":
        raise ValueError(f"unsupported classification model kind: {kind}")
    if architecture != CIFAR10_CLASSIFIER_ARCHITECTURE:
        raise ValueError(
            "classification model architecture must be "
            f"{CIFAR10_CLASSIFIER_ARCHITECTURE}"
        )
    return Cifar10Classifier(Cifar10ClassifierConfig(**values))




def make_initial_federated_classifier_state(config, seed):
    """Create a deterministic classifier state without constructing a codec."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(seed, "federated-classifier-rank-0"))
        return {
            name: value.detach().cpu().clone()
            for name, value in make_federated_classifier(config).state_dict().items()
        }


def select_fscu_device(config):
    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def effective_federated_learning_rate(config, round_number):
    """Use the decayed rate starting at the configured 1-based round."""

    round_number = int(round_number)
    if round_number <= 0:
        raise ValueError("round_number must be positive")
    training = config["training"]
    decay_round = training.get("learning_rate_decay_round")
    if decay_round is not None and round_number >= int(decay_round):
        return float(training["learning_rate_after_decay"])
    return float(training["learning_rate"])


def make_local_training_config(config, device, round_number=1):
    training = config["training"]
    federation = config["federation"]
    kl_reduction = config["codec"].get("kl_reduction", "batchmean_sum")
    default_gradient_clip = 5.0 if kl_reduction == "elementwise_mean" else None
    return LocalReconstructionConfig(
        batch_size=training["batch_size"],
        local_epochs=federation["local_epochs"],
        learning_rate=effective_federated_learning_rate(config, round_number),
        beta=training.get("beta", 0.0),
        snr_db=training["snr_db"],
        reconstruction_weight=training.get("reconstruction_weight", 1.0),
        semantic_weight=training.get("semantic_weight", 0.0),
        augmentation=training.get("augmentation", "none"),
        augmentation_crop_padding=training.get(
            "augmentation_crop_padding", 4
        ),
        augmentation_horizontal_flip_probability=training.get(
            "augmentation_horizontal_flip_probability", 0.5
        ),
        kl_reduction=kl_reduction,
        gradient_clip_norm=training.get(
            "gradient_clip_norm", default_gradient_clip
        ),
        device=str(device),
    )


def make_local_classification_config(config, device):
    """Build one-round raw-image classifier settings."""

    training = config["training"]
    federation = config["federation"]
    return LocalClassificationConfig(
        batch_size=training["batch_size"],
        local_epochs=federation["local_epochs"],
        learning_rate=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
        augmentation=training.get("augmentation", "cifar10_crop_flip_v1"),
        augmentation_crop_padding=training.get(
            "augmentation_crop_padding", 4
        ),
        augmentation_horizontal_flip_probability=training.get(
            "augmentation_horizontal_flip_probability", 0.5
        ),
        preserve_fixed_backdoor_trigger=training.get(
            "preserve_fixed_backdoor_trigger", False
        ),
        device=str(device),
    )




def make_joint_unlearning_config(config, device, epochs=None):
    training = config["training"]
    unlearning = config["unlearning"]
    return {
        "batch_size": int(training["batch_size"]),
        "epochs": int(
            unlearning["forgetting_epochs"] if epochs is None else epochs
        ),
        "learning_rate": float(training["learning_rate"]),
        "snr_db": float(training["snr_db"]),
        "lambda_kl": float(unlearning["lambda_kl"]),
        "lambda_rec": float(unlearning["lambda_rec"]),
        "forgetting_objective": str(
            unlearning.get(
                "forgetting_objective", "posterior_kl_to_original"
            )
        ),
        "target_information_ratio_threshold": float(
            unlearning.get("target_information_ratio_threshold", 0.0)
        ),
        "semantic_weight": float(training.get("semantic_weight", 0.0)),
        "semantic_ascent": bool(unlearning.get("semantic_ascent", True)),
        "lambda_decoder_preservation": float(
            unlearning.get("lambda_decoder_preservation", 0.0)
        ),
        "encoder_unlearning_max_epochs": int(
            unlearning.get(
                "encoder_unlearning_max_epochs",
                unlearning["forgetting_epochs"],
            )
        ),
        "decoder_unlearning_max_epochs": int(
            unlearning.get(
                "decoder_unlearning_max_epochs",
                unlearning["forgetting_epochs"],
            )
        ),
        "encoder_learning_rate": float(
            unlearning.get("encoder_learning_rate", training["learning_rate"])
        ),
        "decoder_learning_rate": float(
            unlearning.get("decoder_learning_rate", training["learning_rate"])
        ),
        "decoder_psnr_drop_db": float(
            unlearning.get("decoder_psnr_drop_db", 2.0)
        ),
        "lambda_encoder_proximal": float(
            unlearning.get("lambda_encoder_proximal", 0.0)
        ),
        "lambda_decoder_proximal": float(
            unlearning.get("lambda_decoder_proximal", 0.0)
        ),
        "device": str(device),
    }


def make_public_compensation_config(config, device, epochs=None):
    training = config["training"]
    unlearning = config["unlearning"]
    return {
        "batch_size": int(training["batch_size"]),
        "epochs": int(
            unlearning["server_cc_epochs"] if epochs is None else epochs
        ),
        "learning_rate": float(unlearning["server_cc_learning_rate"]),
        "temperature": float(unlearning["contrastive_temperature"]),
        "lambda_contrastive": float(unlearning["lambda_contrastive"]),
        "lambda_public_reconstruction": float(
            unlearning["lambda_public_reconstruction"]
        ),
        "lambda_public_prior_kl": float(
            unlearning.get("lambda_public_prior_kl", 0.0)
        ),
        "public_positive_mode": str(
            unlearning.get(
                "public_positive_mode", "deterministic_augmentation"
            )
        ),
        "augmentation_crop_padding": int(
            unlearning.get("augmentation_crop_padding", 4)
        ),
        "augmentation_horizontal_flip_probability": float(
            unlearning.get("augmentation_horizontal_flip_probability", 0.5)
        ),
        "snr_db": float(training["snr_db"]),
        "semantic_weight": float(training.get("semantic_weight", 0.0)),
        "device": str(device),
    }


def make_fscuv4_repair_config(config, device, variant):
    """Build one controlled retained-only repair objective."""
    training = config["training"]
    unlearning = config["unlearning"]
    if variant == "normal_repair":
        output_kd_weight = 0.0
        feature_kd_weight = 0.0
    elif variant == "output_kd":
        output_kd_weight = float(unlearning["lambda_output_kd"])
        feature_kd_weight = 0.0
    elif variant == "feature_kd":
        output_kd_weight = 0.0
        feature_kd_weight = float(unlearning["lambda_feature_kd"])
    else:
        raise ValueError(f"unsupported FSCUv4 repair variant: {variant}")
    return RetainedFeatureKDConfig(
        batch_size=int(training["batch_size"]),
        local_epochs=int(unlearning["local_repair_epochs"]),
        learning_rate=float(unlearning["local_repair_learning_rate"]),
        beta=float(unlearning["lambda_repair_kl"]),
        snr_db=float(training["snr_db"]),
        reconstruction_weight=float(
            unlearning["lambda_repair_reconstruction"]
        ),
        semantic_weight=float(unlearning["lambda_repair_semantic"]),
        output_kd_weight=output_kd_weight,
        feature_kd_weight=feature_kd_weight,
        posterior_mu_kd_weight=float(
            unlearning.get("lambda_posterior_mu_kd", 0.0)
        ),
        posterior_logvar_kd_weight=float(
            unlearning.get("lambda_posterior_logvar_kd", 0.0)
        ),
        marker_outside_consistency_weight=float(
            unlearning.get("lambda_marker_outside_consistency", 0.0)
        ),
        marker_config=config.get("reconstruction_marker"),
        reconstruction_reduction=config["codec"].get(
            "kl_reduction", "batchmean_sum"
        ),
        device=str(device),
    )


def split_interleaved_epoch_budget(total_epochs, cycles):
    """Split an existing total epoch budget across non-empty alternating cycles."""

    total_epochs = int(total_epochs)
    cycles = int(cycles)
    if total_epochs <= 0 or cycles <= 0:
        raise ValueError("epoch budget and cycles must be positive")
    if cycles > total_epochs:
        raise ValueError("interleaving cycles cannot exceed an epoch budget")
    base, remainder = divmod(total_epochs, cycles)
    return [base + int(index < remainder) for index in range(cycles)]


def _evaluate_federated_training_state(
    config,
    device,
    state,
    evaluation_data,
    batch_size,
    seed,
    semantic_classifier=None,
):
    """Evaluate one global state with fixed paired posterior and AWGN draws."""

    images, labels = evaluation_data
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
    model = make_fscu_codec(config).to(device)
    model.load_state_dict(state)
    model.eval()
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    evaluation = config["evaluation"]
    posterior_mode = str(evaluation.get("posterior_mode", "mean"))
    if posterior_mode not in ("mean", "sample"):
        raise ValueError("evaluation.posterior_mode must be mean or sample")
    repetitions = int(evaluation.get("monte_carlo_repetitions", 1))
    if repetitions <= 0:
        raise ValueError("evaluation.monte_carlo_repetitions must be positive")
    latent_dim = int(model.config.latent_dim)
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    posterior_generator = torch.Generator(device=generator_device).manual_seed(
        derive_seed(seed, "federated-evaluation-posterior-epsilon")
    )
    channel_generator = torch.Generator(device=generator_device).manual_seed(
        derive_seed(seed, "federated-evaluation-awgn")
    )

    squared_error = 0.0
    element_count = 0
    correct = 0
    example_count = 0
    posterior_logvar_total = 0.0
    posterior_std_total = 0.0
    posterior_element_count = 0
    prior_kl_total = 0.0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            posterior = model(batch_images, mode="posterior")
            posterior_logvar_total += float(posterior.logvar.sum().cpu())
            posterior_std_total += float(
                posterior.logvar.mul(0.5).exp().sum().cpu()
            )
            posterior_element_count += int(posterior.logvar.numel())
            prior_kl_total += (
                float(model.prior_kl(posterior).cpu()) * len(batch_images)
            )
            for _ in range(repetitions):
                standard_noise = torch.randn(
                    (len(batch_images), latent_dim),
                    dtype=batch_images.dtype,
                    device=device,
                    generator=channel_generator,
                )
                if posterior_mode == "sample":
                    posterior_epsilon = torch.randn(
                        (len(batch_images), latent_dim),
                        dtype=batch_images.dtype,
                        device=device,
                        generator=posterior_generator,
                    )
                    output = model(
                        batch_images,
                        mode="sampled_paired_awgn",
                        posterior_epsilon=posterior_epsilon,
                        standard_noise=standard_noise,
                        snr_db=float(evaluation["snr_db"]),
                    )
                else:
                    output = model(
                        batch_images,
                        mode="paired_awgn",
                        standard_noise=standard_noise,
                        snr_db=float(evaluation["snr_db"]),
                    )
                difference = output.reconstruction - batch_images
                squared_error += float(difference.square().sum().cpu())
                element_count += int(difference.numel())
                if semantic_classifier is not None:
                    logits = semantic_classifier(output.reconstruction)
                    correct += int(
                        (logits.argmax(dim=1) == batch_labels).sum().item()
                    )
            example_count += len(batch_labels)

    normalized_mse = squared_error / element_count
    metrics = {
        "test_normalized_mse": normalized_mse,
        "test_rmse_255": 255.0 * math.sqrt(normalized_mse),
        "test_psnr_db": 10.0 * math.log10(
            1.0 / max(normalized_mse, 1e-12)
        ),
        "test_posterior_logvar_mean": (
            posterior_logvar_total / posterior_element_count
        ),
        "test_posterior_std_mean": (
            posterior_std_total / posterior_element_count
        ),
        "test_prior_kl_loss": prior_kl_total / example_count,
        "test_examples": example_count,
        "test_snr_db": float(evaluation["snr_db"]),
        "test_posterior_mode": posterior_mode,
        "test_monte_carlo_repetitions": repetitions,
    }
    if semantic_classifier is not None:
        metrics["test_accuracy"] = correct / (example_count * repetitions)
    return metrics


def run_federated_phase(
    comm,
    config,
    device,
    initial_state,
    active_client_ids,
    client_data,
    phase_name,
    seed,
    semantic_classifier=None,
    capture_latest_client_states=False,
    evaluation_data=None,
    evaluation_interval=0,
    evaluation_batch_size=None,
    round_reporter=None,
    aggregation_weight_multipliers=None,
):
    """Run one original or retrain FedAvg phase on all MPI ranks."""
    rank = comm.Get_rank()
    client_id = rank - 1
    global_state = initial_state if rank == 0 else None
    history = []
    latest_client_states = {} if rank == 0 and capture_latest_client_states else None
    latest_round_global_state = None
    optimizer_state = None
    persist_optimizer = bool(
        config["federation"]["persist_client_optimizer_state"]
    )
    if (
        semantic_classifier is None
        and float(config["training"].get("semantic_weight", 0.0)) > 0
    ):
        raise ValueError(
            "semantic_classifier is required when semantic_weight is positive"
        )
    serialize = bool(config["federation"].get("serialize_cuda_clients", False))
    aggregation_weight_multipliers = {
        int(client): float(multiplier)
        for client, multiplier in (aggregation_weight_multipliers or {}).items()
    }
    if set(aggregation_weight_multipliers) - set(active_client_ids):
        raise ValueError("aggregation multiplier references an inactive client")
    if any(
        not math.isfinite(multiplier) or multiplier <= 0.0
        for multiplier in aggregation_weight_multipliers.values()
    ):
        raise ValueError("aggregation multipliers must be finite and positive")

    if rank == 0 and evaluation_data is not None:
        if int(evaluation_interval) <= 0:
            raise ValueError("evaluation_interval must be positive")
        initial_metrics = _evaluate_federated_training_state(
            config,
            device,
            global_state,
            evaluation_data,
            evaluation_batch_size or config["evaluation"]["batch_size"],
            derive_seed(seed, f"{phase_name}:paired-evaluation"),
            semantic_classifier,
        )
        initial_row = {
            "phase": phase_name,
            "round": 0,
            "learning_rate": None,
            "objective_loss": None,
            "reconstruction_loss": None,
            "train_mse": None,
            "kl_loss": None,
            "semantic_loss": None,
            "semantic_accuracy": None,
            "semantic_weight": float(
                config["training"].get("semantic_weight", 0.0)
            ),
            "reconstruction_weight": float(
                config["training"].get("reconstruction_weight", 1.0)
            ),
            "beta": float(config["training"].get("beta", 0.0)),
            **initial_metrics,
            "persist_client_optimizer_state": persist_optimizer,
            "optimizer_state_reused": False,
        }
        history.append(initial_row)

    for round_id in range(int(config["federation"]["rounds"])):
        round_number = round_id + 1
        local_config = make_local_training_config(
            config,
            device,
            round_number=round_number,
        )
        if (
            rank == 0
            and capture_latest_client_states
            and round_id == int(config["federation"]["rounds"]) - 1
        ):
            latest_round_global_state = {
                name: value.detach().cpu().clone()
                for name, value in global_state.items()
            }
        global_state = comm.bcast(global_state, root=0)
        payload = None

        def train_current_client():
            nonlocal optimizer_state
            model = make_fscu_codec(config)
            model.load_state_dict(global_state)
            local_seed = derive_seed(
                seed, f"{phase_name}:client-{client_id}:round-{round_id}"
            )
            if persist_optimizer:
                state, metrics, optimizer_state = (
                    train_local_reconstruction_with_optimizer_state(
                        model,
                        client_data,
                        local_config,
                        local_seed,
                        optimizer_state=optimizer_state,
                        semantic_classifier=semantic_classifier,
                    )
                )
            else:
                state, metrics = train_local_reconstruction(
                    model,
                    client_data,
                    local_config,
                    local_seed,
                    semantic_classifier=semantic_classifier,
                )
            return {"client_id": client_id, "state": state, "metrics": metrics}

        if serialize:
            scheduled = sorted(active_client_ids)
            group_size = int(config["federation"]["cuda_client_group_size"])
            for start in range(0, len(scheduled), group_size):
                group = scheduled[start : start + group_size]
                if rank > 0 and client_id in group:
                    payload = train_current_client()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                comm.Barrier()
        elif rank > 0 and client_id in active_client_ids:
            payload = train_current_client()

        received = comm.gather(payload, root=0)
        if rank == 0:
            active_payloads = [item for item in received if item is not None]
            if capture_latest_client_states:
                for item in active_payloads:
                    latest_client_states[int(item["client_id"])] = item["state"]
            aggregation_weights = [
                float(item["metrics"]["examples"])
                * aggregation_weight_multipliers.get(int(item["client_id"]), 1.0)
                for item in active_payloads
            ]
            global_state = average_model_states(
                [item["state"] for item in active_payloads], aggregation_weights
            )
            sample_weights = np.asarray(
                aggregation_weights,
                dtype=np.float64,
            )

            def weighted_metric(name):
                values = [item["metrics"].get(name) for item in active_payloads]
                if any(value is None for value in values):
                    return None
                return float(np.average(values, weights=sample_weights))

            history_row = {
                "phase": phase_name,
                "round": round_number,
                "learning_rate": float(local_config.learning_rate),
                "objective_loss": weighted_metric("objective_loss"),
                "reconstruction_loss": weighted_metric("reconstruction_loss"),
                "train_mse": weighted_metric("train_mse"),
                "kl_loss": weighted_metric("kl_loss"),
                "semantic_loss": weighted_metric("semantic_loss"),
                "semantic_accuracy": weighted_metric("semantic_accuracy"),
                "semantic_weight": float(
                    config["training"].get("semantic_weight", 0.0)
                ),
                "reconstruction_weight": float(
                    config["training"].get("reconstruction_weight", 1.0)
                ),
                "beta": float(config["training"].get("beta", 0.0)),
                "persist_client_optimizer_state": persist_optimizer,
                "optimizer_state_reused": bool(
                    round_id > 0 and persist_optimizer
                ),
                "aggregation_weight_multiplier_max": max(
                    aggregation_weight_multipliers.values(), default=1.0
                ),
            }
            should_evaluate = (
                evaluation_data is not None
                and (
                    round_number % int(evaluation_interval) == 0
                    or round_number == int(config["federation"]["rounds"])
                )
            )
            if should_evaluate:
                history_row.update(
                    _evaluate_federated_training_state(
                        config,
                        device,
                        global_state,
                        evaluation_data,
                        evaluation_batch_size
                        or config["evaluation"]["batch_size"],
                        derive_seed(seed, f"{phase_name}:paired-evaluation"),
                        semantic_classifier,
                    )
                )
            history.append(history_row)
            if round_reporter is not None:
                round_reporter(
                    format_federated_round_log(
                        history_row,
                        int(config["federation"]["rounds"]),
                    )
                )
    if rank == 0:
        if capture_latest_client_states:
            missing = sorted(set(active_client_ids) - set(latest_client_states))
            if missing:
                raise RuntimeError(
                    f"federated phase did not capture latest clients: {missing}"
                )
            return global_state, history, {
                "round": int(config["federation"]["rounds"]),
                "global_base_state": latest_round_global_state,
                "client_states": latest_client_states,
            }
        return global_state, history
    if capture_latest_client_states:
        return None, None, None
    return None, None


def _evaluate_federated_classification_state(
    config,
    device,
    state,
    evaluation_data,
    batch_size,
):
    """Evaluate an independent classifier directly on clean CIFAR-10 images."""

    images, labels = evaluation_data
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
    model = make_federated_classifier(config).to(device)
    model.load_state_dict(state)
    model.eval()
    cross_entropy_total = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_images)
            cross_entropy_total += float(
                functional.cross_entropy(
                    logits, batch_labels, reduction="sum"
                ).cpu()
            )
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            examples += len(batch_labels)
    return {
        "test_cross_entropy": cross_entropy_total / examples,
        "test_accuracy": correct / examples,
        "test_examples": examples,
    }


def run_federated_classification_phase(
    comm,
    config,
    device,
    initial_state,
    active_client_ids,
    client_data,
    phase_name,
    seed,
    evaluation_data=None,
    evaluation_interval=0,
    evaluation_batch_size=None,
    round_reporter=None,
    capture_latest_client_states=False,
    clean_client_data=None,
    backdoor_metadata=None,
):
    """Run classifier-only FedAvg with a fresh local Adam every round."""

    rank = comm.Get_rank()
    client_id = rank - 1
    global_state = initial_state if rank == 0 else None
    local_config = make_local_classification_config(config, device)
    history = []
    serialize = bool(config["federation"].get("serialize_cuda_clients", False))
    latest_round_global_state = None
    latest_client_states = {}

    if rank == 0 and evaluation_data is not None:
        if int(evaluation_interval) <= 0:
            raise ValueError("evaluation_interval must be positive")
        history.append(
            {
                "phase": phase_name,
                "round": 0,
                "train_cross_entropy": None,
                "train_accuracy": None,
                **_evaluate_federated_classification_state(
                    config,
                    device,
                    global_state,
                    evaluation_data,
                    evaluation_batch_size
                    or config["evaluation"]["batch_size"],
                ),
            }
        )

    for round_id in range(int(config["federation"]["rounds"])):
        if (
            rank == 0
            and capture_latest_client_states
            and round_id + 1 == int(config["federation"]["rounds"])
        ):
            latest_round_global_state = {
                name: value.detach().cpu().clone()
                for name, value in global_state.items()
            }
        global_state = comm.bcast(global_state, root=0)
        payload = None

        def train_current_client():
            model = make_federated_classifier(config)
            model.load_state_dict(global_state)
            local_seed = derive_seed(
                seed, f"{phase_name}:client-{client_id}:round-{round_id}"
            )
            state, metrics = train_local_classification(
                model,
                client_data,
                local_config,
                local_seed,
                clean_client_data=clean_client_data,
                fixed_trigger_positions=(
                    backdoor_metadata["target_local_positions"]
                    if backdoor_metadata is not None
                    else None
                ),
            )
            return {"client_id": client_id, "state": state, "metrics": metrics}

        if serialize:
            scheduled = sorted(active_client_ids)
            group_size = int(config["federation"]["cuda_client_group_size"])
            for start in range(0, len(scheduled), group_size):
                group = scheduled[start : start + group_size]
                if rank > 0 and client_id in group:
                    payload = train_current_client()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                comm.Barrier()
        elif rank > 0 and client_id in active_client_ids:
            payload = train_current_client()

        received = comm.gather(payload, root=0)
        if rank == 0:
            active_payloads = [item for item in received if item is not None]
            if (
                capture_latest_client_states
                and round_id + 1 == int(config["federation"]["rounds"])
            ):
                latest_client_states = {
                    int(item["client_id"]): {
                        name: value.detach().cpu().clone()
                        for name, value in item["state"].items()
                    }
                    for item in active_payloads
                }
            global_state = average_model_states(
                [item["state"] for item in active_payloads],
                [item["metrics"]["examples"] for item in active_payloads],
            )
            sample_weights = np.asarray(
                [item["metrics"]["examples"] for item in active_payloads],
                dtype=np.float64,
            )
            history_row = {
                "phase": phase_name,
                "round": round_id + 1,
                "train_cross_entropy": float(
                    np.average(
                        [
                            item["metrics"]["cross_entropy"]
                            for item in active_payloads
                        ],
                        weights=sample_weights,
                    )
                ),
                "train_accuracy": float(
                    np.average(
                        [item["metrics"]["accuracy"] for item in active_payloads],
                        weights=sample_weights,
                    )
                ),
            }
            should_evaluate = (
                evaluation_data is not None
                and (
                    (round_id + 1) % int(evaluation_interval) == 0
                    or round_id + 1 == int(config["federation"]["rounds"])
                )
            )
            if should_evaluate:
                history_row.update(
                    _evaluate_federated_classification_state(
                        config,
                        device,
                        global_state,
                        evaluation_data,
                        evaluation_batch_size
                        or config["evaluation"]["batch_size"],
                    )
                )
            history.append(history_row)
            if round_reporter is not None:
                round_reporter(
                    format_federated_classification_round_log(
                        history_row,
                        int(config["federation"]["rounds"]),
                    )
                )
    if rank == 0:
        if capture_latest_client_states:
            missing = sorted(set(active_client_ids) - set(latest_client_states))
            if missing:
                raise RuntimeError(
                    "classification phase did not capture latest clients: "
                    f"{missing}"
                )
            return global_state, history, {
                "round": int(config["federation"]["rounds"]),
                "global_base_state": latest_round_global_state,
                "client_states": latest_client_states,
            }
        return global_state, history
    if capture_latest_client_states:
        return None, None, None
    return None, None


def format_federated_classification_round_log(history_row, total_rounds):
    """Format classifier-only CE and accuracy without codec metrics."""

    def number(value):
        return "n/a" if value is None else f"{float(value):.6f}"

    def percentage(value):
        return "n/a" if value is None else f"{100.0 * float(value):.2f}%"

    fields = [
        f"[{history_row['phase']}:classification]",
        f"round={int(history_row['round'])}/{int(total_rounds)}",
        f"train_ce={number(history_row.get('train_cross_entropy'))}",
        f"train_accuracy={percentage(history_row.get('train_accuracy'))}",
    ]
    if history_row.get("test_cross_entropy") is not None:
        fields.extend(
            [
                f"test_ce={number(history_row['test_cross_entropy'])}",
                f"test_accuracy={percentage(history_row['test_accuracy'])}",
            ]
        )
    return " ".join(fields)


def format_federated_round_log(history_row, total_rounds):
    """Format one concise server-side training progress line."""

    def number(value):
        return "n/a" if value is None else f"{float(value):.6f}"

    train_mse = history_row.get("train_mse")
    if train_mse is None and "train_mse" not in history_row:
        train_mse = history_row.get("reconstruction_loss")
    train_psnr = None
    if train_mse is not None:
        train_psnr = 10.0 * math.log10(1.0 / max(float(train_mse), 1e-12))
    fields = [
        f"[{history_row['phase']}]",
        f"round={int(history_row['round'])}/{int(total_rounds)}",
        f"learning_rate={number(history_row.get('learning_rate'))}",
        f"train_mse={number(train_mse)}",
        f"train_psnr_db={number(train_psnr)}",
        f"objective_loss={number(history_row.get('objective_loss'))}",
        f"reconstruction_loss={number(history_row.get('reconstruction_loss'))}",
        f"kl_loss={number(history_row.get('kl_loss'))}",
    ]
    train_accuracy = history_row.get("semantic_accuracy")
    accuracy_text = (
        "n/a"
        if train_accuracy is None
        else f"{100.0 * float(train_accuracy):.2f}%"
    )
    fields.extend(
        [
            f"train_accuracy={accuracy_text}",
            f"semantic_loss={number(history_row.get('semantic_loss'))}",
        ]
    )
    if history_row.get("test_normalized_mse") is not None:
        fields.extend(
            [
                f"test_mse={number(history_row['test_normalized_mse'])}",
                f"test_psnr_db={number(history_row.get('test_psnr_db'))}",
                "test_accuracy="
                + (
                    "n/a"
                    if history_row.get("test_accuracy") is None
                    else f"{100.0 * float(history_row['test_accuracy']):.2f}%"
                ),
                "test_posterior_std_mean="
                f"{number(history_row.get('test_posterior_std_mean'))}",
                "test_posterior_logvar_mean="
                f"{number(history_row.get('test_posterior_logvar_mean'))}",
                "test_prior_kl_loss="
                f"{number(history_row.get('test_prior_kl_loss'))}",
            ]
        )
    return " ".join(fields)


def run_target_joint_unlearning(
    comm,
    config,
    device,
    learner_state,
    fixed_original_state,
    client_data,
    seed,
    cycle_id=0,
    epochs=None,
    extract_buffer=True,
    buffer_source="refresh",
    optimizer_state=None,
    semantic_classifier=None,
):
    """Run one target JU step and optionally refresh its canonical latent buffer."""
    rank = comm.Get_rank()
    client_id = rank - 1
    target_client_id = int(config["unlearning"]["target_client_id"])
    learner_state = comm.bcast(
        learner_state if rank == 0 else None, root=0
    )
    payload = None
    next_optimizer_state = optimizer_state
    if rank > 0 and client_id == target_client_id:
        model = make_fscu_codec(config)
        model.load_state_dict(learner_state)
        fixed_model = make_fscu_codec(config)
        fixed_model.load_state_dict(fixed_original_state)
        joint_config = make_joint_unlearning_config(
            config, device, epochs=epochs
        )
        joint_seed = derive_seed(
            seed, f"target-scu-joint-unlearning:cycle-{cycle_id}"
        )
        if joint_config["forgetting_objective"] == "module_wise_budgeted":
            if optimizer_state is not None:
                raise ValueError(
                    "module-wise budgeted unlearning does not reuse a joint "
                    "optimizer state"
                )
            joint_state, metrics = module_wise_budgeted_forget_target_client(
                model,
                client_data,
                fixed_model,
                joint_config,
                joint_seed,
            )
            next_optimizer_state = None
        else:
            joint_state, metrics, next_optimizer_state = forget_target_client(
                model,
                client_data,
                fixed_model,
                joint_config,
                joint_seed,
                optimizer_state=optimizer_state,
                return_optimizer_state=True,
                semantic_classifier=semantic_classifier,
            )
        erased_buffer = None
        if extract_buffer:
            buffer_model = make_fscu_codec(config)
            if buffer_source == "original":
                buffer_model.load_state_dict(fixed_original_state)
            elif buffer_source in ("post_ju_static", "refresh"):
                buffer_model.load_state_dict(joint_state)
            else:
                raise ValueError(
                    f"unsupported negative buffer mode: {buffer_source}"
                )
            erased_buffer = extract_erased_representation_buffer(
                buffer_model,
                client_data,
                batch_size=int(config["training"]["batch_size"]),
                device=device,
                max_examples=int(
                    config["unlearning"]["erased_buffer_max_examples"]
                ),
                seed=derive_seed(seed, f"erased-buffer:cycle-{cycle_id}"),
                sampling_mode=str(
                    config["unlearning"].get(
                        "erased_buffer_sampling", "stratified"
                    )
                ),
            )
        buffer_bytes = (
            int(erased_buffer.numel() * erased_buffer.element_size())
            if erased_buffer is not None
            else 0
        )
        metrics.update(
            {
                "cycle": int(cycle_id + 1),
                "erased_buffer_refreshed": bool(extract_buffer),
                "erased_buffer_examples": (
                    len(erased_buffer) if erased_buffer is not None else 0
                ),
                "erased_buffer_latent_dim": (
                    int(erased_buffer.shape[1])
                    if erased_buffer is not None
                    else int(config["codec"]["latent_dim"])
                ),
                "erased_buffer_bytes": buffer_bytes,
                "negative_buffer_mode": str(buffer_source),
            }
        )
        payload = {
            "state": joint_state,
            "erased_buffer": erased_buffer,
            "metrics": metrics,
        }
    received = comm.gather(payload, root=0)
    if rank != 0:
        return None, None, None, next_optimizer_state
    target_payload = received[target_client_id + 1]
    if target_payload is None:
        raise RuntimeError("target client did not return its FSCU update")
    return (
        target_payload["state"],
        target_payload["erased_buffer"],
        target_payload["metrics"],
        None,
    )


def run_fscuv4_retained_repair(
    comm,
    config,
    device,
    joint_state,
    original_teacher_state,
    client_data,
    seed,
    variant,
    semantic_classifier=None,
    evaluation_data=None,
    evaluation_batch_size=None,
    round_reporter=None,
):
    """Run retained-client repair and preserve configured budget checkpoints."""
    rank = comm.Get_rank()
    client_id = rank - 1
    num_clients = int(config["federation"]["n_clients"])
    target_client_id = int(config["unlearning"]["target_client_id"])
    retained_client_ids = set(range(num_clients)) - {target_client_id}
    rounds = int(config["unlearning"]["distributed_repair_rounds"])
    local_config = make_fscuv4_repair_config(config, device, variant)
    teacher_state = (
        None if variant == "normal_repair" else original_teacher_state
    )
    current_state = joint_state if rank == 0 else None
    history = []
    checkpoint_rounds = {
        int(value)
        for value in config["unlearning"].get(
            "repair_checkpoint_rounds", [rounds]
        )
    }
    if any(value <= 0 or value > rounds for value in checkpoint_rounds):
        raise ValueError("repair checkpoint rounds must be within the repair budget")
    checkpoint_rounds.add(rounds)
    checkpoint_states = {}
    serialize = bool(config["federation"].get("serialize_cuda_clients", False))
    repair_start = time.perf_counter()

    for round_id in range(rounds):
        current_state = comm.bcast(
            current_state if rank == 0 else None, root=0
        )
        payload = None

        def repair_current_client():
            learner = make_fscu_codec(config)
            learner.load_state_dict(current_state)
            teacher = None
            if teacher_state is not None:
                teacher = make_fscu_codec(config)
                teacher.load_state_dict(teacher_state)
            local_seed = derive_seed(
                seed,
                f"fscuv4-retained-repair:client-{client_id}:round-{round_id}",
            )
            state, metrics = train_local_retained_feature_kd(
                learner,
                teacher,
                client_data,
                local_config,
                local_seed,
                semantic_classifier=semantic_classifier,
            )
            return {
                "client_id": int(client_id),
                "state": state,
                "metrics": metrics,
            }

        if serialize:
            scheduled = sorted(retained_client_ids)
            group_size = int(config["federation"]["cuda_client_group_size"])
            for start in range(0, len(scheduled), group_size):
                group = scheduled[start : start + group_size]
                if rank > 0 and client_id in group:
                    payload = repair_current_client()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                comm.Barrier()
        elif rank > 0 and client_id in retained_client_ids:
            payload = repair_current_client()

        received = comm.gather(payload, root=0)
        if rank == 0:
            retained_payloads = [item for item in received if item is not None]
            retained_payloads.sort(key=lambda item: item["client_id"])
            if [item["client_id"] for item in retained_payloads] != sorted(
                retained_client_ids
            ):
                raise RuntimeError(
                    "FSCUv4 did not receive every retained-client repair update"
                )
            aggregation = str(config["unlearning"]["repair_aggregation"])
            if aggregation == "conflict_aware_projection":
                current_state, aggregation_metrics = (
                    conflict_aware_average_updates(
                        current_state,
                        [item["state"] for item in retained_payloads],
                        [item["metrics"]["examples"] for item in retained_payloads],
                        original_teacher_state,
                        joint_state,
                        epsilon=float(
                            config["unlearning"].get("conflict_epsilon", 1e-12)
                        ),
                    )
                )
                for item, diagnostic in zip(
                    retained_payloads,
                    aggregation_metrics["client_diagnostics"],
                ):
                    diagnostic["client_id"] = int(item["client_id"])
            else:
                current_state = average_model_states(
                    [item["state"] for item in retained_payloads],
                    [item["metrics"]["examples"] for item in retained_payloads],
                )
                aggregation_metrics = {
                    "aggregation": "sample_weighted_fedavg",
                    "conflict_reference": None,
                    "projection_scope": None,
                    "forgetting_update_norm": None,
                    "aggregate_repair_update_norm": None,
                    "aggregate_cosine_after": None,
                    "conflict_client_count": None,
                    "conflict_client_fraction": None,
                    "client_diagnostics": [],
                }
            numeric_names = (
                "reconstruction_loss",
                "prior_kl_loss",
                "semantic_loss",
                "output_kd_loss",
                "encoder_feature_kd_loss",
                "decoder_feature_kd_loss",
                "posterior_mu_kd_loss",
                "posterior_logvar_kd_loss",
                "marker_outside_consistency_loss",
                "total_loss",
            )
            round_metrics = {
                name: float(
                    np.mean(
                        [item["metrics"][name] for item in retained_payloads]
                    )
                )
                for name in numeric_names
            }
            round_metrics.update(
                {
                    "round": int(round_id + 1),
                    "variant": variant,
                    "retained_clients": sorted(retained_client_ids),
                    "client_update_steps": int(
                        sum(
                            item["metrics"]["steps"]
                            for item in retained_payloads
                        )
                    ),
                    "examples": int(
                        sum(
                            item["metrics"]["examples"]
                            for item in retained_payloads
                        )
                    ),
                    "output_kd_weight": float(local_config.output_kd_weight),
                    "feature_kd_weight": float(local_config.feature_kd_weight),
                    "posterior_mu_kd_weight": float(
                        local_config.posterior_mu_kd_weight
                    ),
                    "posterior_logvar_kd_weight": float(
                        local_config.posterior_logvar_kd_weight
                    ),
                    "marker_outside_consistency_weight": float(
                        local_config.marker_outside_consistency_weight
                    ),
                    "kl_weight": float(local_config.beta),
                    "beta_weighted_kl_loss": float(
                        local_config.beta * round_metrics["prior_kl_loss"]
                    ),
                    "teacher_scope": (
                        "none"
                        if variant == "normal_repair"
                        else "original_global_on_each_client_local_retained_data_only"
                    ),
                    "channel_pairing": (
                        "standard_training_awgn"
                        if variant == "normal_repair"
                        else "same_standard_awgn_draw"
                    ),
                    "aggregation": aggregation,
                    "aggregation_diagnostics": aggregation_metrics,
                    "cumulative_wall_seconds": float(
                        time.perf_counter() - repair_start
                    ),
                }
            )
            if round_id + 1 in checkpoint_rounds:
                checkpoint_states[int(round_id + 1)] = {
                    name: value.detach().cpu().clone()
                    for name, value in current_state.items()
                }
                if evaluation_data is not None:
                    round_metrics.update(
                        _evaluate_federated_training_state(
                            config,
                            device,
                            current_state,
                            evaluation_data,
                            evaluation_batch_size
                            or config["evaluation"]["batch_size"],
                            derive_seed(
                                seed,
                                f"fscuv4-repair-evaluation-round-{round_id + 1}",
                            ),
                            semantic_classifier=semantic_classifier,
                        )
                    )
            history.append(round_metrics)
            if round_reporter is not None:
                test_psnr = round_metrics.get("test_psnr_db")
                round_reporter(
                    "[retained_repair] "
                    f"round={round_id + 1}/{rounds} "
                    f"variant={variant} aggregation={aggregation} "
                    f"reconstruction_loss={round_metrics['reconstruction_loss']:.6f} "
                    f"kl_loss={round_metrics['prior_kl_loss']:.6f} "
                    f"objective_loss={round_metrics['total_loss']:.6f} "
                    + (
                        f"test_psnr_db={test_psnr:.6f}"
                        if test_psnr is not None
                        else "test_psnr_db=n/a"
                    )
                )

    if rank == 0:
        return current_state, {
            "variant": variant,
            "rounds": rounds,
            "history": history,
            "client_update_steps": int(
                sum(record["client_update_steps"] for record in history)
            ),
            "retained_clients": sorted(retained_client_ids),
            "output_kd_weight": float(local_config.output_kd_weight),
            "feature_kd_weight": float(local_config.feature_kd_weight),
            "posterior_mu_kd_weight": float(
                local_config.posterior_mu_kd_weight
            ),
            "posterior_logvar_kd_weight": float(
                local_config.posterior_logvar_kd_weight
            ),
            "marker_outside_consistency_weight": float(
                local_config.marker_outside_consistency_weight
            ),
            "teacher_scope": (
                "none"
                if variant == "normal_repair"
                else "original_global_on_each_client_local_retained_data_only"
            ),
            "uses_public_data": False,
            "uses_target_data_during_repair": False,
            "aggregation": str(config["unlearning"]["repair_aggregation"]),
            "conflict_reference": (
                "target_joint_unlearning_update"
                if config["unlearning"]["repair_aggregation"]
                == "conflict_aware_projection"
                else None
            ),
        }, checkpoint_states
    return None, None, None




def model_state_nbytes(state):
    return int(
        sum(value.numel() * value.element_size() for value in state.values())
    )


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"required reference file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reference JSON at {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"reference JSON root must be an object: {path}")
    return value


def _resolve_reference_run(reference_run):
    path = Path(reference_run).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"reference run directory does not exist: {path}")
    if (path / "config.json").is_file():
        return path
    candidates = sorted(
        child for child in path.iterdir()
        if child.is_dir() and (child / "config.json").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            "reference path must be a completed run directory or contain exactly "
            f"one completed run: {path}"
        )
    return candidates[0]


def _state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _reference_scenario(config, split_manifest, initial_state, seed):
    federation = config["federation"]
    if config.get("task") == "classification":
        return {
            "schema_version": 4,
            "task": "classification",
            "global_seed": int(seed),
            "initial_state_seed": int(
                derive_seed(seed, "federated-classifier-rank-0")
            ),
            "initial_state_sha256": _state_sha256(initial_state),
            "deterministic": bool(config.get("deterministic", True)),
            "dataset": config["dataset"],
            "public_data": config["public_data"],
            "model": config["model"],
            "federation": {
                name: federation[name]
                for name in REFERENCE_FEDERATION_FIELDS
                if name in federation
            },
            "split": config["split"],
            "training": config["training"],
            "target_client_id": int(config["unlearning"]["target_client_id"]),
            "backdoor": config.get("backdoor"),
            "optimizer": "Adam",
            "aggregation": "sample_weighted_fedavg",
            "split_manifest": split_manifest,
        }
    scenario = {
        "schema_version": 3,
        "global_seed": int(seed),
        "initial_state_seed": int(derive_seed(seed, "federated-rank-0")),
        "initial_state_sha256": _state_sha256(initial_state),
        "deterministic": bool(config.get("deterministic", True)),
        "dataset": config["dataset"],
        "public_data": config["public_data"],
        "codec": config["codec"],
        "semantic_classifier": config["semantic_classifier"],
        "federation": {
            name: federation[name]
            for name in REFERENCE_FEDERATION_FIELDS
            if name in federation
        },
        "split": config["split"],
        "training": config["training"],
        "target_client_id": int(config["unlearning"]["target_client_id"]),
        "backdoor": config.get("backdoor"),
        "optimizer": "Adam",
        "aggregation": "sample_weighted_fedavg",
        "split_manifest": split_manifest,
    }
    if config.get("reconstruction_marker") is not None:
        scenario["reconstruction_marker"] = config["reconstruction_marker"]
        scenario["aggregation"] = (
            "sample_weighted_fedavg_with_target_marker_multiplier"
        )
    return scenario


def _first_difference(expected, actual, path="scenario"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path} keys differ (missing={missing}, extra={extra})"
        for key in sorted(expected):
            difference = _first_difference(
                expected[key], actual[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length differs ({len(expected)} != {len(actual)})"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path} differs ({expected!r} != {actual!r})"
    return None


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"required reference checkpoint does not exist: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"reference checkpoint is not a model state dictionary: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"reference checkpoint contains non-tensor state: {path}")
    return state


def _validate_checkpoint_state(state, expected_state, name):
    difference = _first_difference(
        sorted(expected_state), sorted(state), f"checkpoint.{name}.keys"
    )
    if difference is not None:
        raise ValueError(difference)
    for key, expected_value in expected_state.items():
        value = state[key]
        if value.shape != expected_value.shape:
            raise ValueError(
                f"checkpoint.{name}.{key} shape differs "
                f"({tuple(value.shape)} != {tuple(expected_value.shape)})"
            )
        if value.dtype != expected_value.dtype:
            raise ValueError(
                f"checkpoint.{name}.{key} dtype differs "
                f"({value.dtype} != {expected_value.dtype})"
            )




def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_fscu_reference_manifest(
    run_dir,
    config,
    split_manifest,
    initial_state,
    seed,
    training_wall_seconds,
    latest_client_models=None,
):
    """Build the immutable manifest for a newly trained FSCU reference."""
    run_dir = Path(run_dir).resolve()
    target_client_id = int(config["unlearning"]["target_client_id"])
    checkpoint_names = (
        "initial",
        "original",
        f"retrain_without_client_{target_client_id}",
    )
    checkpoints = {}
    for name in checkpoint_names:
        path = run_dir / "checkpoints" / f"{name}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"reference checkpoint was not saved: {path}")
        checkpoints[name] = {
            "relative_path": str(path.relative_to(run_dir)),
            "sha256": _file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
    scenario = _reference_scenario(
        config, split_manifest, initial_state, seed
    )
    manifest = {
        "schema_version": int(scenario["schema_version"]),
        "immutable_reference": True,
        "global_seed": int(seed),
        "target_client_id": target_client_id,
        "scenario_fingerprint_sha256": _canonical_sha256(scenario),
        "scenario": scenario,
        "checkpoints": checkpoints,
        "training_wall_seconds": {
            "original": float(training_wall_seconds["original_training"]),
            "retrain_without_target": float(
                training_wall_seconds["retrain_without_target"]
            ),
        },
    }
    if latest_client_models is not None:
        manifest["latest_client_models"] = latest_client_models
    return manifest


def load_fscu_reference_models(
    reference_run, config, split_manifest, initial_state, seed
):
    """Validate and load reusable Initial, Original, and Retrain states."""
    run_dir = _resolve_reference_run(reference_run)
    source_config = _read_json(run_dir / "config.json")
    source_metadata = _read_json(run_dir / "run_metadata.json")
    source_split = _read_json(run_dir / "split_manifest.json")
    reference_manifest_path = run_dir / "reference_manifest.json"
    reference_manifest = _read_json(reference_manifest_path)

    if int(source_metadata.get("global_seed", -1)) != int(seed):
        raise ValueError(
            "reference seed differs "
            f"({source_metadata.get('global_seed')!r} != {int(seed)})"
        )

    manifest_scenario = reference_manifest.get("scenario")
    if not isinstance(manifest_scenario, dict):
        raise ValueError("reference_manifest.json is missing scenario")
    manifest_fingerprint = reference_manifest.get("scenario_fingerprint_sha256")
    if _canonical_sha256(manifest_scenario) != manifest_fingerprint:
        raise ValueError("reference scenario fingerprint is invalid")

    loaded_initial = _load_checkpoint(run_dir / "checkpoints" / "initial.pt")
    if _state_sha256(loaded_initial) != _state_sha256(initial_state):
        raise ValueError("reference initial state differs from the current seeded state")

    scenario_version = int(manifest_scenario.get("schema_version", 1))
    if scenario_version in (3, 4):
        source_scenario = _reference_scenario(
            source_config, source_split, loaded_initial, seed
        )
        comparable_split = json.loads(json.dumps(split_manifest))
        stored_marker = manifest_scenario.get("split_manifest", {}).get(
            "target_client_reconstruction_marker", {}
        )
        current_marker = comparable_split.get(
            "target_client_reconstruction_marker", {}
        )
        if (
            "input_mask" not in stored_marker
            and current_marker.get("input_mask") is None
        ):
            current_marker.pop("input_mask", None)
        current_scenario = _reference_scenario(
            config, comparable_split, initial_state, seed
        )
        if (
            scenario_version == 3
            and float(config["training"].get("semantic_weight", 0.0)) == 0.0
        ):
            stored_observer = manifest_scenario.get("semantic_classifier", {})
            current_observer = current_scenario.get("semantic_classifier", {})
            observer_hash_fields = (
                "source_state_sha256",
                "source_checkpoint_sha256",
            )
            if all(
                current_observer.get(name) == stored_observer.get(name)
                for name in observer_hash_fields
            ):
                if "source_run" in stored_observer:
                    current_observer["source_run"] = stored_observer["source_run"]
                else:
                    current_observer.pop("source_run", None)
            stored_split_observer = manifest_scenario.get(
                "split_manifest", {}
            ).get("semantic_classifier_training", {})
            current_split_observer = current_scenario.get(
                "split_manifest", {}
            ).get("semantic_classifier_training", {})
            if all(
                current_split_observer.get(name)
                == stored_split_observer.get(name)
                for name in (
                    "state_sha256",
                    "source_checkpoint_sha256",
                    "remaining_observer_protocol_sha256",
                )
            ):
                for name in (
                    "source_run",
                    "source_run_workspace_relative",
                    "source_checkpoint",
                    "source_split_training_pool_index_sha256",
                    "current_split_training_pool_index_sha256",
                    "source_pool_matches_current_training_pool",
                ):
                    if name in stored_split_observer:
                        current_split_observer[name] = stored_split_observer[name]
                    else:
                        current_split_observer.pop(name, None)
    elif scenario_version in (1, 2):
        raise ValueError(
            f"schema-{scenario_version} joint-head references are incompatible with the "
            "separate frozen semantic classifier"
        )
    else:
        raise ValueError(
            f"unsupported reference scenario schema: {scenario_version}"
        )
    difference = _first_difference(
        manifest_scenario, source_scenario, "stored_reference_scenario"
    )
    if difference is not None:
        raise ValueError(f"invalid FSCU reference metadata: {difference}")

    difference = _first_difference(
        current_scenario, manifest_scenario, "training_scenario"
    )
    if difference is not None:
        raise ValueError(f"incompatible FSCU reference: {difference}")

    split_for_comparison = (
        current_scenario["split_manifest"]
        if scenario_version == 3
        else split_manifest
    )
    difference = _first_difference(
        split_for_comparison, source_split, "split_manifest"
    )
    if difference is not None:
        raise ValueError(f"incompatible FSCU reference: {difference}")

    target_client_id = int(config["unlearning"]["target_client_id"])
    checkpoint_names = (
        "initial",
        "original",
        f"retrain_without_client_{target_client_id}",
    )
    expected_state = make_fscu_codec(config).state_dict()
    states = {}
    checksums = {}
    manifest_checkpoints = reference_manifest.get("checkpoints", {})
    for name in checkpoint_names:
        record = manifest_checkpoints.get(name, {})
        relative_path = record.get("relative_path", f"checkpoints/{name}.pt")
        checkpoint_path = (run_dir / relative_path).resolve()
        try:
            checkpoint_path.relative_to(run_dir)
        except ValueError as error:
            raise ValueError(
                f"reference checkpoint escapes its run directory: {relative_path}"
            ) from error
        checksum = _file_sha256(checkpoint_path)
        if record.get("sha256") is not None and checksum != record["sha256"]:
            raise ValueError(f"reference checkpoint SHA-256 mismatch: {name}")
        if record.get("bytes") is not None and checkpoint_path.stat().st_size != int(
            record["bytes"]
        ):
            raise ValueError(f"reference checkpoint byte size mismatch: {name}")
        state = _load_checkpoint(checkpoint_path)
        _validate_checkpoint_state(state, expected_state, name)
        states[name] = state
        checksums[name] = checksum

    manifest_seconds = reference_manifest.get("training_wall_seconds", {})
    training_wall_seconds = {
        "original_training": manifest_seconds.get("original"),
        "retrain_without_target": manifest_seconds.get(
            "retrain_without_target"
        ),
    }
    efficiency_path = run_dir / "metrics" / "efficiency.json"
    if efficiency_path.is_file():
        phase_seconds = _read_json(efficiency_path).get("phase_wall_seconds", {})
        for name in ("original_training", "retrain_without_target"):
            if training_wall_seconds.get(name) is None:
                training_wall_seconds[name] = phase_seconds.get(name)
    for name, value in training_wall_seconds.items():
        if value is not None:
            training_wall_seconds[name] = float(value)

    retrain_name = checkpoint_names[2]
    return {
        "reference_run": str(run_dir),
        "initial_state": states["initial"],
        "original_state": states["original"],
        "retrain_state": states[retrain_name],
        "checkpoint_sha256": checksums,
        "checkpoint_paths": {
            name: str(
                (run_dir / manifest_checkpoints[name]["relative_path"]).resolve()
            )
            for name in checkpoint_names
        },
        "scenario_fingerprint_sha256": manifest_fingerprint,
        "training_wall_seconds": training_wall_seconds,
        "reference_manifest_verified": True,
    }
