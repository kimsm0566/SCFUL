"""The two optimization stages of public-data Federated SCU."""

import math
from copy import deepcopy

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.data.data import select_representation_buffer_indices
from src.utils.federated import copy_model_state
from src.utils.loss import public_contrastive_loss


def _make_dataset(client_data):
    images, labels = client_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    return TensorDataset(images, labels)


def _make_loader(dataset, batch_size, seed, shuffle=True):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _make_generator(device, seed):
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _optimizer_state_to_cpu(optimizer):
    state = deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for name, value in values.items():
            if torch.is_tensor(value):
                values[name] = value.detach().cpu()
    return state


def _measure_public_posterior(model, dataset, batch_size, device):
    was_training = model.training
    model.eval()
    totals = {
        "posterior_logvar_mean": 0.0,
        "posterior_std_mean": 0.0,
        "posterior_entropy_mean": 0.0,
        "public_prior_kl": 0.0,
        "canonical_norm_mean": 0.0,
    }
    examples = 0
    loader = _make_loader(dataset, batch_size, seed=0, shuffle=False)
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            posterior = model(images, mode="posterior")
            canonical = model.canonical_latent_from_posterior(posterior)
            batch_examples = len(images)
            logvar_mean = posterior.logvar.mean()
            values = {
                "posterior_logvar_mean": logvar_mean,
                "posterior_std_mean": posterior.logvar.mul(0.5).exp().mean(),
                "posterior_entropy_mean": 0.5
                * (1.0 + math.log(2.0 * math.pi) + logvar_mean),
                "public_prior_kl": model.prior_kl(posterior),
                "canonical_norm_mean": canonical.norm(dim=1).mean(),
            }
            for name, value in values.items():
                totals[name] += float(value.cpu()) * batch_examples
            examples += batch_examples
    model.train(was_training)
    return {name: total / examples for name, total in totals.items()}


def _measure_target_information(model, dataset, batch_size, device):
    """Measure the VIB prior-KL rate on one client's inputs."""
    was_training = model.training
    model.eval()
    per_sample_values = []
    loader = _make_loader(dataset, batch_size, seed=0, shuffle=False)
    with torch.no_grad():
        for images, _ in loader:
            posterior = model(images.to(device), mode="posterior")
            per_sample_kl = 0.5 * (
                posterior.mu.square()
                + posterior.logvar.exp()
                - 1.0
                - posterior.logvar
            ).sum(dim=1)
            per_sample_values.append(per_sample_kl.detach().cpu())
    model.train(was_training)
    if not per_sample_values:
        raise ValueError("target client data must contain at least one sample")
    values = torch.cat(per_sample_values)
    return {
        "mean_prior_kl": float(values.mean()),
        "p95_prior_kl": float(torch.quantile(values, 0.95)),
        "mean_prior_kl_per_latent": float(values.mean() / model.config.latent_dim),
        "examples": int(values.numel()),
    }


def extract_erased_representation_buffer(
    model,
    target_data,
    batch_size,
    device,
    max_examples=None,
    seed=0,
    sampling_mode="stratified",
):
    """Extract the canonical normalized posterior mean without raw inputs."""
    if batch_size <= 0:
        raise ValueError("buffer batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    device = torch.device(device)
    model = deepcopy(model).to(device).eval()
    dataset = _make_dataset(target_data)
    selected = select_representation_buffer_indices(
        dataset.tensors[1], max_examples, seed, mode=sampling_mode
    )
    selected_dataset = TensorDataset(
        dataset.tensors[0][selected], dataset.tensors[1][selected]
    )
    loader = _make_loader(selected_dataset, batch_size, seed=seed, shuffle=False)
    representations = []
    with torch.no_grad():
        for images, _ in loader:
            canonical = model(images.to(device), mode="canonical_latent")
            representations.append(canonical.detach().cpu())
    if not representations:
        raise ValueError("target client data must contain at least one sample")
    return torch.cat(representations, dim=0).contiguous()


def forget_target_client(
    model,
    target_data,
    original_model,
    config,
    seed,
    optimizer_state=None,
    return_optimizer_state=False,
    semantic_classifier=None,
):
    """Jointly update encoder and decoder on the target client's local data."""
    device = torch.device(config["device"])
    learner = deepcopy(model).to(device).train()
    reference = deepcopy(original_model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    if float(config.get("semantic_weight", 0.0)) > 0 and semantic_classifier is None:
        raise ValueError(
            "semantic_classifier is required when semantic_weight is positive"
        )

    optimizer = torch.optim.Adam(
        learner.parameters(), lr=float(config["learning_rate"])
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(deepcopy(optimizer_state))
    target_dataset = _make_dataset(target_data)
    loader = _make_loader(target_dataset, int(config["batch_size"]), seed)
    forgetting_objective = str(
        config.get("forgetting_objective", "posterior_kl_to_original")
    )
    if forgetting_objective not in (
        "posterior_kl_to_original",
        "prior_kl_collapse",
    ):
        raise ValueError(f"unsupported forgetting objective: {forgetting_objective}")
    information_ratio_threshold = float(
        config.get("target_information_ratio_threshold", 0.0)
    )
    if not 0.0 <= information_ratio_threshold < 1.0:
        raise ValueError("target information ratio threshold must be in [0, 1)")
    original_information = _measure_target_information(
        reference,
        target_dataset,
        int(config["batch_size"]),
        device,
    )
    posterior_generator = _make_generator(device, seed + 1)
    channel_generator = _make_generator(device, seed + 2)
    preservation_generator = _make_generator(device, seed + 3)
    totals = {
        "target_mse": 0.0,
        "target_reconstruction_loss": 0.0,
        "target_prior_kl": 0.0,
        "posterior_kl_to_original": 0.0,
        "target_semantic_loss": 0.0,
        "target_semantic_accuracy": 0.0,
        "decoder_preservation_loss": 0.0,
        "total_loss": 0.0,
    }
    history = []
    steps = 0
    for epoch in range(int(config["epochs"])):
        epoch_start_totals = dict(totals)
        epoch_steps = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                reference_posterior = reference(images, mode="posterior")
            output = learner(
                images,
                mode="training",
                snr_db=float(config["snr_db"]),
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            posterior_kl = learner.posterior_kl_to(
                output.posterior, reference_posterior
            )
            target_prior_kl = learner.prior_kl(output.posterior)
            target_squared_error = functional.mse_loss(
                output.reconstruction, images, reduction="sum"
            )
            target_mse = target_squared_error / images.numel()
            if learner.config.kl_reduction == "batchmean_sum":
                target_reconstruction_loss = target_squared_error / len(images)
            else:
                target_reconstruction_loss = target_mse
            forgetting_kl = (
                target_prior_kl
                if forgetting_objective == "prior_kl_collapse"
                else posterior_kl
            )
            loss = (
                float(config["lambda_kl"]) * forgetting_kl
                - float(config["lambda_rec"]) * target_reconstruction_loss
            )
            decoder_preservation_weight = float(
                config.get("lambda_decoder_preservation", 0.0)
            )
            if decoder_preservation_weight > 0:
                standard_noise = torch.randn(
                    output.posterior.mu.shape,
                    dtype=output.posterior.mu.dtype,
                    device=device,
                    generator=preservation_generator,
                )
                received = learner.transmit_with_standard_awgn(
                    output.posterior.mu.detach(),
                    standard_noise,
                    float(config["snr_db"]),
                ).detach()
                learner_decoder_output = learner.decode_latent(received)
                with torch.no_grad():
                    reference_decoder_output = reference.decode_latent(received)
                decoder_preservation = functional.mse_loss(
                    learner_decoder_output, reference_decoder_output
                )
                loss += decoder_preservation_weight * decoder_preservation
            else:
                decoder_preservation = torch.zeros((), device=device)
            semantic_weight = float(config.get("semantic_weight", 0.0))
            semantic_ascent = bool(config.get("semantic_ascent", True))
            if semantic_weight > 0:
                logits = semantic_classifier(output.reconstruction)
                semantic_loss = functional.cross_entropy(logits, labels)
                semantic_accuracy = (logits.argmax(dim=1) == labels).float().mean()
                if semantic_ascent:
                    loss -= (
                        float(config["lambda_rec"])
                        * semantic_weight
                        * semantic_loss
                    )
            else:
                semantic_loss = torch.zeros((), device=device)
                semantic_accuracy = torch.zeros((), device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
            optimizer.step()
            totals["target_mse"] += float(target_mse.detach().cpu())
            totals["target_reconstruction_loss"] += float(
                target_reconstruction_loss.detach().cpu()
            )
            totals["target_prior_kl"] += float(
                target_prior_kl.detach().cpu()
            )
            totals["posterior_kl_to_original"] += float(
                posterior_kl.detach().cpu()
            )
            totals["target_semantic_loss"] += float(semantic_loss.detach().cpu())
            totals["target_semantic_accuracy"] += float(
                semantic_accuracy.detach().cpu()
            )
            totals["decoder_preservation_loss"] += float(
                decoder_preservation.detach().cpu()
            )
            totals["total_loss"] += float(loss.detach().cpu())
            steps += 1
            epoch_steps += 1
        current_information = _measure_target_information(
            learner,
            target_dataset,
            int(config["batch_size"]),
            device,
        )
        mean_information_ratio = (
            current_information["mean_prior_kl"]
            / max(original_information["mean_prior_kl"], 1e-12)
        )
        p95_information_ratio = (
            current_information["p95_prior_kl"]
            / max(original_information["p95_prior_kl"], 1e-12)
        )
        history.append(
            {
                "epoch": int(epoch + 1),
                "steps": int(epoch_steps),
                **{
                    name: (totals[name] - epoch_start_totals[name]) / epoch_steps
                    for name in totals
                },
                "measured_target_prior_kl": current_information["mean_prior_kl"],
                "measured_target_prior_kl_p95": current_information["p95_prior_kl"],
                "target_information_ratio": mean_information_ratio,
                "target_information_ratio_p95": p95_information_ratio,
            }
        )
        if (
            forgetting_objective == "prior_kl_collapse"
            and information_ratio_threshold > 0.0
            and mean_information_ratio <= information_ratio_threshold
        ):
            break
    if steps == 0:
        raise ValueError("target client data must contain at least one batch")
    metrics = {
        name: value / steps for name, value in totals.items()
    } | {
        "steps": steps,
        "epochs_completed": len(history),
        "forgetting_objective": forgetting_objective,
        "target_information_ratio_threshold": information_ratio_threshold,
        "original_target_information": original_information,
        "final_target_information": {
            "mean_prior_kl": history[-1]["measured_target_prior_kl"],
            "p95_prior_kl": history[-1]["measured_target_prior_kl_p95"],
            "mean_ratio": history[-1]["target_information_ratio"],
            "p95_ratio": history[-1]["target_information_ratio_p95"],
        },
        "optimizer_state_reused": optimizer_state is not None,
        "semantic_ascent": bool(config.get("semantic_ascent", True)),
        "lambda_decoder_preservation": float(
            config.get("lambda_decoder_preservation", 0.0)
        ),
        "reconstruction_reduction": learner.config.kl_reduction,
        "history": history,
    }
    result = (copy_model_state(learner), metrics)
    if return_optimizer_state:
        return result + (_optimizer_state_to_cpu(optimizer),)
    return result


def train_server_public_contrastive_compensation(
    joint_unlearned_model,
    public_data,
    erased_representation_buffer,
    config,
    seed,
    optimizer_state=None,
    return_optimizer_state=False,
    semantic_classifier=None,
):
    """Repair the full codec using public contrastive and reconstruction losses."""
    if int(config["epochs"]) <= 0:
        raise ValueError("public compensation epochs must be positive")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("public compensation learning_rate must be positive")
    if float(config["temperature"]) <= 0:
        raise ValueError("contrastive temperature must be positive")
    contrastive_weight = float(config["lambda_contrastive"])
    reconstruction_weight = float(config["lambda_public_reconstruction"])
    if contrastive_weight < 0 or reconstruction_weight < 0:
        raise ValueError("public compensation weights must be non-negative")
    if contrastive_weight == 0 and reconstruction_weight == 0:
        raise ValueError("at least one public compensation weight must be positive")

    device = torch.device(config["device"])
    learner = deepcopy(joint_unlearned_model).to(device).train()
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    if float(config.get("semantic_weight", 0.0)) > 0 and semantic_classifier is None:
        raise ValueError(
            "semantic_classifier is required when semantic_weight is positive"
        )
    erased_buffer = torch.as_tensor(
        erased_representation_buffer, dtype=torch.float32, device=device
    ).detach()
    if erased_buffer.ndim != 2 or len(erased_buffer) == 0:
        raise ValueError("erased representation buffer must be a non-empty matrix")
    if erased_buffer.shape[1] != learner.config.latent_dim:
        raise ValueError("erased representation dimension must match latent_dim")

    dataset = _make_dataset(public_data)
    pre_diagnostics = _measure_public_posterior(
        learner, dataset, int(config["batch_size"]), device
    )
    optimizer = torch.optim.Adam(
        learner.parameters(), lr=float(config["learning_rate"])
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(deepcopy(optimizer_state))
    contrastive_generator_one = _make_generator(device, seed + 1)
    contrastive_generator_two = _make_generator(device, seed + 2)
    posterior_generator = _make_generator(device, seed + 3)
    channel_generator = _make_generator(device, seed + 4)
    totals = {
        "loss": 0.0,
        "contrastive_loss": 0.0,
        "public_reconstruction_loss": 0.0,
        "public_semantic_loss": 0.0,
        "public_semantic_accuracy": 0.0,
        "public_prior_kl": 0.0,
        "posterior_logvar_mean": 0.0,
        "posterior_std_mean": 0.0,
        "posterior_entropy_mean": 0.0,
        "positive_cosine": 0.0,
        "negative_cosine_mean": 0.0,
        "negative_cosine_max": 0.0,
        "canonical_norm_mean": 0.0,
    }
    steps = 0
    for epoch in range(int(config["epochs"])):
        loader = _make_loader(
            dataset, int(config["batch_size"]), seed + 100 + epoch
        )
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            contrastive, contrastive_diagnostics = public_contrastive_loss(
                learner,
                images,
                erased_buffer,
                float(config["temperature"]),
                contrastive_generator_one,
                contrastive_generator_two,
                positive_mode=str(
                    config.get(
                        "public_positive_mode", "deterministic_augmentation"
                    )
                ),
                crop_padding=int(config.get("augmentation_crop_padding", 4)),
                horizontal_flip_probability=float(
                    config.get("augmentation_horizontal_flip_probability", 0.5)
                ),
                return_diagnostics=True,
            )
            output = learner(
                images,
                mode="training",
                snr_db=float(config["snr_db"]),
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            reconstruction = functional.mse_loss(output.reconstruction, images)
            prior_kl_weight = float(config.get("lambda_public_prior_kl", 0.0))
            loss = (
                contrastive_weight * contrastive
                + reconstruction_weight * reconstruction
                + prior_kl_weight * contrastive_diagnostics["public_prior_kl"]
            )
            semantic_weight = float(config.get("semantic_weight", 0.0))
            if semantic_weight > 0:
                logits = semantic_classifier(output.reconstruction)
                semantic_loss = functional.cross_entropy(logits, labels)
                semantic_accuracy = (logits.argmax(dim=1) == labels).float().mean()
                loss += semantic_weight * semantic_loss
            else:
                semantic_loss = torch.zeros((), device=device)
                semantic_accuracy = torch.zeros((), device=device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["contrastive_loss"] += float(contrastive.detach().cpu())
            totals["public_reconstruction_loss"] += float(
                reconstruction.detach().cpu()
            )
            totals["public_semantic_loss"] += float(semantic_loss.detach().cpu())
            totals["public_semantic_accuracy"] += float(
                semantic_accuracy.detach().cpu()
            )
            for name in (
                "public_prior_kl",
                "posterior_logvar_mean",
                "posterior_std_mean",
                "posterior_entropy_mean",
                "positive_cosine",
                "negative_cosine_mean",
                "negative_cosine_max",
                "canonical_norm_mean",
            ):
                totals[name] += float(contrastive_diagnostics[name].detach().cpu())
            steps += 1
    if steps == 0:
        raise ValueError("public data must contain at least one batch")
    metrics = {name: value / steps for name, value in totals.items()}
    post_diagnostics = _measure_public_posterior(
        learner, dataset, int(config["batch_size"]), device
    )
    metrics.update(
        {
            "public_examples": len(dataset),
            "erased_buffer_examples": len(erased_buffer),
            "erased_buffer_latent_dim": int(erased_buffer.shape[1]),
            "erased_buffer_bytes": int(
                erased_buffer.numel() * erased_buffer.element_size()
            ),
            "steps": steps,
            "optimizer_state_reused": optimizer_state is not None,
            "public_positive_mode": str(
                config.get("public_positive_mode", "deterministic_augmentation")
            ),
            "lambda_public_prior_kl": float(
                config.get("lambda_public_prior_kl", 0.0)
            ),
            **{
                f"pre_{name}": value
                for name, value in pre_diagnostics.items()
            },
            **{
                f"post_{name}": value
                for name, value in post_diagnostics.items()
            },
        }
    )
    result = (copy_model_state(learner), metrics)
    if return_optimizer_state:
        return result + (_optimizer_state_to_cpu(optimizer),)
    return result
