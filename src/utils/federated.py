"""재구성 semantic codec을 위한 client-local 학습과 FedAvg helper."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.data.data import augment_cifar10_contrastive_batch
from src.utils.loss import normalized_feature_distillation_loss


class LocalReconstructionConfig:
    def __init__(
        self,
        batch_size,
        local_epochs,
        learning_rate,
        beta,
        snr_db,
        reconstruction_weight=1.0,
        semantic_weight=0.0,
        augmentation="none",
        augmentation_crop_padding=4,
        augmentation_horizontal_flip_probability=0.5,
        kl_reduction="batchmean_sum",
        gradient_clip_norm=None,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.snr_db = snr_db
        self.reconstruction_weight = reconstruction_weight
        self.semantic_weight = semantic_weight
        if kl_reduction not in ("elementwise_mean", "batchmean_sum"):
            raise ValueError(
                "kl_reduction must be elementwise_mean or batchmean_sum"
            )
        if gradient_clip_norm is not None:
            gradient_clip_norm = float(gradient_clip_norm)
            if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive or null")
        self.kl_reduction = kl_reduction
        self.gradient_clip_norm = gradient_clip_norm
        if augmentation not in ("none", "cifar10_crop_flip_v1"):
            raise ValueError(f"unsupported local training augmentation: {augmentation}")
        if int(augmentation_crop_padding) < 0:
            raise ValueError("augmentation_crop_padding must be non-negative")
        flip_probability = float(augmentation_horizontal_flip_probability)
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError(
                "augmentation_horizontal_flip_probability must be in [0, 1]"
            )
        self.augmentation = augmentation
        self.augmentation_crop_padding = int(augmentation_crop_padding)
        self.augmentation_horizontal_flip_probability = flip_probability
        self.device = device


class LocalClassificationConfig:
    """Independent raw-image CIFAR-10 client classification settings."""

    def __init__(
        self,
        batch_size,
        local_epochs,
        learning_rate,
        weight_decay=0.0,
        augmentation="cifar10_crop_flip_v1",
        augmentation_crop_padding=4,
        augmentation_horizontal_flip_probability=0.5,
        preserve_fixed_backdoor_trigger=False,
        device="cpu",
    ):
        if int(batch_size) <= 0 or int(local_epochs) <= 0:
            raise ValueError("classification batch_size and local_epochs must be positive")
        if float(learning_rate) <= 0:
            raise ValueError("classification learning_rate must be positive")
        if float(weight_decay) < 0:
            raise ValueError("classification weight_decay must be non-negative")
        if augmentation not in ("none", "cifar10_crop_flip_v1"):
            raise ValueError(
                f"unsupported local classification augmentation: {augmentation}"
            )
        if int(augmentation_crop_padding) < 0:
            raise ValueError("augmentation_crop_padding must be non-negative")
        flip_probability = float(augmentation_horizontal_flip_probability)
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError(
                "augmentation_horizontal_flip_probability must be in [0, 1]"
            )
        self.batch_size = int(batch_size)
        self.local_epochs = int(local_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.augmentation = augmentation
        self.augmentation_crop_padding = int(augmentation_crop_padding)
        self.augmentation_horizontal_flip_probability = flip_probability
        if not isinstance(preserve_fixed_backdoor_trigger, bool):
            raise ValueError("preserve_fixed_backdoor_trigger must be boolean")
        self.preserve_fixed_backdoor_trigger = preserve_fixed_backdoor_trigger
        self.device = device


class RetainedFeatureKDConfig:
    """Local retained-data repair settings for FSCUv4."""

    def __init__(
        self,
        batch_size,
        local_epochs,
        learning_rate,
        beta,
        snr_db,
        reconstruction_weight,
        semantic_weight,
        output_kd_weight,
        feature_kd_weight,
        reconstruction_reduction="batchmean_sum",
        device="cpu",
    ):
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.snr_db = snr_db
        self.reconstruction_weight = reconstruction_weight
        self.semantic_weight = semantic_weight
        self.output_kd_weight = output_kd_weight
        self.feature_kd_weight = feature_kd_weight
        if reconstruction_reduction not in ("elementwise_mean", "batchmean_sum"):
            raise ValueError(
                "reconstruction_reduction must be elementwise_mean or batchmean_sum"
            )
        self.reconstruction_reduction = reconstruction_reduction
        self.device = device


def _make_loader(client_data, batch_size, seed):
    images, labels = client_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _make_generator(device, seed):
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def copy_model_state(model):
    """통신 가능한 CPU model state를 복사한다."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _copy_optimizer_state_to_cpu(optimizer):
    """다음 FL round에서 재사용할 Adam state를 CPU에 안전하게 보관한다."""

    def copy_value(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: copy_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_value(item) for item in value)
        return deepcopy(value)

    return copy_value(optimizer.state_dict())


def average_model_states(states, sample_counts):
    """Client sample 수를 가중치로 full model state를 평균한다."""
    if not states:
        raise ValueError("at least one client state is required")
    if len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must have the same length")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("every sample count must be positive")

    total_samples = float(sum(sample_counts))
    averaged = {}
    for name, first_value in states[0].items():
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value, dtype=torch.float32)
            for state, count in zip(states, sample_counts):
                value += state[name].to(dtype=torch.float32) * (count / total_samples)
            averaged[name] = value.to(dtype=first_value.dtype)
        else:
            averaged[name] = first_value.clone()
    return averaged


def conflict_aware_average_updates(
    base_state,
    local_states,
    sample_counts,
    original_state,
    forgotten_state,
    epsilon=1e-12,
):
    """Project conflicting client repair updates before weighted aggregation.

    The fixed forgetting direction is ``forgotten_state - original_state``.
    Each repair direction is ``local_state - base_state``.  A negative dot
    product means that the client update moves back toward the Original model;
    only that conflicting parallel component is removed.  Orthogonal and
    forgetting-aligned components are retained.
    """
    if not local_states:
        raise ValueError("at least one local repair state is required")
    if len(local_states) != len(sample_counts):
        raise ValueError("local_states and sample_counts must have the same length")
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("every sample count must be positive")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")

    expected_names = list(base_state)
    compared_states = [original_state, forgotten_state, *local_states]
    for state in compared_states:
        if list(state) != expected_names:
            raise ValueError("all model states must have identical ordered keys")
        for name in expected_names:
            if state[name].shape != base_state[name].shape:
                raise ValueError(f"model state shape mismatch for {name}")

    floating_names = [
        name for name in expected_names if torch.is_floating_point(base_state[name])
    ]
    if not floating_names:
        raise ValueError("model state has no floating-point parameters")

    forgetting_norm_sq = 0.0
    for name in floating_names:
        forgetting_delta = (
            forgotten_state[name].to(dtype=torch.float64)
            - original_state[name].to(dtype=torch.float64)
        )
        forgetting_norm_sq += float(
            torch.sum(forgetting_delta * forgetting_delta).item()
        )
    if forgetting_norm_sq <= epsilon:
        raise ValueError("joint forgetting direction has near-zero norm")
    forgetting_norm = math.sqrt(forgetting_norm_sq)

    filtered_deltas = []
    client_diagnostics = []
    for local_state in local_states:
        dot = 0.0
        repair_norm_sq = 0.0
        for name in floating_names:
            repair_delta = (
                local_state[name].to(dtype=torch.float64)
                - base_state[name].to(dtype=torch.float64)
            )
            forgetting_delta = (
                forgotten_state[name].to(dtype=torch.float64)
                - original_state[name].to(dtype=torch.float64)
            )
            dot += float(torch.sum(repair_delta * forgetting_delta).item())
            repair_norm_sq += float(torch.sum(repair_delta * repair_delta).item())

        repair_norm = math.sqrt(repair_norm_sq)
        cosine_before = dot / max(repair_norm * forgetting_norm, epsilon)
        conflicting = dot < 0.0
        projection_coefficient = dot / forgetting_norm_sq if conflicting else 0.0
        filtered = {}
        filtered_norm_sq = 0.0
        filtered_dot = 0.0
        for name in floating_names:
            repair_delta = (
                local_state[name].to(dtype=torch.float64)
                - base_state[name].to(dtype=torch.float64)
            )
            forgetting_delta = (
                forgotten_state[name].to(dtype=torch.float64)
                - original_state[name].to(dtype=torch.float64)
            )
            filtered_delta = repair_delta - projection_coefficient * forgetting_delta
            filtered[name] = filtered_delta
            filtered_norm_sq += float(
                torch.sum(filtered_delta * filtered_delta).item()
            )
            filtered_dot += float(
                torch.sum(filtered_delta * forgetting_delta).item()
            )
        filtered_norm = math.sqrt(filtered_norm_sq)
        cosine_after = filtered_dot / max(filtered_norm * forgetting_norm, epsilon)
        filtered_deltas.append(filtered)
        client_diagnostics.append(
            {
                "conflicting": bool(conflicting),
                "repair_update_norm": repair_norm,
                "filtered_update_norm": filtered_norm,
                "cosine_before": cosine_before,
                "cosine_after": cosine_after,
                "projection_coefficient": projection_coefficient,
                "removed_component_norm": (
                    abs(projection_coefficient) * forgetting_norm
                ),
                "removed_norm_fraction": (
                    min(
                        1.0,
                        abs(projection_coefficient)
                        * forgetting_norm
                        / repair_norm,
                    )
                    if repair_norm > epsilon
                    else 0.0
                ),
            }
        )

    total_samples = float(sum(sample_counts))
    aggregated = {}
    aggregate_dot = 0.0
    aggregate_norm_sq = 0.0
    raw_aggregate_dot = 0.0
    raw_aggregate_norm_sq = 0.0
    for name in expected_names:
        if name not in floating_names:
            aggregated[name] = base_state[name].clone()
            continue
        aggregate_delta = torch.zeros_like(
            base_state[name], dtype=torch.float64
        )
        raw_aggregate_delta = torch.zeros_like(
            base_state[name], dtype=torch.float64
        )
        for filtered, local_state, count in zip(
            filtered_deltas, local_states, sample_counts
        ):
            aggregate_delta += filtered[name] * (float(count) / total_samples)
            raw_aggregate_delta += (
                local_state[name].to(dtype=torch.float64)
                - base_state[name].to(dtype=torch.float64)
            ) * (float(count) / total_samples)
        aggregated[name] = (
            base_state[name].to(dtype=torch.float64) + aggregate_delta
        ).to(dtype=base_state[name].dtype)
        forgetting_delta = (
            forgotten_state[name].to(dtype=torch.float64)
            - original_state[name].to(dtype=torch.float64)
        )
        aggregate_dot += float(
            torch.sum(aggregate_delta * forgetting_delta).item()
        )
        aggregate_norm_sq += float(
            torch.sum(aggregate_delta * aggregate_delta).item()
        )
        raw_aggregate_dot += float(
            torch.sum(raw_aggregate_delta * forgetting_delta).item()
        )
        raw_aggregate_norm_sq += float(
            torch.sum(raw_aggregate_delta * raw_aggregate_delta).item()
        )
    aggregate_norm = math.sqrt(aggregate_norm_sq)
    raw_aggregate_norm = math.sqrt(raw_aggregate_norm_sq)
    aggregate_cosine = aggregate_dot / max(
        aggregate_norm * forgetting_norm, epsilon
    )
    diagnostics = {
        "aggregation": "conflict_aware_projection",
        "conflict_reference": "target_joint_unlearning_update",
        "projection_scope": "global_client_update",
        "forgetting_update_norm": forgetting_norm,
        "raw_aggregate_repair_update_norm": raw_aggregate_norm,
        "aggregate_repair_update_norm": aggregate_norm,
        "aggregate_cosine_before": raw_aggregate_dot
        / max(raw_aggregate_norm * forgetting_norm, epsilon),
        "aggregate_cosine_after": aggregate_cosine,
        "conflict_client_count": int(
            sum(item["conflicting"] for item in client_diagnostics)
        ),
        "conflict_client_fraction": float(
            sum(item["conflicting"] for item in client_diagnostics)
            / len(client_diagnostics)
        ),
        "client_diagnostics": client_diagnostics,
    }
    return aggregated, diagnostics


def train_local_reconstruction_with_optimizer_state(
    model,
    client_data,
    config,
    seed,
    optimizer_state=None,
    semantic_classifier=None,
):
    """Local Adam moment를 받아 한 client를 학습하고 갱신된 state를 반환한다."""
    device = torch.device(config.device)
    local_model = deepcopy(model).to(device)
    local_model.train()
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    if config.semantic_weight > 0 and semantic_classifier is None:
        raise ValueError(
            "semantic_classifier is required when semantic_weight is positive"
        )
    optimizer = torch.optim.Adam(local_model.parameters(), lr=config.learning_rate)
    optimizer_state_reused = optimizer_state is not None
    if optimizer_state_reused:
        optimizer.load_state_dict(deepcopy(optimizer_state))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config.learning_rate)
    loader = _make_loader(client_data, config.batch_size, seed)
    posterior_generator = _make_generator(device, seed + 1)
    channel_generator = _make_generator(device, seed + 2)
    augmentation_generator = _make_generator(device, seed + 3)

    reconstruction_total = 0.0
    squared_error_total = 0.0
    kl_total = 0.0
    semantic_total = 0.0
    objective_total = 0.0
    semantic_correct = 0
    semantic_examples = 0
    processed_examples = 0
    processed_elements = 0
    steps = 0
    for _ in range(config.local_epochs):
        for images, labels in loader:
            images = images.to(device)
            if config.semantic_weight > 0:
                labels = labels.to(device)
            if config.augmentation == "cifar10_crop_flip_v1":
                images = augment_cifar10_contrastive_batch(
                    images,
                    generator=augmentation_generator,
                    crop_padding=config.augmentation_crop_padding,
                    horizontal_flip_probability=(
                        config.augmentation_horizontal_flip_probability
                    ),
                )
            output = local_model(
                images,
                mode="training",
                snr_db=config.snr_db,
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            batch_examples = len(images)
            squared_error = functional.mse_loss(
                output.reconstruction,
                images,
                reduction="sum",
            )
            train_mse = squared_error / images.numel()
            if config.kl_reduction == "batchmean_sum":
                reconstruction_loss = squared_error / batch_examples
            else:
                reconstruction_loss = train_mse
            kl_loss = local_model.prior_kl(output.posterior)
            loss = config.reconstruction_weight * reconstruction_loss + config.beta * kl_loss
            if config.semantic_weight > 0:
                logits = semantic_classifier(output.reconstruction)
                semantic_loss = functional.cross_entropy(logits, labels)
                loss = loss + config.semantic_weight * semantic_loss
                semantic_total += (
                    float(semantic_loss.detach().cpu()) * batch_examples
                )
                semantic_correct += int((logits.argmax(dim=1) == labels).sum().item())
                semantic_examples += len(labels)

            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    max_norm=config.gradient_clip_norm,
                )
            optimizer.step()

            objective_total += float(loss.detach().cpu()) * batch_examples
            reconstruction_total += (
                float(reconstruction_loss.detach().cpu()) * batch_examples
            )
            squared_error_total += float(squared_error.detach().cpu())
            kl_total += float(kl_loss.detach().cpu()) * batch_examples
            processed_examples += batch_examples
            processed_elements += images.numel()
            steps += 1

    if steps == 0:
        raise ValueError("client data must contain at least one batch")
    metrics = {
        "objective_loss": objective_total / processed_examples,
        "reconstruction_loss": reconstruction_total / processed_examples,
        "train_mse": squared_error_total / processed_elements,
        "kl_loss": kl_total / processed_examples,
        "semantic_loss": (
            semantic_total / semantic_examples if semantic_examples else None
        ),
        "semantic_accuracy": (
            semantic_correct / semantic_examples if semantic_examples else None
        ),
        "examples": len(loader.dataset),
        "processed_examples": processed_examples,
        "steps": steps,
        "optimizer_state_reused": optimizer_state_reused,
        "learning_rate": float(config.learning_rate),
        "gradient_clip_norm": config.gradient_clip_norm,
        "kl_reduction": config.kl_reduction,
    }
    return (
        copy_model_state(local_model),
        metrics,
        _copy_optimizer_state_to_cpu(optimizer),
    )


def train_local_reconstruction(
    model, client_data, config, seed, semantic_classifier=None
):
    """한 번의 독립 local update를 수행한다.

    여러 FL round에서 같은 client의 Adam moment를 유지해야 할 때는
    ``train_local_reconstruction_with_optimizer_state``를 사용한다.
    """
    state, metrics, _ = train_local_reconstruction_with_optimizer_state(
        model,
        client_data,
        config,
        seed,
        optimizer_state=None,
        semantic_classifier=semantic_classifier,
    )
    return state, metrics


def _make_classification_loader(
    client_data,
    batch_size,
    seed,
    clean_client_data=None,
    fixed_trigger_positions=None,
):
    """Build a classifier loader with an optional clean augmentation source."""

    images, labels = client_data
    images = torch.as_tensor(images, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if images.max() > 1.0:
        images = images / 255.0
    trigger_mask = torch.zeros(len(images), dtype=torch.bool)
    source_images = images
    if fixed_trigger_positions is not None:
        if clean_client_data is None:
            raise ValueError(
                "fixed trigger positions require clean pre-poison client data"
            )
        clean_images, clean_labels = clean_client_data
        clean_images = torch.as_tensor(clean_images, dtype=torch.float32)
        clean_labels = torch.as_tensor(clean_labels, dtype=torch.long)
        if clean_images.max() > 1.0:
            clean_images = clean_images / 255.0
        if clean_images.shape != images.shape or len(clean_labels) != len(labels):
            raise ValueError(
                "clean pre-poison data must align with classifier client data"
            )
        positions = torch.as_tensor(fixed_trigger_positions, dtype=torch.long)
        if positions.ndim != 1 or len(positions) == 0:
            raise ValueError("fixed trigger positions must be a non-empty vector")
        if len(torch.unique(positions)) != len(positions):
            raise ValueError("fixed trigger positions must be unique")
        if int(positions.min()) < 0 or int(positions.max()) >= len(images):
            raise ValueError("fixed trigger positions are outside client data")
        trigger_mask[positions] = True
        source_images = clean_images
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(source_images, labels, trigger_mask),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_local_classification(
    model,
    client_data,
    config,
    seed,
    clean_client_data=None,
    fixed_trigger_positions=None,
):
    """Train only an independent classifier on one client's raw CIFAR-10 data."""

    device = torch.device(config.device)
    local_model = deepcopy(model).to(device)
    local_model.train()
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if fixed_trigger_positions is not None and not config.preserve_fixed_backdoor_trigger:
        raise ValueError(
            "fixed trigger positions require preserve_fixed_backdoor_trigger=true"
        )
    loader = _make_classification_loader(
        client_data,
        config.batch_size,
        seed,
        clean_client_data=clean_client_data,
        fixed_trigger_positions=fixed_trigger_positions,
    )
    augmentation_generator = _make_generator(device, seed + 3)

    cross_entropy_total = 0.0
    correct = 0
    examples = 0
    steps = 0
    for _ in range(config.local_epochs):
        for images, labels, fixed_trigger_mask in loader:
            images = images.to(device)
            labels = labels.to(device)
            fixed_trigger_mask = fixed_trigger_mask.to(device)
            if config.augmentation == "cifar10_crop_flip_v1":
                images = augment_cifar10_contrastive_batch(
                    images,
                    generator=augmentation_generator,
                    crop_padding=config.augmentation_crop_padding,
                    horizontal_flip_probability=(
                        config.augmentation_horizontal_flip_probability
                    ),
                )
            if bool(fixed_trigger_mask.any()):
                images[fixed_trigger_mask, :, 28:30, 28:30] = 1.0
            logits = local_model(images)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            cross_entropy_total += float(loss.detach().cpu()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += len(labels)
            steps += 1

    if steps == 0:
        raise ValueError("client data must contain at least one batch")
    return copy_model_state(local_model), {
        "cross_entropy": cross_entropy_total / examples,
        "accuracy": correct / examples,
        "examples": len(loader.dataset),
        "steps": steps,
    }


def train_local_retained_feature_kd(
    model,
    original_teacher,
    client_data,
    config,
    seed,
    semantic_classifier=None,
):
    """Repair one client model using only that client's retained raw data.

    KD variants pair the frozen teacher and student on the same retained input.
    Pure reconstruction repair accepts ``original_teacher=None`` and does not
    construct or evaluate an Original model.  No target-client data enters this
    routine.
    """
    device = torch.device(config.device)
    local_model = deepcopy(model).to(device)
    local_model.train()
    uses_teacher = bool(
        float(config.output_kd_weight) > 0.0
        or float(config.feature_kd_weight) > 0.0
    )
    if uses_teacher and original_teacher is None:
        raise ValueError("a frozen Original teacher is required for KD repair")
    teacher = None
    if uses_teacher:
        teacher = deepcopy(original_teacher).to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    if semantic_classifier is not None:
        semantic_classifier = semantic_classifier.to(device).freeze()
    if float(config.semantic_weight) > 0 and semantic_classifier is None:
        raise ValueError(
            "semantic_classifier is required when semantic_weight is positive"
        )

    loader = _make_loader(client_data, config.batch_size, seed)
    optimizer = torch.optim.Adam(
        local_model.parameters(), lr=config.learning_rate
    )
    posterior_generator = _make_generator(device, seed + 1)
    channel_generator = _make_generator(device, seed + 2)

    totals = {
        "reconstruction_loss": 0.0,
        "prior_kl_loss": 0.0,
        "semantic_loss": 0.0,
        "output_kd_loss": 0.0,
        "encoder_feature_kd_loss": 0.0,
        "decoder_feature_kd_loss": 0.0,
        "total_loss": 0.0,
    }
    semantic_correct = 0
    semantic_examples = 0
    processed_examples = 0
    steps = 0
    latent_dim = int(local_model.config.latent_dim)

    for _ in range(int(config.local_epochs)):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if uses_teacher:
                standard_noise = torch.randn(
                    (len(images), latent_dim),
                    dtype=images.dtype,
                    device=device,
                    generator=channel_generator,
                )
                with torch.no_grad():
                    teacher_output = teacher(
                        images,
                        mode="distillation",
                        standard_noise=standard_noise,
                        snr_db=float(config.snr_db),
                    )
                student_output = local_model(
                    images,
                    mode="distillation",
                    standard_noise=standard_noise,
                    snr_db=float(config.snr_db),
                )
            else:
                teacher_output = None
                student_output = local_model(
                    images,
                    mode="training",
                    snr_db=float(config.snr_db),
                    posterior_generator=posterior_generator,
                    channel_generator=channel_generator,
                )

            reconstruction_squared_error = functional.mse_loss(
                student_output.reconstruction, images, reduction="sum"
            )
            output_kd_squared_error = (
                functional.mse_loss(
                    student_output.reconstruction,
                    teacher_output.reconstruction.detach(),
                    reduction="sum",
                )
                if uses_teacher
                else torch.zeros((), device=device)
            )
            if config.reconstruction_reduction == "batchmean_sum":
                reconstruction_loss = reconstruction_squared_error / len(images)
                output_kd_loss = output_kd_squared_error / len(images)
            else:
                reconstruction_loss = reconstruction_squared_error / images.numel()
                output_kd_loss = output_kd_squared_error / images.numel()
            prior_kl_loss = local_model.prior_kl(student_output.posterior)
            if uses_teacher:
                encoder_feature_loss = normalized_feature_distillation_loss(
                    student_output.encoder_shallow,
                    teacher_output.encoder_shallow,
                )
                decoder_feature_loss = normalized_feature_distillation_loss(
                    student_output.decoder_shallow,
                    teacher_output.decoder_shallow,
                )
            else:
                encoder_feature_loss = torch.zeros((), device=device)
                decoder_feature_loss = torch.zeros((), device=device)
            feature_kd_loss = 0.5 * (
                encoder_feature_loss + decoder_feature_loss
            )
            loss = (
                float(config.reconstruction_weight) * reconstruction_loss
                + float(config.beta) * prior_kl_loss
                + float(config.output_kd_weight) * output_kd_loss
                + float(config.feature_kd_weight) * feature_kd_loss
            )

            semantic_loss = torch.zeros((), device=device)
            if float(config.semantic_weight) > 0:
                logits = semantic_classifier(student_output.reconstruction)
                semantic_loss = functional.cross_entropy(logits, labels)
                loss = loss + float(config.semantic_weight) * semantic_loss
                semantic_correct += int(
                    (logits.argmax(dim=1) == labels).sum().item()
                )
                semantic_examples += len(labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_examples = len(images)
            totals["reconstruction_loss"] += (
                float(reconstruction_loss.detach().cpu()) * batch_examples
            )
            totals["prior_kl_loss"] += (
                float(prior_kl_loss.detach().cpu()) * batch_examples
            )
            totals["semantic_loss"] += (
                float(semantic_loss.detach().cpu()) * batch_examples
            )
            totals["output_kd_loss"] += (
                float(output_kd_loss.detach().cpu()) * batch_examples
            )
            totals["encoder_feature_kd_loss"] += (
                float(encoder_feature_loss.detach().cpu()) * batch_examples
            )
            totals["decoder_feature_kd_loss"] += (
                float(decoder_feature_loss.detach().cpu()) * batch_examples
            )
            totals["total_loss"] += float(loss.detach().cpu()) * batch_examples
            processed_examples += batch_examples
            steps += 1

    if steps == 0:
        raise ValueError("retained client data must contain at least one batch")
    metrics = {
        name: value / processed_examples for name, value in totals.items()
    }
    metrics.update(
        {
            "semantic_accuracy": (
                semantic_correct / semantic_examples
                if semantic_examples
                else None
            ),
            "examples": len(loader.dataset),
            "steps": steps,
            "output_kd_weight": float(config.output_kd_weight),
            "feature_kd_weight": float(config.feature_kd_weight),
            "teacher_scope": (
                "original_global_on_local_retained_data_only"
                if uses_teacher
                else "none"
            ),
            "channel_pairing": (
                "same_standard_awgn_draw" if uses_teacher else "standard_training_awgn"
            ),
            "latent_source": "normalized_posterior_mean",
            "reconstruction_reduction": config.reconstruction_reduction,
            "uses_target_data": False,
        }
    )
    return copy_model_state(local_model), metrics
