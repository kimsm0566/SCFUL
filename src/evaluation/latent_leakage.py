"""Class-matched IDI evaluation for semantic-codec encoder representations."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset


class LatentLeakageConfig:
    """Training and feature settings for the layer-wise IDI critics."""

    def __init__(
        self,
        batch_size=128,
        critic_epochs=50,
        repetitions=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        hidden_dim=128,
        embedding_dim=32,
        temperature=0.2,
        spatial_pool_size=4,
        feature_layers=None,
        device="cpu",
    ):
        self.batch_size = int(batch_size)
        self.critic_epochs = int(critic_epochs)
        self.repetitions = int(repetitions)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.temperature = float(temperature)
        self.spatial_pool_size = int(spatial_pool_size)
        self.feature_layers = (
            None
            if feature_layers is None
            else tuple(str(name) for name in feature_layers)
        )
        self.device = str(device)
        self._validate()

    def _validate(self):
        integer_fields = (
            "batch_size",
            "critic_epochs",
            "repetitions",
            "hidden_dim",
            "embedding_dim",
            "spatial_pool_size",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"latent leakage {name} must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("latent leakage learning_rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("latent leakage weight_decay must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("latent leakage temperature must be positive")
        if self.feature_layers is not None and (
            not self.feature_layers or len(set(self.feature_layers)) != len(self.feature_layers)
        ):
            raise ValueError("IDI feature_layers must be a nonempty unique sequence")


class BinaryInfoNCECritic(nn.Module):
    """InfoNCE critic with two learned label embeddings."""

    def __init__(self, input_dim, hidden_dim, embedding_dim, temperature):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )
        self.label_embeddings = nn.Parameter(torch.empty(2, int(embedding_dim)))
        nn.init.normal_(self.label_embeddings, std=0.02)
        self.temperature = float(temperature)

    def forward(self, features):
        mapped = functional.normalize(self.feature_map(features), dim=1)
        labels = functional.normalize(self.label_embeddings, dim=1)
        return mapped @ labels.t() / self.temperature


def _binary_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("client leakage AUC requires both memberships")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _classification_metrics(logits, labels):
    probabilities = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
    labels_numpy = labels.detach().cpu().numpy()
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    positive = labels_numpy == 1
    negative = ~positive
    true_positive_rate = float((predictions[positive] == 1).mean())
    true_negative_rate = float((predictions[negative] == 0).mean())
    return {
        "accuracy": float((predictions == labels_numpy).mean()),
        "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "roc_auc": _binary_auc(probabilities, labels_numpy),
    }


def _standardize(train_features, evaluation_features):
    mean = train_features.mean(dim=0, keepdim=True)
    standard_deviation = train_features.std(dim=0, unbiased=False, keepdim=True)
    standard_deviation = standard_deviation.clamp_min(1e-6)
    return (
        (train_features - mean) / standard_deviation,
        (evaluation_features - mean) / standard_deviation,
    )


def _train_binary_head(
    train_features,
    train_labels,
    evaluation_features,
    evaluation_labels,
    config,
    seed,
):
    device = torch.device(config.device)
    train_features, evaluation_features = _standardize(
        train_features.float(), evaluation_features.float()
    )
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    head = BinaryInfoNCECritic(
        train_features.shape[1],
        config.hidden_dim,
        config.embedding_dim,
        config.temperature,
    )
    epochs = config.critic_epochs
    head = head.to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    loader = DataLoader(
        TensorDataset(train_features, train_labels.long()),
        batch_size=min(config.batch_size, len(train_features)),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    head.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            logits = head(batch_features.to(device))
            loss = functional.cross_entropy(logits, batch_labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        evaluation_logits = head(evaluation_features.to(device)).cpu()
        cross_entropy = float(
            functional.cross_entropy(
                evaluation_logits, evaluation_labels.long()
            ).item()
        )
    metrics = _classification_metrics(evaluation_logits, evaluation_labels.long())
    metrics["cross_entropy_nats"] = cross_entropy
    metrics["info_nce_mi_lower_bound_nats"] = max(
        0.0, math.log(2.0) - cross_entropy
    )
    return metrics


def _compress_feature(feature, pool_size, preserve_spatial_resolution=False):
    feature = feature.detach().cpu().float()
    if feature.ndim == 4:
        if preserve_spatial_resolution:
            feature = feature.flatten(start_dim=1)
        else:
            feature = functional.adaptive_avg_pool2d(
                feature, (int(pool_size), int(pool_size))
            ).flatten(start_dim=1)
    elif feature.ndim != 2:
        raise ValueError("encoder leakage features must be rank two or four")
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("encoder leakage features contain non-finite values")
    return feature


def _extract_encoder_features(model, images, batch_size, device, pool_size):
    chunks = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)].to(device)
            feature_dict = model(batch, mode="encoder_features")
            for name, feature in feature_dict.items():
                chunks.setdefault(name, []).append(
                    _compress_feature(
                        feature,
                        pool_size,
                        preserve_spatial_resolution=name
                        in {
                            "posterior_mu",
                            "posterior_logvar",
                            "canonical_latent",
                        },
                    )
                )
    return {name: torch.cat(values) for name, values in chunks.items()}


def _mean_standard_error(values):
    values = np.asarray(values, dtype=np.float64)
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return float(values.mean()), standard_error


def summarize_idi(layer_rows, original_name, retrain_name):
    """Normalize layer-summed MI gaps to Original-minus-Retrain."""

    models = list(dict.fromkeys(row["model"] for row in layer_rows))
    layers = list(dict.fromkeys(row["layer"] for row in layer_rows))
    values = {
        (row["model"], row["layer"]): float(row["mi_lower_bound_nats"])
        for row in layer_rows
    }
    if original_name not in models or retrain_name not in models:
        raise ValueError("IDI requires Original and Retrain model rows")
    for model in models:
        if any((model, layer) not in values for layer in layers):
            raise ValueError("Client-IDI rows must cover the same layers per model")
    retrain_total = sum(values[(retrain_name, layer)] for layer in layers)
    original_total = sum(values[(original_name, layer)] for layer in layers)
    denominator = original_total - retrain_total
    stable = denominator > 1e-6
    rows = []
    for model in models:
        total = sum(values[(model, layer)] for layer in layers)
        difference = total - retrain_total
        rows.append(
            {
                "model": model,
                "layers": len(layers),
                "layer_mi_sum_nats": total,
                "information_difference_to_retrain_nats": difference,
                "original_minus_retrain_denominator_nats": denominator,
                "idi": difference / denominator if stable else None,
                "status": "ok" if stable else "nonpositive_or_tiny_denominator",
                "reference_model": retrain_name,
                "original_model": original_name,
            }
        )
    return rows


def evaluate_encoder_idi(
    model_factory,
    model_states,
    probe_data,
    config,
    seed,
    original_name,
    retrain_name,
):
    """Evaluate the published layer-wise InfoNCE IDI construction."""

    if original_name not in model_states or retrain_name not in model_states:
        raise ValueError("latent leakage evaluation requires Original and Retrain")
    images = torch.as_tensor(probe_data["images"], dtype=torch.float32)
    if images.max() > 1.0:
        images = images / 255.0
    membership = torch.as_tensor(probe_data["membership"], dtype=torch.long)
    partitions = np.asarray(probe_data["partitions"], dtype=object)
    if len(images) != len(membership) or len(images) != len(partitions):
        raise ValueError("latent leakage probe arrays must align")
    training_indices = np.flatnonzero(partitions == "training")
    evaluation_indices = np.flatnonzero(partitions == "evaluation")
    if min(len(training_indices), len(evaluation_indices)) == 0:
        raise ValueError("latent leakage needs training and evaluation examples")
    training_indices_tensor = torch.as_tensor(training_indices, dtype=torch.long)
    evaluation_indices_tensor = torch.as_tensor(
        evaluation_indices, dtype=torch.long
    )
    device = torch.device(config.device)
    critic_repeat_rows = []
    layer_rows = []
    layer_order = None
    for model_position, (model_name, state) in enumerate(model_states.items()):
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()
        extracted = _extract_encoder_features(
            model,
            images,
            config.batch_size,
            device,
            config.spatial_pool_size,
        )
        if config.feature_layers is not None:
            missing = [
                name for name in config.feature_layers if name not in extracted
            ]
            if missing:
                raise ValueError(f"IDI feature layers are missing: {missing}")
            extracted = {
                name: extracted[name] for name in config.feature_layers
            }
        del model
        if layer_order is None:
            layer_order = list(extracted)
        elif list(extracted) != layer_order:
            raise RuntimeError("encoder feature layers changed between models")
        for layer_position, layer_name in enumerate(layer_order):
            layer_features = extracted[layer_name]
            repetition_metrics = []
            for repetition in range(config.repetitions):
                repetition_seed = int(seed) + 10000 * model_position + 100 * layer_position + repetition
                metrics = _train_binary_head(
                    layer_features[training_indices_tensor],
                    membership[training_indices_tensor],
                    layer_features[evaluation_indices_tensor],
                    membership[evaluation_indices_tensor],
                    config,
                    repetition_seed,
                )
                repetition_metrics.append(metrics)
                critic_repeat_rows.append(
                    {
                        "model": model_name,
                        "layer": layer_name,
                        "repetition": repetition,
                        "seed": repetition_seed,
                        **metrics,
                    }
                )
            mi_mean, mi_se = _mean_standard_error(
                [row["info_nce_mi_lower_bound_nats"] for row in repetition_metrics]
            )
            accuracy_mean, accuracy_se = _mean_standard_error(
                [row["accuracy"] for row in repetition_metrics]
            )
            auc_mean, auc_se = _mean_standard_error(
                [row["roc_auc"] for row in repetition_metrics]
            )
            layer_rows.append(
                {
                    "model": model_name,
                    "layer": layer_name,
                    "layer_index": layer_position,
                    "feature_dimensions": int(layer_features.shape[1]),
                    "mi_lower_bound_nats": mi_mean,
                    "mi_standard_error": mi_se,
                    "critic_accuracy": accuracy_mean,
                    "critic_accuracy_standard_error": accuracy_se,
                    "critic_roc_auc": auc_mean,
                    "critic_roc_auc_standard_error": auc_se,
                    "training_examples": int(len(training_indices)),
                    "evaluation_examples": int(len(evaluation_indices)),
                }
            )
        del extracted

    idi_rows = summarize_idi(layer_rows, original_name, retrain_name)

    protocol = {
        **dict(probe_data["metadata"]),
        "semantic_class_rows": probe_data["class_rows"],
        "feature_layers": layer_order,
        "spatial_feature_compression": (
            f"intermediate encoder blocks use adaptive average pooling to "
            f"{config.spatial_pool_size}x{config.spatial_pool_size}; posterior "
            "and canonical latent maps retain full spatial resolution"
        ),
        "posterior_features": "raw flattened posterior mean/log-variance and power-normalized mean",
        "idi_estimator": "binary-label InfoNCE lower bound, log(2)-held-out cross-entropy, clipped at zero",
        "idi_normalization": "sum_layer(MI_model-MI_retrain) / sum_layer(MI_original-MI_retrain)",
        "critic_epochs": config.critic_epochs,
        "repetitions": config.repetitions,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "temperature": config.temperature,
        "seed": int(seed),
        "reference_model": retrain_name,
        "original_model": original_name,
        "warning": "empirical white-box IDI estimator; class matching reduces semantic-class confounding but cannot prove absence of client information",
    }
    return {
        "layer_rows": layer_rows,
        "idi_rows": idi_rows,
        "critic_repeat_rows": critic_repeat_rows,
        "protocol": protocol,
    }
