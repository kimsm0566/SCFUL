"""Loss-based black-box membership inference audits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


class ReconstructionMIAConfig:
    """Class-matched, holdout-calibrated reconstruction MIA의 설정."""

    def __init__(
        self,
        batch_size=256,
        max_examples_per_class=1000,
        min_examples_per_class=20,
        calibration_fraction=0.5,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.max_examples_per_class = max_examples_per_class
        self.min_examples_per_class = min_examples_per_class
        self.calibration_fraction = calibration_fraction
        self.device = device


class ClassificationMIAConfig:
    """Class-matched, holdout-calibrated classification-loss MIA settings."""

    def __init__(
        self,
        batch_size=256,
        max_examples_per_class=1000,
        min_examples_per_class=20,
        calibration_fraction=0.5,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.max_examples_per_class = max_examples_per_class
        self.min_examples_per_class = min_examples_per_class
        self.calibration_fraction = calibration_fraction
        self.device = device


def _normalize_images(images):
    images = torch.as_tensor(images, dtype=torch.float32)
    if images.ndim != 4:
        raise ValueError("membership audit images must be a four-dimensional tensor")
    if images.max() > 1.0:
        images = images / 255.0
    return images


def _validate_config(config):
    if config.batch_size <= 0:
        raise ValueError("MIA batch_size must be positive")
    if config.max_examples_per_class <= 0:
        raise ValueError("MIA max_examples_per_class must be positive")
    if config.min_examples_per_class < 2:
        raise ValueError("MIA min_examples_per_class must be at least two")
    if not 0.0 < config.calibration_fraction < 1.0:
        raise ValueError("MIA calibration_fraction must be in (0, 1)")


def _class_matched_partitions(member_labels, nonmember_labels, config, seed):
    """Member/non-member를 class별 1:1로 맞춰 calibration/evaluation으로 나눈다."""
    member_labels = np.asarray(member_labels, dtype=np.int64)
    nonmember_labels = np.asarray(nonmember_labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    calibration_member = []
    calibration_nonmember = []
    evaluation_member = []
    evaluation_nonmember = []
    class_records = []

    for label in np.intersect1d(np.unique(member_labels), np.unique(nonmember_labels)):
        member_indices = np.flatnonzero(member_labels == label)
        nonmember_indices = np.flatnonzero(nonmember_labels == label)
        count = min(
            len(member_indices),
            len(nonmember_indices),
            config.max_examples_per_class,
        )
        calibration_count = int(np.floor(count * config.calibration_fraction))
        evaluation_count = count - calibration_count
        if (
            count < config.min_examples_per_class
            or calibration_count == 0
            or evaluation_count == 0
        ):
            continue

        rng.shuffle(member_indices)
        rng.shuffle(nonmember_indices)
        member_indices = member_indices[:count]
        nonmember_indices = nonmember_indices[:count]
        calibration_member.extend(member_indices[:calibration_count].tolist())
        calibration_nonmember.extend(nonmember_indices[:calibration_count].tolist())
        evaluation_member.extend(member_indices[calibration_count:].tolist())
        evaluation_nonmember.extend(nonmember_indices[calibration_count:].tolist())
        class_records.append(
            {
                "label": int(label),
                "paired_examples_per_side": int(count),
                "calibration_examples_per_side": int(calibration_count),
                "evaluation_examples_per_side": int(evaluation_count),
            }
        )

    if not class_records:
        raise ValueError("MIA has no class with enough member/non-member examples")
    return {
        "calibration_member": np.asarray(calibration_member, dtype=np.int64),
        "calibration_nonmember": np.asarray(calibration_nonmember, dtype=np.int64),
        "evaluation_member": np.asarray(evaluation_member, dtype=np.int64),
        "evaluation_nonmember": np.asarray(evaluation_nonmember, dtype=np.int64),
        "class_records": class_records,
    }


def _reconstruction_errors(model, images, batch_size, device):
    errors = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            reconstruction = model.decode_posterior_mean(batch).reconstruction
            errors.append(
                (reconstruction - batch)
                .flatten(start_dim=1)
                .square()
                .mean(dim=1)
                .cpu()
            )
    return torch.cat(errors).numpy()


def _classification_cross_entropy_losses(
    model, images, labels, batch_size, device
):
    losses = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size].to(device)
            batch_labels = torch.as_tensor(
                labels[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = model(batch_images)
            if logits.ndim != 2 or len(logits) != len(batch_labels):
                raise ValueError(
                    "classification MIA model must return (B, classes) logits"
                )
            losses.append(
                functional.cross_entropy(
                    logits, batch_labels, reduction="none"
                ).cpu()
            )
    losses = torch.cat(losses)
    if not bool(torch.isfinite(losses).all()):
        raise RuntimeError("classification MIA produced non-finite CE losses")
    return losses.numpy()


def _roc_auc(scores, labels):
    """점수 클수록 member일 때의 tie-aware ROC-AUC를 계산한다."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC-AUC needs both member and non-member examples")

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
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _classification_metrics(predictions, labels):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    positive = labels == 1
    negative = ~positive
    tpr = float(predictions[positive].mean())
    fpr = float(predictions[negative].mean())
    accuracy = float((predictions == positive).mean())
    return {
        "attack_accuracy": accuracy,
        "attack_balanced_accuracy": 0.5 * (tpr + 1.0 - fpr),
        "attack_tpr": tpr,
        "attack_fpr": fpr,
        "attack_advantage": tpr - fpr,
    }


def _fit_loss_threshold(member_losses, nonmember_losses):
    """별도 calibration subset에서 threshold 및 공격 방향을 선택한다."""
    losses = np.concatenate((member_losses, nonmember_losses))
    labels = np.concatenate(
        (
            np.ones(len(member_losses), dtype=np.int64),
            np.zeros(len(nonmember_losses), dtype=np.int64),
        )
    )
    values = np.unique(losses)
    thresholds = np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            (values[:-1] + values[1:]) / 2.0,
            [np.nextafter(values[-1], np.inf)],
        )
    )
    best = None
    for member_if_lower in (True, False):
        for threshold in thresholds:
            predictions = losses <= threshold if member_if_lower else losses >= threshold
            metrics = _classification_metrics(predictions, labels)
            score = (
                metrics["attack_balanced_accuracy"],
                metrics["attack_advantage"],
                int(member_if_lower),
                -float(threshold),
            )
            if best is None or score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "member_if_lower_loss": bool(member_if_lower),
                    "calibration": metrics,
                    "score": score,
                }
    return best


def _evaluate_threshold(member_losses, nonmember_losses, threshold, member_if_lower):
    losses = np.concatenate((member_losses, nonmember_losses))
    labels = np.concatenate(
        (
            np.ones(len(member_losses), dtype=np.int64),
            np.zeros(len(nonmember_losses), dtype=np.int64),
        )
    )
    predictions = losses <= threshold if member_if_lower else losses >= threshold
    return _classification_metrics(predictions, labels)


def evaluate_reconstruction_membership_inference(
    model_factory,
    model_states,
    member_data,
    nonmember_data,
    config,
    seed,
):
    """Reconstruction-loss black-box MIA를 각 checkpoint에 동일하게 적용한다.

    ``member_data``는 삭제 client의 원래 local training data, ``nonmember_data``는
    어느 training/compensation 단계에도 사용되지 않은 holdout test data다. Label은
    class matching에만 사용하며 공격 feature에는 deterministic reconstruction MSE 하나만
    사용한다.
    """
    _validate_config(config)
    member_images, member_labels = member_data
    nonmember_images, nonmember_labels = nonmember_data
    member_images = _normalize_images(member_images)
    nonmember_images = _normalize_images(nonmember_images)
    member_labels = np.asarray(member_labels, dtype=np.int64)
    nonmember_labels = np.asarray(nonmember_labels, dtype=np.int64)
    if len(member_images) != len(member_labels) or len(nonmember_images) != len(nonmember_labels):
        raise ValueError("MIA image/label lengths must match")

    partitions = _class_matched_partitions(
        member_labels, nonmember_labels, config, seed
    )
    device = torch.device(config.device)
    summary_records = []
    class_records = []
    for model_name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()

        calibration_member_losses = _reconstruction_errors(
            model,
            member_images[partitions["calibration_member"]],
            config.batch_size,
            device,
        )
        calibration_nonmember_losses = _reconstruction_errors(
            model,
            nonmember_images[partitions["calibration_nonmember"]],
            config.batch_size,
            device,
        )
        evaluation_member_losses = _reconstruction_errors(
            model,
            member_images[partitions["evaluation_member"]],
            config.batch_size,
            device,
        )
        evaluation_nonmember_losses = _reconstruction_errors(
            model,
            nonmember_images[partitions["evaluation_nonmember"]],
            config.batch_size,
            device,
        )
        fitted = _fit_loss_threshold(
            calibration_member_losses, calibration_nonmember_losses
        )
        evaluation = _evaluate_threshold(
            evaluation_member_losses,
            evaluation_nonmember_losses,
            fitted["threshold"],
            fitted["member_if_lower_loss"],
        )
        evaluation_labels = np.concatenate(
            (
                np.ones(len(evaluation_member_losses), dtype=np.int64),
                np.zeros(len(evaluation_nonmember_losses), dtype=np.int64),
            )
        )
        auc_low_loss_member = _roc_auc(
            -np.concatenate((evaluation_member_losses, evaluation_nonmember_losses)),
            evaluation_labels,
        )
        summary_records.append(
            {
                "model": model_name,
                "score_feature": "deterministic_reconstruction_mse",
                "attack_calibration": "separate_class_matched_half",
                "evaluation_member_examples": int(len(evaluation_member_losses)),
                "evaluation_nonmember_examples": int(len(evaluation_nonmember_losses)),
                "member_mse_mean": float(evaluation_member_losses.mean()),
                "nonmember_mse_mean": float(evaluation_nonmember_losses.mean()),
                "mia_auc_low_loss_member": auc_low_loss_member,
                "mia_auc_best_direction": max(auc_low_loss_member, 1.0 - auc_low_loss_member),
                "threshold": fitted["threshold"],
                "decision_rule": (
                    "member_if_loss_leq_threshold"
                    if fitted["member_if_lower_loss"]
                    else "member_if_loss_geq_threshold"
                ),
                "calibration_balanced_accuracy": fitted["calibration"][
                    "attack_balanced_accuracy"
                ],
                **evaluation,
            }
        )

        for class_record in partitions["class_records"]:
            label = class_record["label"]
            member_positions = partitions["evaluation_member"][
                member_labels[partitions["evaluation_member"]] == label
            ]
            nonmember_positions = partitions["evaluation_nonmember"][
                nonmember_labels[partitions["evaluation_nonmember"]] == label
            ]
            member_losses = _reconstruction_errors(
                model, member_images[member_positions], config.batch_size, device
            )
            nonmember_losses = _reconstruction_errors(
                model, nonmember_images[nonmember_positions], config.batch_size, device
            )
            per_class = _evaluate_threshold(
                member_losses,
                nonmember_losses,
                fitted["threshold"],
                fitted["member_if_lower_loss"],
            )
            labels = np.concatenate(
                (
                    np.ones(len(member_losses), dtype=np.int64),
                    np.zeros(len(nonmember_losses), dtype=np.int64),
                )
            )
            auc = _roc_auc(-np.concatenate((member_losses, nonmember_losses)), labels)
            class_records.append(
                {
                    "model": model_name,
                    **class_record,
                    "member_mse_mean": float(member_losses.mean()),
                    "nonmember_mse_mean": float(nonmember_losses.mean()),
                    "mia_auc_low_loss_member": auc,
                    "mia_auc_best_direction": max(auc, 1.0 - auc),
                    **per_class,
                }
            )
        del model

    metadata = {
        "attack_type": "black_box_reconstruction_loss_threshold",
        "score_feature": "deterministic posterior-mean reconstruction MSE",
        "member_source": "target client original local training data",
        "nonmember_source": "unseen CIFAR-10 test data",
        "class_matching": "member and non-member counts are equal within each included class",
        "threshold_calibration": "independent class-matched half; reported attack metrics use the other half",
        "included_classes": partitions["class_records"],
        "calibration_member_examples": int(len(partitions["calibration_member"])),
        "evaluation_member_examples": int(len(partitions["evaluation_member"])),
    }
    return summary_records, class_records, metadata


def evaluate_classification_membership_inference(
    model_factory,
    model_states,
    member_data,
    nonmember_data,
    config,
    seed,
):
    """Evaluate per-example CE-loss membership inference for classifiers.

    ``member_data`` must contain the target client's clean, pre-poison images
    and true labels with every recorded poisoned position removed. The
    non-member side must be unseen clean data. Labels are used both for the
    cross-entropy score and for class matching.
    """

    _validate_config(config)
    if not model_states:
        raise ValueError("classification MIA requires at least one model state")
    member_images, member_labels = member_data
    nonmember_images, nonmember_labels = nonmember_data
    member_images = _normalize_images(member_images)
    nonmember_images = _normalize_images(nonmember_images)
    member_labels = np.asarray(member_labels, dtype=np.int64)
    nonmember_labels = np.asarray(nonmember_labels, dtype=np.int64)
    if len(member_images) != len(member_labels) or len(nonmember_images) != len(
        nonmember_labels
    ):
        raise ValueError("MIA image/label lengths must match")

    partitions = _class_matched_partitions(
        member_labels, nonmember_labels, config, seed
    )
    device = torch.device(config.device)
    summary_records = []
    class_records = []
    for model_name, state in model_states.items():
        model = model_factory().to(device)
        model.load_state_dict(state)
        model.eval()

        calibration_member_losses = _classification_cross_entropy_losses(
            model,
            member_images[partitions["calibration_member"]],
            member_labels[partitions["calibration_member"]],
            config.batch_size,
            device,
        )
        calibration_nonmember_losses = _classification_cross_entropy_losses(
            model,
            nonmember_images[partitions["calibration_nonmember"]],
            nonmember_labels[partitions["calibration_nonmember"]],
            config.batch_size,
            device,
        )
        evaluation_member_losses = _classification_cross_entropy_losses(
            model,
            member_images[partitions["evaluation_member"]],
            member_labels[partitions["evaluation_member"]],
            config.batch_size,
            device,
        )
        evaluation_nonmember_losses = _classification_cross_entropy_losses(
            model,
            nonmember_images[partitions["evaluation_nonmember"]],
            nonmember_labels[partitions["evaluation_nonmember"]],
            config.batch_size,
            device,
        )
        fitted = _fit_loss_threshold(
            calibration_member_losses, calibration_nonmember_losses
        )
        evaluation = _evaluate_threshold(
            evaluation_member_losses,
            evaluation_nonmember_losses,
            fitted["threshold"],
            fitted["member_if_lower_loss"],
        )
        evaluation_labels = np.concatenate(
            (
                np.ones(len(evaluation_member_losses), dtype=np.int64),
                np.zeros(len(evaluation_nonmember_losses), dtype=np.int64),
            )
        )
        auc_low_loss_member = _roc_auc(
            -np.concatenate(
                (evaluation_member_losses, evaluation_nonmember_losses)
            ),
            evaluation_labels,
        )
        summary_records.append(
            {
                "model": model_name,
                "score_feature": "per_example_cross_entropy",
                "attack_calibration": "separate_class_matched_half",
                "evaluation_member_examples": int(len(evaluation_member_losses)),
                "evaluation_nonmember_examples": int(
                    len(evaluation_nonmember_losses)
                ),
                "member_cross_entropy_mean": float(
                    evaluation_member_losses.mean()
                ),
                "nonmember_cross_entropy_mean": float(
                    evaluation_nonmember_losses.mean()
                ),
                "mia_auc_low_loss_member": auc_low_loss_member,
                "mia_auc_best_direction": max(
                    auc_low_loss_member, 1.0 - auc_low_loss_member
                ),
                "threshold": fitted["threshold"],
                "decision_rule": (
                    "member_if_loss_leq_threshold"
                    if fitted["member_if_lower_loss"]
                    else "member_if_loss_geq_threshold"
                ),
                "calibration_balanced_accuracy": fitted["calibration"][
                    "attack_balanced_accuracy"
                ],
                **evaluation,
            }
        )

        for class_record in partitions["class_records"]:
            label = class_record["label"]
            member_positions = partitions["evaluation_member"][
                member_labels[partitions["evaluation_member"]] == label
            ]
            nonmember_positions = partitions["evaluation_nonmember"][
                nonmember_labels[partitions["evaluation_nonmember"]] == label
            ]
            member_losses = _classification_cross_entropy_losses(
                model,
                member_images[member_positions],
                member_labels[member_positions],
                config.batch_size,
                device,
            )
            nonmember_losses = _classification_cross_entropy_losses(
                model,
                nonmember_images[nonmember_positions],
                nonmember_labels[nonmember_positions],
                config.batch_size,
                device,
            )
            per_class = _evaluate_threshold(
                member_losses,
                nonmember_losses,
                fitted["threshold"],
                fitted["member_if_lower_loss"],
            )
            labels = np.concatenate(
                (
                    np.ones(len(member_losses), dtype=np.int64),
                    np.zeros(len(nonmember_losses), dtype=np.int64),
                )
            )
            auc = _roc_auc(
                -np.concatenate((member_losses, nonmember_losses)), labels
            )
            class_records.append(
                {
                    "model": model_name,
                    **class_record,
                    "member_cross_entropy_mean": float(member_losses.mean()),
                    "nonmember_cross_entropy_mean": float(
                        nonmember_losses.mean()
                    ),
                    "mia_auc_low_loss_member": auc,
                    "mia_auc_best_direction": max(auc, 1.0 - auc),
                    **per_class,
                }
            )
        del model

    metadata = {
        "attack_type": "black_box_classification_loss_threshold",
        "score_feature": "per-example true-label cross entropy",
        "member_source": (
            "target client clean pre-poison training images and true labels, "
            "with recorded poisoned positions excluded"
        ),
        "nonmember_source": "unseen clean CIFAR-10 test data",
        "class_matching": (
            "member and non-member counts are equal within each included class"
        ),
        "threshold_calibration": (
            "independent class-matched half; reported attack metrics use the other half"
        ),
        "included_classes": partitions["class_records"],
        "calibration_member_examples": int(len(partitions["calibration_member"])),
        "evaluation_member_examples": int(len(partitions["evaluation_member"])),
    }
    return summary_records, class_records, metadata
