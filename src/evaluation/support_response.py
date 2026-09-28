"""삭제 client의 class support 감소와 model behavior를 함께 기록한다."""

import numpy as np


class SupportGroupingConfig:
    """삭제 비율로 class support 구간을 나누는 기준이다."""

    def __init__(
        self,
        min_target_samples=500,
        shared_high_max_deletion_fraction=0.20,
        low_support_min_deletion_fraction=0.60,
    ):
        self.min_target_samples = min_target_samples
        self.shared_high_max_deletion_fraction = shared_high_max_deletion_fraction
        self.low_support_min_deletion_fraction = low_support_min_deletion_fraction


def _validate_support_inputs(class_labels, train_allocation, target_client_id, config):
    class_labels = np.asarray(class_labels, dtype=np.int64)
    train_allocation = np.asarray(train_allocation, dtype=np.int64)
    if class_labels.ndim != 1 or len(class_labels) == 0:
        raise ValueError("class_labels must be a non-empty one-dimensional array")
    if train_allocation.ndim != 2:
        raise ValueError("train_allocation must be a two-dimensional array")
    if train_allocation.shape[1] != len(class_labels):
        raise ValueError("train_allocation columns must match class_labels")
    if not 0 <= target_client_id < train_allocation.shape[0]:
        raise ValueError("target_client_id must identify a row in train_allocation")
    if np.any(train_allocation < 0):
        raise ValueError("train_allocation cannot contain negative counts")
    if config.min_target_samples < 1:
        raise ValueError("min_target_samples must be positive")
    if not 0.0 <= config.shared_high_max_deletion_fraction <= 1.0:
        raise ValueError("shared_high_max_deletion_fraction must be between zero and one")
    if not 0.0 <= config.low_support_min_deletion_fraction <= 1.0:
        raise ValueError("low_support_min_deletion_fraction must be between zero and one")
    if (
        config.shared_high_max_deletion_fraction
        >= config.low_support_min_deletion_fraction
    ):
        raise ValueError("shared_high threshold must be smaller than low_support threshold")
    return class_labels, train_allocation


def _support_group(target_count, retained_count, deletion_fraction, config):
    if target_count > 0 and retained_count == 0:
        return "target_specific"
    if target_count < config.min_target_samples:
        return "not_targeted"
    if deletion_fraction <= config.shared_high_max_deletion_fraction:
        return "shared_high"
    if deletion_fraction >= config.low_support_min_deletion_fraction:
        return "low_support"
    return "medium_support"


def build_class_support_records(
    class_labels, train_allocation, target_client_id, config=None
):
    """각 class에서 client 삭제로 사라지는 data support를 계산한다."""
    if config is None:
        config = SupportGroupingConfig()
    class_labels, train_allocation = _validate_support_inputs(
        class_labels, train_allocation, target_client_id, config
    )

    records = []
    for column, label in enumerate(class_labels):
        target_count = int(train_allocation[target_client_id, column])
        total_count = int(train_allocation[:, column].sum())
        if total_count <= 0:
            raise ValueError(f"class {label} has no training samples")
        retained_count = total_count - target_count
        deletion_fraction = target_count / total_count
        records.append(
            {
                "label": int(label),
                "target_train_samples": target_count,
                "retained_train_samples": retained_count,
                "total_train_samples": total_count,
                "deletion_fraction": deletion_fraction,
                "remaining_fraction": 1.0 - deletion_fraction,
                "support_group": _support_group(
                    target_count, retained_count, deletion_fraction, config
                ),
            }
        )
    return records


def merge_support_with_class_metrics(
    support_records,
    class_metric_records,
    reference_model="retrain_without_client_0",
):
    """Class별 model metric에 삭제 support와 재학습 MSE 차이를 붙인다."""
    support_by_label = {record["label"]: record for record in support_records}
    reference_mse = {}
    for metric in class_metric_records:
        if metric.get("group") != "class":
            continue
        if metric.get("model") == reference_model:
            reference_mse[int(metric["group_id"])] = float(metric["mse_input"])

    records = []
    for metric in class_metric_records:
        if metric.get("group") != "class":
            continue
        label = int(metric["group_id"])
        if label not in support_by_label:
            raise ValueError(f"class metric label {label} is missing from support records")
        if label not in reference_mse:
            raise ValueError(
                f"reference model {reference_model} is missing class metric for label {label}"
            )
        record = dict(support_by_label[label])
        record.update(metric)
        record["test_samples"] = int(metric["samples"])
        record["mse_delta_to_retrain"] = float(metric["mse_input"]) - reference_mse[label]
        records.append(record)
    if not records:
        raise ValueError("class_metric_records must contain at least one class metric")
    return records


def _weighted_mean_and_std(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return mean, float(np.sqrt(variance))


def summarize_support_groups(records, candidate_models=None):
    """Model과 support 구간별 class metric 평균·표준편차를 계산한다."""
    if candidate_models is None:
        candidate_models = sorted({record["model"] for record in records})
    metric_names = (
        "mse_input",
        "psnr",
        "fg_output",
        "fg_mu",
        "fg_latent",
        "mse_delta_to_retrain",
    )
    group_order = (
        "shared_high",
        "medium_support",
        "low_support",
        "target_specific",
    )
    summary = []
    for model in candidate_models:
        for support_group in group_order:
            group_records = [
                record
                for record in records
                if record["model"] == model
                and record["support_group"] == support_group
            ]
            if not group_records:
                continue
            weights = [record["test_samples"] for record in group_records]
            deletion_mean, deletion_std = _weighted_mean_and_std(
                [record["deletion_fraction"] for record in group_records], weights
            )
            row = {
                "model": model,
                "support_group": support_group,
                "label_count": len(group_records),
                "test_samples": int(sum(weights)),
                "deletion_fraction_mean": deletion_mean,
                "deletion_fraction_std": deletion_std,
            }
            for metric_name in metric_names:
                mean, std = _weighted_mean_and_std(
                    [record[metric_name] for record in group_records], weights
                )
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_std"] = std
            summary.append(row)
    return summary
