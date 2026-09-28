"""Oracle Behavior artifact용 비대화형 figure 생성기."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.posterior_audit import ClientGaussianSummary


def _image_for_plot(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 3:
        return np.moveaxis(array, 0, -1), {}
    return array.squeeze(), {"cmap": "gray", "vmin": 0, "vmax": 1}


def _plot_reconstruction_grid(
    predictions: dict[str, Any], output_path: Path, probes_per_group: int
) -> None:
    target = predictions["target_train_member"]
    references = target["references"]
    ordered: list[int] = []
    for group in ("shared_high", "target_specific"):
        ordered.extend(
            [
                index
                for index, reference in enumerate(references)
                if reference["support_group"] == group
            ][:probes_per_group]
        )
    columns = ["input", "original", "retrain", "scu_forget_only", "scu_self_sharing"]
    figure, axes = plt.subplots(
        len(ordered), len(columns), figsize=(9, 1.8 * len(ordered)), squeeze=False
    )
    for row, position in enumerate(ordered):
        images = [target["inputs"][position]] + [
            target["models"][model_name]["posterior_mean_reconstruction"][position]
            for model_name in columns[1:]
        ]
        for column, image in enumerate(images):
            plotted_image, options = _image_for_plot(image)
            axes[row, column].imshow(plotted_image, **options)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(columns[column].replace("_", " "))
        axes[row, 0].set_ylabel(references[position]["support_group"], fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_latent_comparison(predictions: dict[str, Any], output_path: Path) -> None:
    target = predictions["target_train_member"]
    references = target["references"]
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), squeeze=False)
    for row, group in enumerate(("shared_high", "target_specific")):
        positions = [
            index
            for index, reference in enumerate(references)
            if reference["support_group"] == group
        ]
        retrain = target["models"]["retrain"]["mu"][positions, :2].numpy()
        for column, model_name in enumerate(("original", "scu_self_sharing")):
            source = target["models"][model_name]["mu"][positions, :2].numpy()
            axis = axes[row, column]
            axis.scatter(retrain[:, 0], retrain[:, 1], s=12, label="retrain", alpha=0.8)
            axis.quiver(
                source[:, 0],
                source[:, 1],
                retrain[:, 0] - source[:, 0],
                retrain[:, 1] - source[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.002,
                alpha=0.35,
            )
            axis.set_title(f"{group}: {model_name} -> retrain")
            axis.set_xlabel("latent dimension 0")
            axis.set_ylabel("latent dimension 1")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_client_gaussian_diagnostic(
    summaries: list[ClientGaussianSummary], output_path: Path
) -> None:
    clients = [summary.client_id for summary in summaries]
    positions = np.arange(len(clients))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(
        positions - 0.2,
        [summary.metrics["mmd_single"] for summary in summaries],
        width=0.4,
        label="single",
    )
    axes[0].bar(
        positions + 0.2,
        [summary.metrics["mmd_empirical_noise"] for summary in summaries],
        width=0.4,
        label="empirical noise",
    )
    axes[0].set_title("RBF MMD²")
    axes[0].legend()
    axes[1].bar(
        positions,
        [summary.metrics["mean_abs_offdiag_correlation"] for summary in summaries],
    )
    axes[1].set_title("mean |off-diagonal correlation|")
    axes[2].bar(
        positions,
        [summary.metrics["label_mixture_nll_gain_per_dim"] for summary in summaries],
    )
    axes[2].set_title("label-mixture NLL gain / dim")
    for axis in axes:
        axis.set_xticks(positions, clients, rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_oracle_figures(
    predictions: dict[str, Any],
    client_summaries: list[ClientGaussianSummary],
    output_dir: Path,
    probes_per_group: int,
) -> None:
    """Oracle Behavior run에 필요한 세 figure를 동일 output directory에 저장한다."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_reconstruction_grid(
        predictions, output_dir / "reconstruction_grid.png", probes_per_group
    )
    _plot_latent_comparison(predictions, output_dir / "latent_comparison.png")
    _plot_client_gaussian_diagnostic(
        client_summaries, output_dir / "client_gaussian_diagnostic.png"
    )


def save_federated_training_curve_figures(
    history, output_dir, split_name, phase_name=None
):
    """Save round-wise optimization and clean-test performance curves."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_prefix = str(split_name).lower()
    if phase_name is not None:
        split_prefix = f"{split_prefix}_{str(phase_name).lower()}"
    split_label = str(split_name).replace("_", " ").upper()
    training_rows = [
        row
        for row in history
        if int(row["round"]) > 0 and row.get("objective_loss") is not None
    ]
    performance_rows = [
        row for row in history if row.get("test_psnr_db") is not None
    ]
    if not training_rows:
        raise ValueError("training history has no loss rows")
    if not performance_rows:
        raise ValueError("training history has no performance rows")

    rounds = np.asarray([row["round"] for row in training_rows])
    reconstruction_weight = float(training_rows[0]["reconstruction_weight"])
    beta = float(training_rows[0]["beta"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), squeeze=False)
    objective_axis = axes[0, 0]
    objective_axis.plot(
        rounds,
        [row["objective_loss"] for row in training_rows],
        color="#1f77b4",
        linewidth=1.8,
        label="total objective",
    )
    objective_axis.set_title("Federated training objective")
    objective_axis.set_xlabel("Global round")
    objective_axis.set_ylabel("Weighted local objective")
    objective_axis.grid(alpha=0.25)
    objective_axis.legend()

    component_axis = axes[0, 1]
    component_axis.plot(
        rounds,
        [reconstruction_weight * row["reconstruction_loss"] for row in training_rows],
        label=f"reconstruction x {reconstruction_weight:g}",
        linewidth=1.7,
    )
    component_axis.plot(
        rounds,
        [beta * row["kl_loss"] for row in training_rows],
        label=f"KL x {beta:g}",
        linewidth=1.7,
    )
    component_axis.set_title("Objective components")
    component_axis.set_xlabel("Global round")
    component_axis.set_ylabel("Weighted contribution")
    component_axis.grid(alpha=0.25)
    component_axis.legend()
    figure.suptitle(f"{split_label} CIFAR-10 training loss")
    figure.tight_layout()
    figure.savefig(output_dir / f"{split_prefix}_training_loss.png", dpi=180)
    plt.close(figure)

    performance_rounds = np.asarray([row["round"] for row in performance_rows])
    panels = [
        ("test_psnr_db", "PSNR", "dB", "#2ca02c"),
        ("test_rmse_255", "RMSE-255", "0-255 scale", "#d62728"),
    ]
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.8 * len(panels), 4.2),
        squeeze=False,
    )
    for column, (field, title, ylabel, color) in enumerate(panels):
        axis = axes[0, column]
        values = [row[field] for row in performance_rows]
        axis.plot(
            performance_rounds,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
        )
        axis.set_title(title)
        axis.set_xlabel("Global round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"{split_label} CIFAR-10 clean-test performance at AWGN 20 dB"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{split_prefix}_training_performance.png", dpi=180
    )
    plt.close(figure)


def save_fscuv4_optimization_figures(
    joint_metrics, variant_metrics, output_dir
):
    """Save joint-forgetting, retained-repair, and conflict diagnostics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_history = joint_metrics.get("history", [])
    if not joint_history:
        raise ValueError("FSCUv4 joint-unlearning history is empty")

    epochs = [row["epoch"] for row in joint_history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    axes[0, 0].plot(epochs, [row["total_loss"] for row in joint_history])
    axes[0, 0].set_title("Joint forgetting objective")
    axes[0, 0].set_xlabel("Target-client epoch")
    axes[0, 0].set_ylabel("Objective")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(
        epochs,
        [row["target_mse"] for row in joint_history],
        label="normalized target MSE",
    )
    axes[0, 1].set_title("Target reconstruction damage")
    axes[0, 1].set_xlabel("Target-client epoch")
    axes[0, 1].set_ylabel("Normalized MSE")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()
    if "target_information_ratio" in joint_history[0]:
        axes[0, 2].plot(
            epochs,
            [100.0 * row["target_information_ratio"] for row in joint_history],
            marker="o",
            label="mean target prior-KL ratio",
        )
        threshold = 100.0 * float(
            joint_metrics.get("target_information_ratio_threshold", 0.0)
        )
        if threshold > 0.0:
            axes[0, 2].axhline(
                threshold,
                color="tab:red",
                linestyle="--",
                label=f"stop threshold ({threshold:g}%)",
            )
        axes[0, 2].set_ylabel("Percent of Original")
        axes[0, 2].set_title("Target VIB information rate")
    else:
        axes[0, 2].plot(
            epochs,
            [row["posterior_kl_to_original"] for row in joint_history],
            label="posterior KL to Original",
        )
        axes[0, 2].set_title("Posterior deviation")
    axes[0, 2].set_xlabel("Target-client epoch")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fscuv4_joint_forgetting.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    has_conflict_diagnostics = False
    for variant, metrics in variant_metrics.items():
        history = metrics.get("history", [])
        rounds = [row["round"] for row in history]
        axes[0, 0].plot(
            rounds,
            [row["total_loss"] for row in history],
            label=f"{variant}: total",
        )
        axes[0, 0].plot(
            rounds,
            [row["reconstruction_loss"] for row in history],
            label=f"{variant}: reconstruction",
        )
        axes[0, 1].plot(
            rounds,
            [row.get("beta_weighted_kl_loss", 0.0) for row in history],
            label=variant,
        )
        evaluation_rows = [
            row for row in history if row.get("test_psnr_db") is not None
        ]
        if evaluation_rows:
            axes[1, 0].plot(
                [row["round"] for row in evaluation_rows],
                [row["test_psnr_db"] for row in evaluation_rows],
                marker="o",
                label=f"{variant}: held-out PSNR",
            )
        if history and history[0]["aggregation"] == "conflict_aware_projection":
            has_conflict_diagnostics = True
            axes[1, 1].plot(
                rounds,
                [
                    row["aggregation_diagnostics"]["conflict_client_fraction"]
                    for row in history
                ],
                marker="o",
                label=variant,
            )
        elif evaluation_rows and evaluation_rows[0].get("test_accuracy") is not None:
            axes[1, 1].plot(
                [row["round"] for row in evaluation_rows],
                [100.0 * row["test_accuracy"] for row in evaluation_rows],
                marker="o",
                label=f"{variant}: decoded accuracy",
            )
    axes[0, 0].set_title("Retained-client repair objective")
    axes[0, 0].set_xlabel("Repair round")
    axes[0, 0].set_ylabel("Mean local objective")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()
    axes[0, 1].set_title("Beta-weighted VIB KL")
    axes[0, 1].set_xlabel("Repair round")
    axes[0, 1].set_ylabel("Weighted KL objective")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()
    axes[1, 0].set_title("Held-out reconstruction quality")
    axes[1, 0].set_xlabel("Repair round")
    axes[1, 0].set_ylabel("PSNR (dB), paired AWGN 20 dB")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()
    axes[1, 1].set_title(
        "Conflicting retained clients"
        if has_conflict_diagnostics
        else "Held-out decoded accuracy"
    )
    axes[1, 1].set_xlabel("Repair round")
    axes[1, 1].set_ylabel("Fraction" if has_conflict_diagnostics else "Accuracy (%)")
    if has_conflict_diagnostics:
        axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fscuv4_retained_repair.png", dpi=180)
    plt.close(figure)


def save_unlearning_headline_figure(rows, output_dir, method_name):
    """Save the registered utility, forgetting, privacy, and runtime metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("headline metric rows are empty")
    model_names = [row["model"] for row in rows]
    panels = (
        ("clean_rmse_255", "Clean RMSE-255", "0-255 RMSE"),
        ("clean_psnr_db", "Clean PSNR", "dB"),
        ("erased_rmse_255", "Erased RMSE-255", "0-255 RMSE"),
        ("erased_psnr_db", "Erased PSNR", "dB"),
        ("model_accuracy", "Decoded accuracy", "fraction"),
        ("backdoor_test_probe_asr", "ASR", "fraction"),
        ("mia_auc", "MIA ROC-AUC", "AUC"),
        ("backdoor_accuracy", "BA", "fraction"),
        ("running_time_seconds", "Runtime", "seconds"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    positions = np.arange(len(rows))
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        values = [
            float(row[field]) if row.get(field) not in (None, "") else np.nan
            for row in rows
        ]
        axis.bar(positions, values, color="#4c78a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, model_names, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"{method_name} headline comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "unlearning_headline_comparison.png", dpi=180)
    plt.close(figure)


def save_retrain_fid_figure(rows, output_dir):
    """Plot split-level FID from metrics/retrain_fid.csv (retrain outputs as reference)."""
    if not rows:
        raise ValueError("retrain FID rows are empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    model_order = list(dict.fromkeys(row["model"] for row in rows))
    for axis, split in zip(axes, ("clean_test", "erased_train")):
        by_name = {row["model"]: row for row in rows if row["split"] == split}
        if set(by_name) != set(model_order):
            plt.close(figure)
            raise ValueError("FID figure needs both splits with identical model sets")
        split_rows = [by_name[name] for name in model_order]
        positions = np.arange(len(model_order))
        values = [float(row["fid_to_retrain"]) for row in split_rows]
        if not np.isfinite(values).all():
            plt.close(figure)
            raise ValueError("FID plot values must be finite")
        axis.bar(positions, values, color="#4c78a8")
        axis.set_xticks(positions, model_order, rotation=35, ha="right")
        axis.set_ylabel("FID to retrain outputs (lower is closer)")
        first = split_rows[0]
        axis.set_title(f"{split}: N={first['samples']}, paired AWGN {first['evaluation_snr_db']} dB")
        axis.grid(axis="y", alpha=0.25)
    smoke = str(rows[0].get("smoke_test", False)).lower() == "true"
    prefix = "SMOKE ONLY | " if smoke else ""
    figure.suptitle(prefix + "Retrain-reference FID | posterior mean | retrain self-comparison = 0")
    figure.tight_layout()
    figure.savefig(output_dir / "retrain_fid_comparison.png", dpi=180)
    plt.close(figure)


def save_target_information_rate_figure(rows, output_dir):
    """Plot target-input VIB prior-KL for every audited checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("target information-rate rows are empty")

    def order(record):
        name = str(record["model"])
        if name == "original":
            return (0, 0)
        if name.startswith("retrain_without_client_"):
            return (1, 0)
        if name == "client_joint_unlearning":
            return (2, 0)
        marker = "_R"
        if marker in name:
            try:
                return (3, int(name.rsplit(marker, 1)[1]))
            except ValueError:
                return (3, 0)
        if name == "initial":
            return (5, 0)
        return (4, 0)

    ordered = sorted(rows, key=order)
    names = [str(row["model"]) for row in ordered]
    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), squeeze=False)
    axes[0, 0].bar(
        positions,
        [float(row["target_prior_kl_mean"]) for row in ordered],
        color="#4c78a8",
    )
    axes[0, 0].set_title("Target-input VIB prior-KL")
    axes[0, 0].set_ylabel("Mean summed KL per image")
    axes[0, 1].bar(
        positions,
        [
            100.0 * float(row["target_prior_kl_ratio_to_original"])
            for row in ordered
        ],
        color="#f58518",
    )
    axes[0, 1].axhline(1.0, color="tab:red", linestyle="--", label="JU goal: 1%")
    axes[0, 1].set_title("Target information relative to Original")
    axes[0, 1].set_ylabel("Percent of Original")
    axes[0, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(positions, names, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "target_information_rate.png", dpi=180)
    plt.close(figure)

def save_federated_classification_curve_figures(history, output_dir, split_name):
    """Save CE/accuracy curves for the independent raw-image classifier."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_prefix = str(split_name).lower()
    split_label = str(split_name).replace("_", " ").upper()
    training_rows = [
        row
        for row in history
        if int(row["round"]) > 0
        and row.get("train_cross_entropy") is not None
    ]
    performance_rows = [
        row for row in history if row.get("test_cross_entropy") is not None
    ]
    if not training_rows or not performance_rows:
        raise ValueError("classification history is missing CE/accuracy rows")

    train_rounds = np.asarray([row["round"] for row in training_rows])
    test_rounds = np.asarray([row["round"] for row in performance_rows])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
    axes[0, 0].plot(
        train_rounds,
        [row["train_cross_entropy"] for row in training_rows],
        label="train CE",
        linewidth=1.8,
    )
    axes[0, 0].plot(
        test_rounds,
        [row["test_cross_entropy"] for row in performance_rows],
        label="test CE",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )
    axes[0, 0].set_title("Cross-entropy")
    axes[0, 0].set_xlabel("Global round")
    axes[0, 0].set_ylabel("CE")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(
        train_rounds,
        [row["train_accuracy"] for row in training_rows],
        label="train accuracy",
        linewidth=1.8,
    )
    axes[0, 1].plot(
        test_rounds,
        [row["test_accuracy"] for row in performance_rows],
        label="test accuracy",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Global round")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()
    figure.suptitle(f"{split_label} CIFAR-10 classifier-only FedAvg")
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{split_prefix}_classification_training.png", dpi=180
    )
    plt.close(figure)


def save_remaining_classifier_figures(history, client_splits, selected_round, output_dir):
    """Local-validation curves and client class counts, all backed by run metadata."""
    if not history or not client_splits:
        raise ValueError("observer figures need training and split rows")
    valid = [row for row in history if row.get("validation_cross_entropy") is not None]
    if not valid:
        raise ValueError("observer figures need held-out validation rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, train_key, val_key, label in zip(
        axes, ["train_cross_entropy", "train_accuracy"],
        ["validation_cross_entropy", "validation_accuracy"], ["Cross-entropy", "Accuracy (fraction)"],
    ):
        axis.plot([row["round"] for row in history], [row[train_key] for row in history], label="local train (augmented)")
        axis.plot([row["round"] for row in valid], [row[val_key] for row in valid], marker="o", label="local held-out validation (clean)")
        axis.axvline(selected_round, color="green", linestyle="--", label="selected: minimum validation CE")
        axis.axvline(history[-1]["round"], color="grey", linestyle=":", label="final round")
        axis.set(xlabel="Global round", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Remaining-client classifier-only FedAvg | raw CIFAR-10, no channel")
    figure.tight_layout()
    figure.savefig(output_dir / "remaining_classifier_training.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, field, title in zip(axes, ["train_class_counts", "validation_class_counts"], ["Local train", "Local held-out validation"]):
        values = np.asarray([row[field] for row in client_splits])
        drawing = axis.imshow(values, aspect="auto", cmap="Blues")
        axis.set(xlabel="CIFAR-10 class", ylabel="Remaining client ID", title=title)
        axis.set_yticks(range(len(client_splits)), [row["client_id"] for row in client_splits])
        figure.colorbar(drawing, ax=axis, label="Examples")
    figure.tight_layout()
    figure.savefig(output_dir / "remaining_classifier_partition.png", dpi=160)
    plt.close(figure)


def save_output_comparison_figure(rows, output_dir):
    """Image-wise and semantic retrain agreement, clean/erased on separate panels."""
    if not rows:
        raise ValueError("output comparison needs nonempty metric rows")
    fields = [("psnr_to_retrain_db", "PSNR-to-Retrain (dB; cap 120)"),
              ("ssim_to_retrain", "SSIM-to-Retrain"),
              ("class_agreement_to_retrain", "Class agreement (fraction)"),
              ("probability_js_to_retrain", "Probability JS (nats)"),
              ("logit_mse_to_retrain", "Logit MSE")]
    figure, axes = plt.subplots(len(fields), 2, figsize=(14, 15), squeeze=False, sharey="row")
    models = list(dict.fromkeys(row["model"] for row in rows))
    for col, split in enumerate(("clean_test", "erased_train")):
        selected = {row["model"]: row for row in rows if row["split"] == split}
        if set(selected) != set(models):
            raise ValueError("both splits need the same model rows")
        for i, (key, title) in enumerate(fields):
            axis = axes[i, col]
            axis.bar(range(len(models)), [selected[name][key] for name in models])
            axis.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=7)
            axis.set_ylabel(title)
            first = selected[models[0]]
            axis.set_title(f"{split}, n={first['samples']} | paired mean AWGN {first['evaluation_snr_db']:g} dB", fontsize=9)
            axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "retrain_output_semantic_comparison.png", dpi=160)
    plt.close(figure)
