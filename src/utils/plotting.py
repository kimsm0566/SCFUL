"""Oracle Behavior artifact용 비대화형 figure 생성기."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.posterior_audit import ClientGaussianSummary


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def reconstruction_comparison_sort_key(model_name):
    """Keep Original, Retrain, then unlearned methods in a stable order."""
    name = str(model_name).lower()
    if name in {"origin", "original"}:
        return 0
    if name == "retrain" or name.startswith("retrain_"):
        return 1
    return 2


def _image_for_plot(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 3:
        return np.moveaxis(array, 0, -1), {}
    return array.squeeze(), {"cmap": "gray", "vmin": 0, "vmax": 1}


def save_reconstruction_exemplar_figure(
    inputs,
    labels,
    outputs,
    output_dir,
    filename="reconstruction_exemplars.png",
    title="Reconstruction exemplars",
):
    """Save one vertical input/output row per represented CIFAR-10 class."""

    if not outputs:
        raise ValueError("reconstruction exemplars require at least one output")
    labels = np.asarray(labels.detach().cpu(), dtype=np.int64)
    if len(inputs) == 0 or len(inputs) != len(labels):
        raise ValueError("reconstruction exemplar inputs and labels must align")
    if len(np.unique(labels)) != len(labels):
        raise ValueError("reconstruction exemplars require one image per class")
    for name, images in outputs.items():
        if len(images) != len(inputs):
            raise ValueError(f"reconstruction exemplar column {name} is misaligned")

    columns = [("Input", inputs), *list(outputs.items())]
    figure, axes = plt.subplots(
        len(inputs),
        len(columns),
        figsize=(1.65 * len(columns), 1.42 * len(inputs)),
        squeeze=False,
    )
    for row, label in enumerate(labels):
        for column, (name, images) in enumerate(columns):
            axis = axes[row, column]
            image, options = _image_for_plot(images[row])
            axis.imshow(
                np.clip(image, 0.0, 1.0), interpolation="nearest", **options
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(str(name).replace("_", " "), fontsize=9)
            if column == 0:
                class_name = (
                    CIFAR10_CLASS_NAMES[label]
                    if 0 <= label < len(CIFAR10_CLASS_NAMES)
                    else "unknown"
                )
                axis.set_ylabel(
                    f"{label} {class_name}", rotation=0, ha="right", va="center",
                    fontsize=8, labelpad=6,
                )
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


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

    if joint_metrics.get("forgetting_objective") == "module_wise_budgeted":
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        stages = [row["stage"] for row in joint_history]
        colors = [
            "tab:blue" if stage == "encoder_information_rate" else "tab:orange"
            for stage in stages
        ]
        axes[0, 0].plot(
            epochs,
            [100.0 * row["target_information_ratio"] for row in joint_history],
            marker="o",
        )
        axes[0, 0].axhline(
            100.0 * float(joint_metrics["target_information_ratio_threshold"]),
            color="tab:red",
            linestyle="--",
            label="encoder stop threshold",
        )
        axes[0, 0].set_title("Encoder information-rate budget")
        axes[0, 0].set_ylabel("Target prior-KL (% of Original)")
        axes[0, 0].legend()

        axes[0, 1].plot(
            epochs,
            [row["decoder_probe_psnr_db"] for row in joint_history],
            marker="o",
        )
        axes[0, 1].axhline(
            float(joint_metrics["decoder_stage"]["psnr_threshold_db"]),
            color="tab:red",
            linestyle="--",
            label="decoder stop threshold",
        )
        axes[0, 1].set_title("Decoder distortion budget")
        axes[0, 1].set_ylabel("Original-encoder probe PSNR (dB)")
        axes[0, 1].legend()

        axes[1, 0].plot(
            epochs,
            [row["target_psnr_db"] for row in joint_history],
            marker="o",
        )
        axes[1, 0].set_title("End-to-end target reconstruction")
        axes[1, 0].set_ylabel("PSNR (dB)")

        axes[1, 1].scatter(
            epochs,
            [row["total_loss"] for row in joint_history],
            c=colors,
        )
        axes[1, 1].plot(
            epochs,
            [row["total_loss"] for row in joint_history],
            color="0.65",
            linewidth=1.0,
        )
        axes[1, 1].set_title("Stage objective")
        axes[1, 1].set_ylabel("Optimization objective")
        for axis in axes.flat:
            axis.set_xlabel("Sequential local epoch")
            axis.grid(alpha=0.25)
        figure.suptitle(
            "Module-wise target-client unlearning: encoder then decoder"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / "fscuv4_module_wise_unlearning.png", dpi=180
        )
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

    regularizer_fields = (
        (
            "posterior_mu_kd_loss",
            "posterior_mu_kd_weight",
            "Posterior mean KD",
        ),
        (
            "posterior_logvar_kd_loss",
            "posterior_logvar_kd_weight",
            "Posterior log-variance KD",
        ),
        (
            "marker_outside_consistency_loss",
            "marker_outside_consistency_weight",
            "Outside-marker consistency",
        ),
    )
    has_regularizers = any(
        any(float(row.get(weight_field, 0.0)) > 0.0 for row in metrics.get("history", []))
        for metrics in variant_metrics.values()
        for _loss_field, weight_field, _title in regularizer_fields
    )
    if has_regularizers:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
        for variant, metrics in variant_metrics.items():
            history = metrics.get("history", [])
            rounds = [row["round"] for row in history]
            for column, (loss_field, weight_field, title) in enumerate(
                regularizer_fields
            ):
                axis = axes[0, column]
                axis.plot(
                    rounds,
                    [row.get(loss_field, 0.0) for row in history],
                    label=(
                        f"{variant}, weight="
                        f"{history[0].get(weight_field, 0.0):g}"
                        if history
                        else variant
                    ),
                )
                axis.set_title(title)
                axis.set_xlabel("Repair round")
                axis.set_ylabel("Mean local loss")
                axis.grid(alpha=0.25)
                axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / "fscuv4_repair_regularizers.png", dpi=180
        )
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


def save_encoder_retrain_gap_figure(rows, output_dir):
    """Plot clean/erased posterior gaps to the Retrain encoder."""

    if not rows:
        raise ValueError("encoder Retrain-gap rows are empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [row["model"] for row in rows]
    panels = (
        (
            "clean_latent_mean_mse_to_retrain",
            "Clean posterior-mean gap",
            "Mean MSE to Retrain",
        ),
        (
            "erased_latent_mean_mse_to_retrain",
            "Erased posterior-mean gap",
            "Mean MSE to Retrain",
        ),
        (
            "clean_posterior_symmetric_kl_to_retrain",
            "Clean posterior distribution gap",
            "Symmetric KL to Retrain",
        ),
        (
            "erased_posterior_symmetric_kl_to_retrain",
            "Erased posterior distribution gap",
            "Symmetric KL to Retrain",
        ),
    )
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        axis.bar(positions, [float(row[field]) for row in rows], color="#4c78a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, model_names, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Encoder equivalence to Retrain | lower is closer")
    figure.tight_layout()
    figure.savefig(output_dir / "encoder_retrain_gap_comparison.png", dpi=180)
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


def save_reconstruction_marker_figures(rows, previews, marker_config, output_dir):
    """Save marker-score panels and reconstruction/crop previews."""

    if not rows:
        raise ValueError("reconstruction marker figures need nonempty rows")
    rows = sorted(rows, key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = previews.get("evaluation_scenario", {})
    scenario_title = scenario.get(
        "label_en", rows[0].get("evaluation_scenario_id", rows[0]["protocol"])
    )
    models = [row["model"] for row in rows]
    positions = np.arange(len(models))
    panels = [
        ("marker_projection_score", "Marker projection", "ideal-marker units"),
        ("marker_correlation", "Marker correlation", "cosine"),
        (
            "marker_region_mse_to_desired_response",
            "Marker-region error",
            "MSE",
        ),
        ("outside_watermark_rmse_255", "Outside-marker utility", "RMSE-255"),
    ]
    if rows[0].get("output_probe_sensitivity") is not None:
        panels.extend(
            [
                (
                    "output_probe_sensitivity",
                    "Output probe sensitivity",
                    "output-change / input-change",
                ),
                (
                    "latent_probe_sensitivity",
                    "Latent probe sensitivity",
                    "latent-change / input-change",
                ),
            ]
        )
    figure, axes = plt.subplots(
        1, len(panels), figsize=(4 * len(panels), 4.4), squeeze=False
    )
    for column, (field, title, ylabel) in enumerate(panels):
        axis = axes[0, column]
        values = [row[field] for row in rows]
        errors = None
        if field in (
            "marker_projection_score",
            "marker_correlation",
            "output_probe_sensitivity",
            "latent_probe_sensitivity",
        ):
            errors = [row[f"{field}_standard_error"] for row in rows]
        axis.bar(positions, values, yerr=errors, capsize=4, color=["#4c78a8", "#f58518", "#54a24b"][:len(models)])
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"{scenario_title} | paired AWGN {rows[0]['evaluation_snr_db']:g} dB"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction_marker_comparison.png", dpi=180)
    plt.close(figure)

    utility_panels = [
        ("clean_rmse_255", "Clean-input reconstruction", "RMSE-255"),
        ("probe_input_rmse_255", "Probe-input reconstruction", "RMSE-255"),
        ("clean_psnr_db", "Clean-input reconstruction", "PSNR (dB)"),
        ("probe_input_psnr_db", "Probe-input reconstruction", "PSNR (dB)"),
    ]
    figure, axes = plt.subplots(
        1, len(utility_panels), figsize=(16, 4.4), squeeze=False
    )
    for column, (field, title, ylabel) in enumerate(utility_panels):
        axis = axes[0, column]
        axis.bar(
            positions,
            [row[field] for row in rows],
            color=["#4c78a8", "#f58518", "#54a24b"][: len(models)],
        )
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Clean versus probe reconstruction | {scenario_title} | paired AWGN {rows[0]['evaluation_snr_db']:g} dB"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "reconstruction_marker_clean_probe_utility.png", dpi=180
    )
    plt.close(figure)

    model_outputs = previews["model_outputs"]
    row_names = ["clean input", "probe input", "desired target", *models]
    row_images = [
        previews["clean_inputs"],
        previews["probe_inputs"],
        previews["desired_targets"],
        *[model_outputs[name] for name in models],
    ]
    sample_count = min(8, len(previews["clean_inputs"]))
    figure, axes = plt.subplots(
        len(row_names), sample_count, figsize=(1.8 * sample_count, 1.7 * len(row_names)), squeeze=False
    )
    for row_index, (name, images) in enumerate(zip(row_names, row_images)):
        for column in range(sample_count):
            axis = axes[row_index, column]
            image, options = _image_for_plot(images[column])
            axis.imshow(np.clip(image, 0.0, 1.0), **options)
            axis.axis("off")
            if column == 0:
                axis.set_title(name, fontsize=9, loc="left")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction_marker_grid.png", dpi=180)
    plt.close(figure)

    clean_model_outputs = previews["clean_model_outputs"]
    comparison_names = ["clean input"]
    comparison_images = [previews["clean_inputs"]]
    for name in models:
        comparison_names.append(f"{name} on clean")
        comparison_images.append(clean_model_outputs[name])
    protocol = str(rows[0]["protocol"])
    probe_label = (
        "masked probe input"
        if protocol == "masked_input_marker_reconstruction_v1"
        else "probe input"
    )
    comparison_names.append(probe_label)
    comparison_images.append(previews["probe_inputs"])
    comparison_names.append("marked target")
    comparison_images.append(previews["desired_targets"])
    for name in models:
        comparison_names.append(f"{name} on probe")
        comparison_images.append(model_outputs[name])
    figure, axes = plt.subplots(
        len(comparison_names),
        sample_count,
        figsize=(1.8 * sample_count, 1.7 * len(comparison_names)),
        squeeze=False,
    )
    for row_index, (name, images) in enumerate(
        zip(comparison_names, comparison_images)
    ):
        for column in range(sample_count):
            axis = axes[row_index, column]
            image, options = _image_for_plot(images[column])
            axis.imshow(np.clip(image, 0.0, 1.0), **options)
            axis.axis("off")
            if column == 0:
                axis.set_title(name, fontsize=9, loc="left")
    figure.tight_layout()
    figure.savefig(
        output_dir / "reconstruction_marker_clean_probe_grid.png", dpi=180
    )
    plt.close(figure)

    region = marker_config["output_watermark"]
    top = int(region["top"])
    left = int(region["left"])
    height = int(region["height"])
    width = int(region["width"])
    zoom_names = ["desired target", *models]
    zoom_images = [
        previews["desired_targets"],
        *[model_outputs[name] for name in models],
    ]
    figure, axes = plt.subplots(
        len(zoom_names), sample_count, figsize=(1.7 * sample_count, 1.55 * len(zoom_names)), squeeze=False
    )
    for row_index, (name, images) in enumerate(zip(zoom_names, zoom_images)):
        crops = images[:, :, top : top + height, left : left + width]
        for column in range(sample_count):
            axis = axes[row_index, column]
            image, options = _image_for_plot(crops[column])
            axis.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest", **options)
            axis.axis("off")
            if column == 0:
                axis.set_title(name, fontsize=9, loc="left")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction_marker_zoom.png", dpi=220)
    plt.close(figure)

    save_reconstruction_exemplar_figure(
        previews["clean_inputs"],
        previews["labels"],
        {
            "Probe input": previews["probe_inputs"],
            "Desired marker": previews["desired_targets"],
            **{name: model_outputs[name] for name in models},
        },
        output_dir,
        filename="reconstruction_marker_exemplars.png",
        title=f"{scenario_title}: class-balanced reconstruction exemplars",
    )


def save_client_partition_figure(
    split_manifest, output_dir, target_client_id=None
):
    """Visualize per-client CIFAR-10 counts recorded in the split manifest."""

    clients = split_manifest.get("clients", [])
    class_labels = split_manifest.get("class_labels", [])
    if not clients or not class_labels:
        raise ValueError("client partition figure requires a complete split manifest")
    counts = np.asarray(
        [client["train_label_count"] for client in clients], dtype=np.int64
    )
    if counts.shape != (len(clients), len(class_labels)):
        raise ValueError("client partition counts do not match manifest labels")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.2))
    drawing = axis.imshow(counts, aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(class_labels)), class_labels)
    axis.set_yticks(
        range(len(clients)), [client["client_id"] for client in clients]
    )
    axis.set_xlabel("CIFAR-10 class")
    axis.set_ylabel("Client ID")
    axis.set_title(
        f"Client training allocation | {split_manifest['partition']} split"
    )
    if target_client_id is not None:
        client_ids = [int(client["client_id"]) for client in clients]
        if int(target_client_id) not in client_ids:
            plt.close(figure)
            raise ValueError("target client is absent from the split manifest")
        target_row = client_ids.index(int(target_client_id))
        axis.axhline(target_row - 0.5, color="tab:red", linewidth=2.2)
        axis.axhline(target_row + 0.5, color="tab:red", linewidth=2.2)
        axis.text(
            len(class_labels) - 0.1,
            target_row,
            " target",
            color="tab:red",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
        )
    figure.colorbar(drawing, ax=axis, label="Training examples")
    figure.tight_layout()
    figure.savefig(output_dir / "client_class_distribution.png", dpi=180)
    plt.close(figure)


def save_target_client_class_distribution_figure(
    split_manifest, target_client_id, output_dir
):
    """Save the target client's exact class counts and return CSV-ready rows."""

    clients = split_manifest.get("clients", [])
    class_labels = [int(value) for value in split_manifest.get("class_labels", [])]
    target = next(
        (
            client
            for client in clients
            if int(client["client_id"]) == int(target_client_id)
        ),
        None,
    )
    if target is None or not class_labels:
        raise ValueError("target-client distribution requires a complete manifest")
    counts = np.asarray(target["train_label_count"], dtype=np.int64)
    if counts.shape != (len(class_labels),):
        raise ValueError("target-client counts do not match manifest labels")
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("target client has no training examples")

    ranked_positions = sorted(
        range(len(class_labels)), key=lambda index: (-int(counts[index]), index)
    )
    rank_by_position = {
        position: rank for rank, position in enumerate(ranked_positions, start=1)
    }
    rows = []
    for position, (class_id, count) in enumerate(zip(class_labels, counts)):
        class_name = (
            CIFAR10_CLASS_NAMES[class_id]
            if 0 <= class_id < len(CIFAR10_CLASS_NAMES)
            else f"class_{class_id}"
        )
        rows.append(
            {
                "partition": str(split_manifest["partition"]),
                "target_client_id": int(target_client_id),
                "class_id": int(class_id),
                "class_name": class_name,
                "train_examples": int(count),
                "train_fraction": float(count) / float(total),
                "rank_by_count": int(rank_by_position[position]),
            }
        )

    colors = ["#9ecae1"] * len(class_labels)
    highlight_colors = ("#d62728", "#ff7f0e", "#f2c14e")
    for rank, position in enumerate(ranked_positions[:3]):
        colors[position] = highlight_colors[rank]
    positions = np.arange(len(class_labels))
    figure, axis = plt.subplots(figsize=(11, 5.4))
    bars = axis.bar(positions, counts, color=colors)
    labels = [
        (
            f"{class_id}\n{CIFAR10_CLASS_NAMES[class_id]}"
            if 0 <= class_id < len(CIFAR10_CLASS_NAMES)
            else str(class_id)
        )
        for class_id in class_labels
    ]
    axis.set_xticks(positions, labels)
    axis.set_xlabel("CIFAR-10 class")
    axis.set_ylabel("Target-client training examples")
    axis.set_title(
        f"Target client {int(target_client_id)} class distribution | "
        f"{split_manifest['partition']} | N={total}"
    )
    axis.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        if int(count) <= 0:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(count)}\n({100.0 * float(count) / float(total):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "target_client_class_distribution.png", dpi=180
    )
    plt.close(figure)
    return rows


def save_encoder_idi_figures(layer_rows, idi_rows, output_dir):
    """Plot layer-wise InfoNCE MI and Retrain-normalized IDI."""

    if not layer_rows or not idi_rows:
        raise ValueError("encoder IDI figures need nonempty metric rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(row["model"] for row in layer_rows))
    layers = list(dict.fromkeys(row["layer"] for row in layer_rows))
    positions = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(11, 5.6))
    for model in models:
        selected = {
            row["layer"]: row for row in layer_rows if row["model"] == model
        }
        if set(selected) != set(layers):
            raise ValueError("latent leakage models must share the same layers")
        values = [float(selected[layer]["mi_lower_bound_nats"]) for layer in layers]
        errors = [float(selected[layer]["mi_standard_error"]) for layer in layers]
        axis.errorbar(
            positions,
            values,
            yerr=errors,
            marker="o",
            capsize=3,
            label=model,
        )
    axis.set_xticks(positions, [layer.replace("_", "\n") for layer in layers])
    axis.set_ylabel("InfoNCE MI lower bound (nats)")
    axis.set_xlabel("Encoder representation")
    axis.set_title("Class-matched encoder information difference")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "encoder_idi_layerwise.png", dpi=180)
    plt.close(figure)

    valid_rows = [row for row in idi_rows if row.get("idi") not in (None, "")]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if valid_rows:
        axis.bar(
            np.arange(len(valid_rows)),
            [float(row["idi"]) for row in valid_rows],
            color="#4c78a8",
        )
        axis.set_xticks(
            np.arange(len(valid_rows)),
            [row["model"] for row in valid_rows],
            rotation=18,
            ha="right",
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.axhline(1.0, color="#d62728", linestyle="--", linewidth=1)
    else:
        axis.text(
            0.5,
            0.5,
            "IDI undefined: Original-Retrain MI denominator is not positive",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
    axis.set_ylabel("IDI (Retrain=0, Original=1)")
    axis.set_title("Retrain-normalized encoder information difference")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "encoder_idi_summary.png", dpi=180)
    plt.close(figure)


def save_selected_core_metrics_figure(rows, output_dir):
    """Plot MPR/PUL/FID/PSNR/SSIM/runtime; dFIL has its own diagnostic figure."""

    if not rows:
        raise ValueError("selected core metric figure needs nonempty rows")
    rows = sorted(rows, key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in rows]
    positions = np.arange(len(models))
    width = 0.36
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), squeeze=False)

    pul_values = [
        np.nan
        if row.get("output_pul_percent") in (None, "")
        else float(row["output_pul_percent"])
        for row in rows
    ]
    marker_rates = [
        np.nan
        if row.get("output_marker_positive_percent") in (None, "")
        else float(row["output_marker_positive_percent"])
        for row in rows
    ]
    axes[0, 0].bar(positions, marker_rates, color="#b279a2")
    axes[0, 0].set_ylabel("Marker-positive rate (%) | compare to Retrain")
    axes[0, 0].set_title("Absolute marker detection (fixed-region detector)")
    axes[0, 1].bar(positions, pul_values, color="#e15759")
    axes[0, 1].set_ylabel("PUL (%) | higher is better")
    axes[0, 1].set_title("Marker-positive output removal")

    axes[0, 2].bar(
        positions,
        [float(row["retain_fid_to_real"]) for row in rows],
        color="#f28e2b",
    )
    axes[0, 2].set_ylabel("FID | lower is better")
    axes[0, 2].set_title("Retain reconstructions to real retain images")

    axes[1, 0].bar(
        positions - width / 2,
        [float(row["clean_psnr_db"]) for row in rows],
        width,
        label="clean test",
        color="#4e79a7",
    )
    axes[1, 0].bar(
        positions + width / 2,
        [float(row["erased_psnr_db"]) for row in rows],
        width,
        label="erased train",
        color="#76b7b2",
    )
    axes[1, 0].set_ylabel("PSNR (dB) | compare to Retrain on both splits")
    axes[1, 0].set_title("Reconstruction fidelity")
    axes[1, 0].legend()

    axes[1, 1].bar(
        positions - width / 2,
        [float(row["clean_ssim"]) for row in rows],
        width,
        label="clean test",
        color="#59a14f",
    )
    axes[1, 1].bar(
        positions + width / 2,
        [float(row["erased_ssim"]) for row in rows],
        width,
        label="erased train",
        color="#8cd17d",
    )
    axes[1, 1].set_ylabel("SSIM | compare to Retrain on both splits")
    axes[1, 1].set_title("Structural reconstruction fidelity")
    axes[1, 1].legend()
    axes[1, 2].bar(
        positions,
        [np.nan if row["running_time_seconds"] is None
         else float(row["running_time_seconds"]) for row in rows],
        color="#9c755f",
    )
    axes[1, 2].set_ylabel("Algorithm wall time (seconds)")
    axes[1, 2].set_title("Offline metric evaluation excluded")
    for axis in axes.flat:
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Federated reconstruction unlearning | selected core metrics")
    figure.tight_layout()
    figure.savefig(output_dir / "selected_core_metrics.png", dpi=180)
    plt.close(figure)


def save_encoder_dfil_figure(rows, output_dir):
    """Plot split-wise dFIL backed by encoder_dfil.csv (no batch FID analogy)."""
    if not rows:
        raise ValueError("dFIL figure requires nonempty metric rows")
    rows = sorted(rows, key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = list(dict.fromkeys(row["split"] for row in rows))
    figure, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False)
    for axis, split in zip(axes.flat, splits):
        selected = [row for row in rows if row["split"] == split]
        axis.bar(
            [row["model"] for row in selected], [row["dfil"] for row in selected],
            yerr=[row["mean_mc_standard_error"] for row in selected],
            capsize=3,
        )
        axis.set_title(f"{split} | n={selected[0]['samples']}")
        axis.set_ylabel("dFIL (inverse squared [0,1] pixel units)")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Pre-channel Gaussian posterior dFIL | error bars: trace Monte Carlo SE")
    figure.tight_layout()
    figure.savefig(output_dir / "encoder_dfil.png", dpi=180)
    plt.close(figure)


def save_retain_ufid_figure(rows, output_dir):
    """Plot reconstruction FID to real clean remaining-client images."""

    if not rows:
        raise ValueError("retain u-FID figure needs nonempty metric rows")
    rows = sorted(rows, key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(
        positions,
        [float(row["retain_ufid"]) for row in rows],
        color="#f28e2b",
    )
    axis.set_xticks(
        positions,
        [row["model"] for row in rows],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("retain u-FID (lower is better)")
    axis.set_title(
        "Retain reconstruction distribution vs real retain inputs | "
        f"n={int(rows[0]['samples'])}"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "retain_ufid_comparison.png", dpi=180)
    plt.close(figure)


def save_marker_spread_figures(result, marker_config, output_dir):
    """Plot CSV-backed global summaries/maps with shared scales across codecs."""
    rows = sorted(result["summary"], key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    if not rows or not result["map_rows"] or not result["per_sample"]:
        raise ValueError("marker spread figures require nonempty summary, map and per-input rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in rows]
    panels = [
        ("global_peak_amplitude_mean", "Whole-image peak amplitude", "ideal-template units"),
        ("outside_peak_amplitude_mean", "Peak outside original marker", "ideal-template units"),
        ("outside_amplitude_rms_mean", "Outside response RMS", "ideal-template units"),
        ("global_peak_ncc_mean", "Whole-image peak |NCC|", "normalized correlation"),
        ("outside_peak_ncc_mean", "Outside peak |NCC|", "normalized correlation"),
        ("global_mpr_percent", "Global marker-positive rate", "% of held-out images"),
        ("coverage_percent_mean", "Whole-image marker coverage", "% of pixels (union)"),
        ("outside_coverage_percent_mean", "Outside marker coverage", "% of outside pixels (union)"),
        ("outside_color_bias_mean", "Outside color residual (nonspecific)", "signed RGB residual projection"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    for axis, (field, title, unit) in zip(axes.flat, panels):
        error_key = field.replace("_mean", "_standard_error")
        errors = None if field == "global_mpr_percent" else [row[error_key] for row in rows]
        axis.bar(models, [row[field] for row in rows], yerr=errors, capsize=3)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    protocol = result["protocol"]
    figure.suptitle(f"Marker spread | n={rows[0]['samples']} | compare to Retrain | offline evaluation")
    figure.tight_layout()
    figure.savefig(output_dir / "marker_spread_comparison.png", dpi=180)
    plt.close(figure)

    maps = result["maps"]["models"]
    amplitude_max = max(float(maps[name]["mean_amplitude"].max()) for name in models) or 1.0
    palette = np.asarray(marker_config["output_watermark"]["first_rgb"]) - np.asarray(marker_config["output_watermark"]["second_rgb"])
    palette = palette / max(float(np.linalg.norm(palette)), 1e-12)
    residuals = {name: (maps[name]["mean_residual"].numpy() * palette[:, None, None]).sum(axis=0) for name in models}
    residual_max = max(float(np.abs(value).max()) for value in residuals.values()) or 1.0
    figure, axes = plt.subplots(len(models), 4, figsize=(13, 3.5 * len(models)), squeeze=False)
    region = marker_config["output_watermark"]
    from matplotlib.patches import Rectangle
    for row_index, name in enumerate(models):
        values = [maps[name]["mean_amplitude"], maps[name]["mean_ncc"], maps[name]["coverage_frequency"], residuals[name]]
        limits = [(0, amplitude_max), (0, 1), (0, 1), (-residual_max, residual_max)]
        titles = ["Mean |projection| at window centers", "Mean max |NCC|", "Pixel detection frequency", "Mean color residual (nonspecific)"]
        for column, (value, (vmin, vmax), title) in enumerate(zip(values, limits, titles)):
            axis = axes[row_index, column]
            picture = axis.imshow(value, vmin=vmin, vmax=vmax, cmap="coolwarm" if column == 3 else "magma")
            axis.add_patch(Rectangle((int(region["left"]) - .5, int(region["top"]) - .5), int(region["width"]), int(region["height"]), fill=False, edgecolor="cyan", linewidth=1))
            axis.set_title(f"{name}\n{title}")
            axis.axis("off")
            figure.colorbar(picture, ax=axis, fraction=.046)
    figure.suptitle("Held-out mean response maps | shared scales | cyan: original marker region")
    figure.tight_layout()
    figure.savefig(output_dir / "marker_spread_maps.png", dpi=180)
    plt.close(figure)

    example_max = max(float(maps[name]["example_amplitude"].max()) for name in models) or 1.0
    figure, axes = plt.subplots(len(models), 4, figsize=(12, 3.2 * len(models)), squeeze=False)
    for row_index, name in enumerate(models):
        item = maps[name]
        axes[row_index, 0].imshow(item["example_input"].permute(1, 2, 0).numpy().clip(0, 1))
        axes[row_index, 1].imshow(item["example_output"].permute(1, 2, 0).numpy().clip(0, 1))
        axes[row_index, 2].imshow(item["example_amplitude"], vmin=0, vmax=example_max, cmap="magma")
        axes[row_index, 3].imshow(item["example_coverage"], vmin=0, vmax=1, cmap="magma")
        for axis, title in zip(axes[row_index], ("Input", name, "Global |projection|", "Detected pixel union")):
            axis.set_title(title)
            axis.axis("off")
    figure.suptitle(f"First held-out input (not effect-selected) | dataset index {result['maps']['dataset_indices'][0]}")
    figure.tight_layout()
    figure.savefig(output_dir / "marker_spread_examples.png", dpi=180)
    plt.close(figure)


def save_marker_response_figure(rows, output_dir):
    """Plot raw amplitude/NCC with standard errors, without reference rescaling."""
    if not rows:
        raise ValueError("marker response figure needs nonempty metric rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scopes = list(dict.fromkeys(row["scope"] for row in rows))
    amplitudes = [row for row in rows if row["metric"] == "amplitude"]
    lower = min([0.] + [float(r["mean"]) - float(r["standard_error"]) for r in amplitudes])
    upper = max([0.] + [float(r["mean"]) + float(r["standard_error"]) for r in amplitudes])
    padding = max((upper - lower) * .08, .0001)
    limits = {"amplitude": (lower - padding, upper + padding), "ncc": (-1.05, 1.05)}
    figure, axes = plt.subplots(len(scopes), 2, figsize=(12, 3.8 * len(scopes)), squeeze=False)
    for row_index, scope in enumerate(scopes):
        for column, metric in enumerate(("amplitude", "ncc")):
            axis = axes[row_index, column]
            selected = sorted([r for r in rows if r["scope"] == scope and r["metric"] == metric],
                              key=lambda r: reconstruction_comparison_sort_key(r["model"]))
            models = [r["model"] for r in selected]
            axis.set_xticks(range(len(models)), models, rotation=15)
            axis.bar(range(len(models)), [r["mean"] for r in selected],
                     yerr=[r["standard_error"] for r in selected], capsize=3,
                     color=["#4c78a8", "#f58518", "#54a24b"][:len(models)])
            axis.set_title(f"{scope}: {metric} | n={selected[0]['samples']}")
            axis.set_ylabel("raw projection coefficient" if metric == "amplitude" else "normalized correlation")
            axis.set_ylim(*limits[metric])
            axis.axhline(0, color="gray", linewidth=.8)
            axis.grid(axis="y", alpha=.2)
    figure.suptitle("Unscaled amplitude/NCC means | same held-out inputs | error bars: standard error")
    figure.tight_layout()
    figure.savefig(output_dir / "marker_response_comparison.png", dpi=180)
    plt.close(figure)


def save_marker_pul_figure(rows, output_dir):
    """Plot held-out marker occurrence and PUL for a learned-marker protocol."""

    if not rows:
        raise ValueError("Marker-PUL figure needs nonempty metric rows")
    rows = sorted(rows, key=lambda row: reconstruction_comparison_sort_key(row["model"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(rows))
    models = [row["model"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), squeeze=False)
    axes[0, 0].bar(
        positions,
        [100.0 * float(row["marker_positive_rate"]) for row in rows],
        color="#b279a2",
    )
    axes[0, 0].set_ylabel("Marker-positive outputs (%)")
    axes[0, 0].set_title("Held-out marker occurrence")
    pul_values = [
        np.nan if row.get("marker_pul_percent") in (None, "") else float(row["marker_pul_percent"])
        for row in rows
    ]
    axes[0, 1].bar(positions, pul_values, color="#e15759")
    axes[0, 1].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Marker-PUL (%)")
    axes[0, 1].set_title("Reduction relative to Original")
    for axis in axes[0]:
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    protocol = str(rows[0].get("protocol", "learned-marker protocol"))
    figure.suptitle(f"Matched-filter Marker-PUL | {protocol}")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstruction_marker_pul.png", dpi=180)
    plt.close(figure)
