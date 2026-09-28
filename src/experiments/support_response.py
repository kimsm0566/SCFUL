"""여러 Dirichlet 삭제 run의 class support 반응을 집계한다."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from src.data.data import partition_equal_size_dirichlet
from src.evaluation.support_response import (
    SupportGroupingConfig,
    build_class_support_records,
)
from src.utils.artifacts import create_run_artifacts
from src.utils.config import load_json_config
from src.data.data import fscu_split_seed


matplotlib.use("Agg")
from matplotlib import pyplot as plt


INTEGER_COLUMNS = {
    "label",
    "target_train_samples",
    "retained_train_samples",
    "total_train_samples",
    "group_id",
    "samples",
    "test_samples",
}
FLOAT_COLUMNS = {
    "deletion_fraction",
    "remaining_fraction",
    "mse_input",
    "psnr",
    "fg_output",
    "fg_mu",
    "fg_latent",
    "mse_delta_to_retrain",
}


def collect_support_response_runs(runs_root):
    """완료된 support artifact를 가진 run directory만 정렬해 찾는다."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise ValueError(f"runs_root does not exist: {runs_root}")
    if not runs_root.is_dir():
        raise ValueError(f"runs_root must be a directory: {runs_root}")
    return sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path / "metrics" / "support_by_class.csv").is_file()
    )


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {path}") from error


def _read_support_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "label",
            "model",
            "support_group",
            "deletion_fraction",
            "mse_input",
            "fg_output",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"support CSV has required columns missing: {path}")
        rows = []
        for raw_row in reader:
            row = dict(raw_row)
            for name in INTEGER_COLUMNS.intersection(row):
                row[name] = int(row[name])
            for name in FLOAT_COLUMNS.intersection(row):
                row[name] = float(row[name])
            rows.append(row)
    if not rows:
        raise ValueError(f"support CSV contains no records: {path}")
    return rows


def _read_run_rows(run_directory):
    run_directory = Path(run_directory)
    metadata_path = run_directory / "run_metadata.json"
    config_path = run_directory / "config.json"
    support_path = run_directory / "metrics" / "support_by_class.csv"
    if not metadata_path.is_file() or not config_path.is_file() or not support_path.is_file():
        raise ValueError(f"run is missing metadata, config, or support metrics: {run_directory}")
    metadata = _read_json(metadata_path)
    if "global_seed" not in metadata or "split_seed" not in metadata:
        raise ValueError(f"run metadata has seed values missing: {metadata_path}")
    config = _read_json(config_path)
    rows = _read_support_rows(support_path)
    for row in rows:
        row["run_directory"] = str(run_directory)
        row["global_seed"] = int(metadata["global_seed"])
        row["split_seed"] = int(metadata["split_seed"])
        row["experiment_name"] = str(config.get("experiment_name", "unknown"))
    return rows, metadata


def _mean_and_std(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _build_group_summary(rows):
    metric_names = (
        "deletion_fraction",
        "mse_input",
        "psnr",
        "fg_output",
        "fg_mu",
        "fg_latent",
        "mse_delta_to_retrain",
    )
    summary = []
    models = sorted({row["model"] for row in rows})
    groups = ("shared_high", "medium_support", "low_support", "target_specific")
    for model in models:
        for support_group in groups:
            group_rows = [
                row
                for row in rows
                if row["model"] == model and row["support_group"] == support_group
            ]
            if not group_rows:
                continue
            record = {
                "model": model,
                "support_group": support_group,
                "label_count": len(group_rows),
                "run_count": len({row["run_directory"] for row in group_rows}),
            }
            for metric_name in metric_names:
                mean, std = _mean_and_std(group_rows, metric_name)
                record[f"{metric_name}_mean"] = mean
                record[f"{metric_name}_std"] = std
            summary.append(record)
    return summary


def _save_scatter(rows, destination, model_names, title):
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        axis.scatter(
            [row["deletion_fraction"] for row in model_rows],
            [row["fg_output"] for row in model_rows],
            label=model_name,
            alpha=0.8,
        )
    axis.set_xlabel("Client 0 deletion fraction (rho)")
    axis.set_ylabel("Output gap to retrain")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def aggregate_support_response(run_directories, output_root, seed):
    """여러 run의 support CSV를 합쳐 table과 scatter figure를 저장한다."""
    run_directories = [Path(path) for path in run_directories]
    if not run_directories:
        raise ValueError("at least one completed support run is required")

    rows = []
    source_runs = []
    for run_directory in run_directories:
        run_rows, metadata = _read_run_rows(run_directory)
        rows.extend(run_rows)
        source_runs.append(
            {
                "run_directory": str(run_directory),
                "global_seed": metadata["global_seed"],
                "split_seed": metadata["split_seed"],
            }
        )

    artifacts = create_run_artifacts(Path(output_root), seed)
    artifacts.write_json(
        "config.json",
        {
            "analysis": "dirichlet_support_response",
            "source_run_count": len(source_runs),
        },
    )
    artifacts.write_json("source_runs.json", source_runs)
    artifacts.write_csv("metrics/support_response.csv", rows)
    summary = _build_group_summary(rows)
    if summary:
        artifacts.write_csv("metrics/support_group_summary.csv", summary)
    original_rows = [row for row in rows if row["model"] == "original"]
    if not original_rows:
        raise ValueError("source runs do not include an original model")
    _save_scatter(
        original_rows,
        artifacts.path("figures/original_support_response.png"),
        ["original"],
        "Original model Dirichlet support response",
    )
    model_names = sorted(
        {
            row["model"]
            for row in rows
            if row["model"] != "retrain_without_client_0"
        }
    )
    _save_scatter(
        rows,
        artifacts.path("figures/model_support_response.png"),
        model_names,
        "Model Dirichlet support response",
    )
    artifacts.append_log(f"source_runs={len(source_runs)}")
    artifacts.append_log("completed")
    return artifacts.run_dir


def screen_dirichlet_support(config_path, global_seeds):
    """학습 전에 target client의 class support 구간을 계산한다."""
    config = load_json_config(Path(config_path))
    if config["dataset"]["name"] != "CIFAR10":
        raise ValueError("support screening currently requires the CIFAR10 config")
    if config["split"].get("type", "equal_size_dirichlet") != "equal_size_dirichlet":
        raise ValueError("support screening requires an equal_size_dirichlet split")

    class_labels = np.arange(10, dtype=np.int64)
    labels = np.repeat(class_labels, 5000)
    support_config = SupportGroupingConfig(**config.get("support_analysis", {}))
    records = []
    for global_seed in global_seeds:
        split_seed = fscu_split_seed(int(global_seed), config)
        _, allocation = partition_equal_size_dirichlet(
            labels,
            config["federation"]["n_clients"],
            config["split"]["alpha"],
            split_seed,
        )
        seed_records = build_class_support_records(
            class_labels,
            allocation,
            config["unlearning"]["target_client_id"],
            support_config,
        )
        for record in seed_records:
            record["global_seed"] = int(global_seed)
            record["split_seed"] = int(split_seed)
        records.extend(seed_records)
    return records
