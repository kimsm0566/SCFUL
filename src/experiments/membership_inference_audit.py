"""완료된 federated deletion run의 reconstruction MIA를 독립 artifact로 기록한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data.data import (
    dataset_labels,
    load_cifar10_test_dataset,
    load_cifar10_train_dataset,
    select_cifar10_client_data,
)
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
from src.utils.artifacts import create_run_artifacts
from src.utils.config import load_json_config
from src.utils.reproducibility import collect_environment, derive_seed, set_global_seed


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


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


def _source_identity(source_run):
    return str(source_run.resolve())


def _audit_one_source(source_run, device, config, seed):
    source_run = Path(source_run)
    source_config = load_json_config(source_run / "config.json")
    manifest = json.loads((source_run / "split_manifest.json").read_text(encoding="utf-8"))
    if source_config["dataset"]["name"] != "CIFAR10":
        raise ValueError("membership inference audit currently supports CIFAR-10 runs")

    target_client_id = int(manifest["target_client_id"] if "target_client_id" in manifest else source_config["unlearning"]["target_client_id"])
    target_train_indices = np.asarray(
        manifest["clients"][target_client_id]["train_indices"], dtype=np.int64
    )
    split_seed = int(manifest["seed"])
    train_dataset = load_cifar10_train_dataset(
        source_config["dataset"], WORKSPACE_ROOT
    )
    test_dataset = load_cifar10_test_dataset(
        source_config["dataset"], WORKSPACE_ROOT
    )
    member_data = select_cifar10_client_data(train_dataset, target_train_indices)
    nonmember_data = select_cifar10_client_data(
        test_dataset, np.arange(len(test_dataset), dtype=np.int64)
    )
    checkpoint_paths = sorted((source_run / "checkpoints").glob("*.pt"))
    if not checkpoint_paths:
        raise ValueError(f"MIA source run has no checkpoints: {source_run}")
    states = {path.stem: _load_state(path) for path in checkpoint_paths}
    records, per_class, metadata = evaluate_reconstruction_membership_inference(
        model_factory=lambda: _make_codec(source_config),
        model_states=states,
        member_data=member_data,
        nonmember_data=nonmember_data,
        config=config,
        seed=derive_seed(seed, f"mia:{_source_identity(source_run)}"),
    )
    source_key = source_run.parent.name
    for record in records:
        record["source_key"] = source_key
        record["source_run"] = _source_identity(source_run)
    for record in per_class:
        record["source_key"] = source_key
        record["source_run"] = _source_identity(source_run)
    return records, per_class, {
        "source_key": source_key,
        "source_run": _source_identity(source_run),
        "target_client_id": target_client_id,
        "target_train_examples": int(len(target_train_indices)),
        "target_train_index_sha256": hashlib.sha256(
            target_train_indices.tobytes()
        ).hexdigest(),
        "test_nonmember_pool_examples": int(len(test_dataset)),
        "member_label_count": {
            str(label): int(count)
            for label, count in zip(*np.unique(dataset_labels(train_dataset)[target_train_indices], return_counts=True))
        },
        **metadata,
    }


def run_membership_inference_audit(source_runs, output_root, seed, device="auto"):
    """여러 완료 run을 동일한 class-matched loss MIA로 비교한다."""
    if not source_runs:
        raise ValueError("membership_inference_audit needs at least one --source-run")
    device = _select_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(seed, deterministic=True)
    artifacts = create_run_artifacts(Path(output_root), seed)
    audit_config = ReconstructionMIAConfig(device=str(device))
    all_records = []
    all_per_class = []
    source_metadata = []
    for source_run in source_runs:
        records, per_class, metadata = _audit_one_source(
            Path(source_run), device, audit_config, seed
        )
        all_records.extend(records)
        all_per_class.extend(per_class)
        source_metadata.append(metadata)

    artifacts.write_csv("metrics/reconstruction_mia_summary.csv", all_records)
    artifacts.write_csv("metrics/reconstruction_mia_per_class.csv", all_per_class)
    artifacts.write_json(
        "metrics/reconstruction_mia_metadata.json",
        {
            "audit_config": {
                "batch_size": audit_config.batch_size,
                "max_examples_per_class": audit_config.max_examples_per_class,
                "min_examples_per_class": audit_config.min_examples_per_class,
                "calibration_fraction": audit_config.calibration_fraction,
                "device": str(device),
            },
            "sources": source_metadata,
        },
    )
    artifacts.write_json("environment.json", collect_environment(device, WORKSPACE_ROOT))
    artifacts.append_log(f"seed={seed}")
    artifacts.append_log(f"sources={len(source_runs)}")
    artifacts.append_log("completed")
    return artifacts.run_dir
