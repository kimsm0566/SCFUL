"""Lightweight class-balanced reconstruction contact sheet from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.data import (
    load_cifar10_test_dataset,
    select_cifar10_client_data,
)
from src.experiments.reconstruction_channel_audit import (
    WORKSPACE_ROOT,
    _evaluate_reconstruction_exemplars,
    _load_state,
    _make_codec,
    _reconstruction_visualization_state_names,
    _select_device,
)
from src.utils.artifacts import create_run_artifacts
from src.utils.config import load_json_config
from src.utils.plotting import save_reconstruction_exemplar_figure
from src.utils.reproducibility import (
    collect_environment,
    derive_seed,
    set_global_seed,
)


def run_reconstruction_visualization(
    source_run,
    output_root,
    seed,
    device="auto",
    snr_db=20.0,
):
    """Decode one official CIFAR-10 example per class without full audit work."""

    source_run = Path(source_run).resolve()
    output_root = Path(output_root)
    source_config = load_json_config(source_run / "config.json")
    if source_config["dataset"]["name"] != "CIFAR10":
        raise ValueError("reconstruction visualization supports only CIFAR-10")
    checkpoint_paths = sorted((source_run / "checkpoints").glob("*.pt"))
    states = {path.stem: _load_state(path) for path in checkpoint_paths}
    reference_names = [
        name for name in states if name.startswith("retrain_without_client_")
    ]
    if len(reference_names) != 1 or "original" not in states:
        raise ValueError(
            "visualization source needs original and one retrain-without-client checkpoint"
        )
    visualization_names = _reconstruction_visualization_state_names(
        states, reference_names[0], source_config
    )
    if len(visualization_names) < 3:
        raise ValueError(
            "visualization source needs a primary unlearned checkpoint"
        )

    device = _select_device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(seed, deterministic=True)
    test_dataset = load_cifar10_test_dataset(
        source_config["dataset"], WORKSPACE_ROOT
    )
    test_indices = np.arange(len(test_dataset), dtype=np.int64)
    test_data = select_cifar10_client_data(test_dataset, test_indices)
    exemplar_seed = derive_seed(seed, "clean-reconstruction-exemplars")
    exemplars = _evaluate_reconstruction_exemplars(
        model_factory=lambda: _make_codec(source_config),
        model_states={name: states[name] for name in visualization_names},
        test_data=test_data,
        dataset_indices=test_indices,
        device=device,
        snr_db=float(snr_db),
        seed=exemplar_seed,
    )

    artifacts = create_run_artifacts(output_root, seed)
    artifacts.write_json(
        "config.json",
        {
            "experiment": "reconstruction_visualization",
            "source_run": str(source_run),
            "seed": int(seed),
            "evaluation_snr_db": float(snr_db),
            "source_config": source_config,
        },
    )
    artifacts.write_json(
        "environment.json",
        collect_environment(device, WORKSPACE_ROOT),
    )
    torch.save(
        exemplars,
        artifacts.path("predictions/clean_test_reconstruction_exemplars.pt"),
    )
    artifacts.write_json(
        "metrics/reconstruction_exemplar_protocol.json",
        {
            **exemplars["protocol"],
            "source_run": str(source_run),
            "dataset_indices": exemplars["dataset_indices"].tolist(),
            "labels": exemplars["labels"].tolist(),
            "tensor_artifact": (
                "predictions/clean_test_reconstruction_exemplars.pt"
            ),
            "figure": "figures/clean_test_reconstruction_exemplars.png",
        },
    )
    save_reconstruction_exemplar_figure(
        exemplars["inputs"],
        exemplars["labels"],
        exemplars["outputs"],
        artifacts.path("figures"),
        filename="clean_test_reconstruction_exemplars.png",
        title=(
            f"Clean CIFAR-10 reconstruction exemplars | paired AWGN "
            f"{float(snr_db):g} dB"
        ),
    )
    artifacts.append_log(f"source_run={source_run}")
    artifacts.append_log(f"model_order={json.dumps(visualization_names)}")
    artifacts.append_log("completed")
    return artifacts.run_dir
