"""통제된 non-IID client deletion의 retraining-oracle behavior workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.data import (
    ProbeDataset,
    ProbeRef,
    build_controlled_noniid_split,
    dataset_labels,
    load_cifar10_test_dataset,
    load_cifar10_train_dataset,
    select_class_balanced_subset,
    select_audit_indices,
)
from src.evaluation.posterior_audit import (
    ClientPosteriorAuditConfig,
    audit_client_gaussian_sufficiency,
    client_posterior_summary_payload,
)
from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    Posterior,
    SUPPORTED_DEEPJSCC_ARCHITECTURES,
    VariationalDeepJSCC,
    variational_deepjscc_config_for_architecture,
)
from src.utils.reconstruction import TrainingConfig, train_reconstruction
from src.utils.scu import SCUConfig, adapt_scu_style
from src.utils.artifacts import create_run_artifacts
from src.utils.config import WORKSPACE_ROOT, build_effective_config, select_device
from src.utils.metrics import per_sample_mse, summarize_records, symmetric_diagonal_kl
from src.utils.plotting import save_oracle_figures
from src.utils.reproducibility import collect_environment, derive_seed, set_global_seed


def _cpu_state_dict(model: VariationalDeepJSCC) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def _build_codec(dataset_name: str, config_values: Mapping[str, Any]):
    if dataset_name != "CIFAR10":
        raise ValueError("oracle behavior now supports CIFAR-10 only")
    codec_values = dict(config_values)
    architecture = codec_values.pop("architecture", DEEPJSCC_ARCHITECTURE)
    if architecture not in SUPPORTED_DEEPJSCC_ARCHITECTURES:
        raise ValueError(f"unsupported CIFAR-10 codec architecture: {architecture}")
    model_config = variational_deepjscc_config_for_architecture(
        architecture, **codec_values
    )
    return VariationalDeepJSCC(model_config)


def _new_codec_from_initial(
    dataset_name: str,
    config_values: Mapping[str, Any],
    initial_state: dict[str, Tensor],
    device: torch.device,
) -> VariationalDeepJSCC:
    model = _build_codec(dataset_name, config_values).to(device)
    model.load_state_dict(initial_state)
    return model


def _training_seeds(global_seed: int, model_id: str) -> dict[str, int]:
    return {
        "data_order": derive_seed(global_seed, f"{model_id}:data-order"),
        "posterior": derive_seed(global_seed, f"{model_id}:posterior"),
        "channel": derive_seed(global_seed, f"{model_id}:channel"),
    }


def _train_with_seed_bundle(
    model: VariationalDeepJSCC,
    dataset: Dataset[Any],
    config: TrainingConfig,
    device: torch.device,
    seeds: dict[str, int],
) -> Any:
    """SCU와 공유하는 seed bundle을 trainer의 명시적 argument에 연결한다."""
    return train_reconstruction(
        model,
        dataset,
        config,
        device,
        data_order_seed=seeds["data_order"],
        posterior_seed=seeds["posterior"],
        channel_seed=seeds["channel"],
    )


def _records_for_batch(
    refs: list[ProbeRef],
    positions: Tensor,
    model_outputs: dict[str, tuple[Tensor, Posterior]],
    inputs: Tensor,
    condition: str,
    repetitions: int,
) -> list[dict[str, Any]]:
    retrain_reconstruction, retrain_posterior = model_outputs["retrain"]
    records: list[dict[str, Any]] = []
    for model_name, (reconstruction, posterior) in model_outputs.items():
        if reconstruction.ndim == 5:
            mse_input = (
                (reconstruction - inputs.unsqueeze(0)).flatten(start_dim=2).square().mean(dim=2).mean(dim=0)
            )
            functional_output = (
                (reconstruction - retrain_reconstruction)
                .flatten(start_dim=2)
                .square()
                .mean(dim=2)
                .mean(dim=0)
            )
        else:
            mse_input = per_sample_mse(reconstruction, inputs)
            functional_output = per_sample_mse(reconstruction, retrain_reconstruction)
        functional_mu = (posterior.mu - retrain_posterior.mu).square().mean(dim=1)
        functional_latent = symmetric_diagonal_kl(posterior, retrain_posterior)
        if model_name == "retrain":
            functional_output = torch.zeros_like(mse_input)
            functional_mu = torch.zeros_like(mse_input)
            functional_latent = torch.zeros_like(mse_input)
        for batch_position, local_position in enumerate(positions.tolist()):
            reference = refs[local_position]
            records.append(
                {
                    "dataset_split": reference.dataset_split,
                    "source_index": reference.source_index,
                    "probe_partition": reference.probe_partition,
                    "support_group": reference.support_group,
                    "model": model_name,
                    "evaluation_condition": condition,
                    "channel_repetitions": repetitions,
                    "mse_input": float(mse_input[batch_position].cpu()),
                    "fg_output": float(functional_output[batch_position].cpu()),
                    "fg_mu": float(functional_mu[batch_position].cpu()),
                    "fg_latent": float(functional_latent[batch_position].cpu()),
                }
            )
    return records


def evaluate_probe_partition(
    models: dict[str, VariationalDeepJSCC],
    dataset: ProbeDataset,
    device: torch.device,
    evaluation_config: dict[str, Any],
    noise_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """동일 probe/noise로 모든 model을 retrain oracle에 대해 평가한다."""
    loader = DataLoader(
        dataset,
        batch_size=int(evaluation_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    repetitions = int(evaluation_config["awgn_repetitions"])
    snr_db = float(evaluation_config["awgn_snr_db"])
    noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    records: list[dict[str, Any]] = []
    prediction_chunks: dict[str, dict[str, list[Tensor]]] = {
        model_name: {
            "mu": [],
            "logvar": [],
            "posterior_mean_reconstruction": [],
            "awgn_mean": [],
            "awgn_std": [],
        }
        for model_name in models
    }
    input_chunks: list[Tensor] = []

    for model in models.values():
        model.eval()
    with torch.no_grad():
        for inputs, _labels, positions in loader:
            inputs = inputs.to(device)
            deterministic: dict[str, tuple[Tensor, Posterior]] = {}
            awgn: dict[str, tuple[Tensor, Posterior]] = {}
            for model_name, model in models.items():
                output = model.decode_posterior_mean(inputs)
                deterministic[model_name] = (output.reconstruction, output.posterior)
                prediction_chunks[model_name]["mu"].append(output.posterior.mu.cpu())
                prediction_chunks[model_name]["logvar"].append(output.posterior.logvar.cpu())
                prediction_chunks[model_name]["posterior_mean_reconstruction"].append(
                    output.reconstruction.cpu()
                )

            noise = torch.randn(
                repetitions,
                len(inputs),
                next(iter(models.values())).config.latent_dim,
                generator=noise_generator,
            ).to(device) * next(iter(models.values())).awgn_std(snr_db)
            for model_name, model in models.items():
                posterior = deterministic[model_name][1]
                normalized = model.normalized_latent(posterior.mu)
                reconstructions = torch.stack(
                    [model.decode_latent(normalized + noise[draw]) for draw in range(repetitions)]
                )
                awgn[model_name] = (reconstructions, posterior)
                prediction_chunks[model_name]["awgn_mean"].append(
                    reconstructions.mean(dim=0).cpu()
                )
                prediction_chunks[model_name]["awgn_std"].append(
                    reconstructions.std(dim=0, unbiased=False).cpu()
                )

            records.extend(
                _records_for_batch(
                    dataset.refs,
                    positions,
                    deterministic,
                    inputs,
                    "posterior_mean",
                    1,
                )
            )
            records.extend(
                _records_for_batch(
                    dataset.refs,
                    positions,
                    awgn,
                    inputs,
                    f"awgn_{snr_db:g}db",
                    repetitions,
                )
            )
            input_chunks.append(inputs.cpu())

    predictions = {
        "references": [asdict_probe(reference) for reference in dataset.refs],
        "inputs": torch.cat(input_chunks),
        "models": {
            model_name: {
                name: torch.cat(chunks)
                for name, chunks in prediction_chunks[model_name].items()
            }
            for model_name in models
        },
        "fixed_noise_metadata": {
            "seed": noise_seed,
            "snr_db": snr_db,
            "repetitions": repetitions,
            "policy": "same raw normalized-latent AWGN draw for every model and probe repetition",
        },
    }
    return records, predictions


def asdict_probe(reference: ProbeRef) -> dict[str, Any]:
    return {
        "dataset_split": reference.dataset_split,
        "source_index": reference.source_index,
        "probe_partition": reference.probe_partition,
        "support_group": reference.support_group,
    }


def _training_metadata(result: Any, seeds: dict[str, int]) -> dict[str, Any]:
    return {
        "steps": result.steps,
        "examples": getattr(result, "examples", None),
        "target_examples": getattr(result, "target_examples", None),
        "retained_examples": getattr(result, "retained_examples", None),
        "seeds": seeds,
        "config": result.config,
    }


def _epoch_mean_loss_records(model_name: str, loss_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Step-level loss trace를 모델별 epoch 평균 CSV record로 압축한다."""
    totals: dict[int, dict[str, float]] = {}
    counts: dict[int, int] = {}

    for trace in loss_trace:
        epoch = int(trace["epoch"])
        totals.setdefault(epoch, {})
        counts[epoch] = counts.get(epoch, 0) + 1
        for name, value in trace.items():
            if name == "epoch" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[epoch][name] = totals[epoch].get(name, 0.0) + float(value)

    records: list[dict[str, Any]] = []
    for epoch in sorted(totals):
        record: dict[str, Any] = {
            "model": model_name,
            "epoch": epoch,
            "step_count": counts[epoch],
        }
        for name, total in sorted(totals[epoch].items()):
            record[name] = total / counts[epoch]
        records.append(record)
    return records


def run_oracle_behavior(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_root: Path,
    smoke_test: bool = False,
) -> Path:
    """하나의 global seed에서 full/retrain/SCU oracle study를 실행한다."""
    effective_config = build_effective_config(config, smoke_test=smoke_test)
    device = select_device(effective_config)
    set_global_seed(derive_seed(seed, "global"), bool(effective_config["deterministic"]))
    artifacts = create_run_artifacts(output_root, seed)
    artifacts.append_log(f"run_dir={artifacts.run_dir}")
    artifacts.append_log(f"device={device}")

    dataset_name = str(effective_config["dataset"]["name"])
    if dataset_name != "CIFAR10":
        raise ValueError("oracle behavior now supports CIFAR-10 only")
    dataset_config = effective_config["dataset"]
    train_dataset = load_cifar10_train_dataset(dataset_config, WORKSPACE_ROOT)
    test_dataset = load_cifar10_test_dataset(dataset_config, WORKSPACE_ROOT)
    train_labels = dataset_labels(train_dataset)
    test_labels = dataset_labels(test_dataset)
    subset_metadata = None
    if smoke_test:
        train_dataset, train_labels, train_source_indices = (
            select_class_balanced_subset(
                train_dataset,
                int(dataset_config["smoke_train_per_class"]),
                derive_seed(seed, "smoke-train-subset"),
            )
        )
        test_dataset, test_labels, test_source_indices = select_class_balanced_subset(
            test_dataset,
            int(dataset_config["smoke_test_per_class"]),
            derive_seed(seed, "smoke-test-subset"),
        )
        subset_metadata = {
            "train_size": int(len(train_source_indices)),
            "test_size": int(len(test_source_indices)),
            "train_index_sha256": hashlib.sha256(
                train_source_indices.tobytes()
            ).hexdigest(),
            "test_index_sha256": hashlib.sha256(
                test_source_indices.tobytes()
            ).hexdigest(),
        }
    probe_dataset_name = "cifar10"
    train_labels = torch.as_tensor(train_labels, dtype=torch.long)
    test_labels = torch.as_tensor(test_labels, dtype=torch.long)
    split_config = effective_config["split"]
    split_seed = derive_seed(seed, "split")
    split = build_controlled_noniid_split(
        train_labels,
        split_seed=split_seed,
        shared_labels=tuple(split_config["shared_labels"]),
        shared_per_label=int(split_config["shared_per_label"]),
        target_specific_label=int(split_config["target_specific_label"]),
        retained_clients=int(split_config["retained_clients"]),
    )
    probe_seed = derive_seed(seed, "probe")
    probes = select_audit_indices(
        split,
        train_labels,
        test_labels,
        probe_seed=probe_seed,
        target_per_group=int(effective_config["audit"]["target_per_group"]),
        retained_total=int(effective_config["audit"]["retained_total"]),
        dataset_name=probe_dataset_name,
    )
    split_manifest = split.manifest()
    split_manifest.update(
        {
            "dataset": {
                "name": "CIFAR10",
                "version": "torchvision CIFAR10",
                "data_root": str(
                    (WORKSPACE_ROOT / dataset_config["data_root"]).resolve()
                    if not Path(dataset_config["data_root"]).is_absolute()
                    else Path(dataset_config["data_root"])
                ),
                "transform": "ToTensor",
                "smoke_subset": subset_metadata,
            },
            "global_seed": seed,
            "probe_seed": probe_seed,
            "probe_counts": {partition: len(refs) for partition, refs in probes.items()},
        }
    )
    artifacts.write_json("split_manifest.json", split_manifest)

    effective_config["run_seed"] = seed
    effective_config["seed_derivation"] = {
        "initial_model": derive_seed(seed, "initial-model"),
        "split": split_seed,
        "probe": probe_seed,
        "evaluation_noise": derive_seed(seed, "evaluation-noise"),
    }
    artifacts.write_json("config.json", effective_config)
    artifacts.write_json("environment.json", collect_environment(device, WORKSPACE_ROOT))

    set_global_seed(derive_seed(seed, "initial-model"), bool(effective_config["deterministic"]))
    initial_codec = _build_codec(dataset_name, effective_config["codec"])
    initial_state = _cpu_state_dict(initial_codec)
    artifacts.save_checkpoint("initial", initial_state)

    full_dataset = Subset(train_dataset, list(range(len(train_dataset))))
    target_dataset = Subset(train_dataset, split.client_indices["client_0"])
    retained_indices = [
        index
        for client_id, indices in split.client_indices.items()
        if client_id != "client_0"
        for index in indices
    ]
    retained_dataset = Subset(train_dataset, retained_indices)
    training_config = TrainingConfig(**effective_config["training"])
    training_results: dict[str, Any] = {}

    original_seeds = _training_seeds(seed, "original")
    original_result = _train_with_seed_bundle(
        _new_codec_from_initial(dataset_name, effective_config["codec"], initial_state, device),
        full_dataset,
        training_config,
        device,
        original_seeds,
    )
    training_results["original"] = _training_metadata(original_result, original_seeds)
    artifacts.save_checkpoint("original", _cpu_state_dict(original_result.model))
    artifacts.append_log(f"original: steps={original_result.steps} examples={original_result.examples}")

    retrain_seeds = _training_seeds(seed, "retrain")
    retrain_result = _train_with_seed_bundle(
        _new_codec_from_initial(dataset_name, effective_config["codec"], initial_state, device),
        retained_dataset,
        training_config,
        device,
        retrain_seeds,
    )
    training_results["retrain"] = _training_metadata(retrain_result, retrain_seeds)
    artifacts.save_checkpoint("retrain", _cpu_state_dict(retrain_result.model))
    artifacts.append_log(f"retrain: steps={retrain_result.steps} examples={retrain_result.examples}")

    replica_seeds = _training_seeds(seed, "retrain_replica")
    replica_result = _train_with_seed_bundle(
        _new_codec_from_initial(dataset_name, effective_config["codec"], initial_state, device),
        retained_dataset,
        training_config,
        device,
        replica_seeds,
    )
    training_results["retrain_replica"] = _training_metadata(replica_result, replica_seeds)
    artifacts.save_checkpoint("retrain_replica", _cpu_state_dict(replica_result.model))
    artifacts.append_log(f"retrain_replica: steps={replica_result.steps} examples={replica_result.examples}")

    client_config = ClientPosteriorAuditConfig(**effective_config["client_posterior"])
    client_summaries = audit_client_gaussian_sufficiency(
        original_result.model,
        train_dataset,
        split.client_indices,
        train_labels,
        client_config,
        device,
        derive_seed(seed, "client-posterior"),
    )
    artifacts.write_csv(
        "client_posterior/per_client_gaussian.csv",
        [summary.to_record() for summary in client_summaries],
    )
    artifacts.write_json(
        "client_posterior/summary.json",
        client_posterior_summary_payload(client_summaries, client_config),
    )

    scu_config_values = dict(effective_config["scu"])
    scu_config_values.pop("optimizer_state")
    scu_config_values.pop("retained_sampling")
    scu_config = SCUConfig(**scu_config_values)
    forget_seeds = _training_seeds(seed, "scu_forget_only")
    forget_result = adapt_scu_style(
        original_result.model,
        target_dataset,
        retained_dataset,
        scu_config,
        device,
        "forget_only",
        forget_seeds,
    )
    training_results["scu_forget_only"] = _training_metadata(forget_result, forget_seeds)
    artifacts.save_checkpoint("scu_forget_only", _cpu_state_dict(forget_result.model))
    artifacts.append_log(f"scu_forget_only: steps={forget_result.steps} target_examples={forget_result.target_examples}")

    sharing_seeds = _training_seeds(seed, "scu_self_sharing")
    sharing_result = adapt_scu_style(
        original_result.model,
        target_dataset,
        retained_dataset,
        scu_config,
        device,
        "self_sharing",
        sharing_seeds,
    )
    training_results["scu_self_sharing"] = _training_metadata(sharing_result, sharing_seeds)
    artifacts.save_checkpoint("scu_self_sharing", _cpu_state_dict(sharing_result.model))
    artifacts.append_log(f"scu_self_sharing: steps={sharing_result.steps} target_examples={sharing_result.target_examples}")

    training_loss_records: list[dict[str, Any]] = []
    for model_name, result in (
        ("original", original_result),
        ("retrain", retrain_result),
        ("retrain_replica", replica_result),
        ("scu_forget_only", forget_result),
        ("scu_self_sharing", sharing_result),
    ):
        training_loss_records.extend(
            _epoch_mean_loss_records(model_name, result.loss_trace)
        )
    artifacts.write_csv("metrics/training_by_epoch.csv", training_loss_records)

    models = {
        "original": original_result.model,
        "retrain": retrain_result.model,
        "retrain_replica": replica_result.model,
        "scu_forget_only": forget_result.model,
        "scu_self_sharing": sharing_result.model,
    }
    all_records: list[dict[str, Any]] = []
    all_predictions: dict[str, Any] = {}
    for partition, refs in probes.items():
        source_dataset = train_dataset if refs[0].dataset_split.endswith("_train") else test_dataset
        partition_dataset = ProbeDataset(source_dataset, refs)
        records, predictions = evaluate_probe_partition(
            models,
            partition_dataset,
            device,
            effective_config["evaluation"],
            derive_seed(seed, f"evaluation-noise:{partition}"),
        )
        all_records.extend(records)
        all_predictions[partition] = predictions

    artifacts.write_records(all_records)
    summary = summarize_records(
        all_records,
        bootstrap_seed=derive_seed(seed, "metric-bootstrap"),
        resamples=int(effective_config["evaluation"]["bootstrap_resamples"]),
    )
    summary.update(
        {
            "primary_question": "SCU-style output/posterior functional gap to retraining oracle",
            "mse_interpretation": "MSE_input alone is not an unlearning success criterion.",
            "centralized_comparator": "scu_self_sharing directly consumes retained raw data and is not a federated method.",
            "retrain_noise_floor_model": "retrain_replica compared against retrain",
            "training": training_results,
        }
    )
    artifacts.write_json("metrics/summary.json", summary)
    artifacts.save_predictions(all_predictions)
    save_oracle_figures(
        all_predictions,
        client_summaries,
        artifacts.path("figures"),
        int(effective_config["evaluation"]["figure_probes_per_group"]),
    )
    artifacts.append_log("completed")
    return artifacts.run_dir
