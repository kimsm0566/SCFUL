import csv
import torch
import torch.nn.functional as F
import copy
import torch.optim as optim
import re
import shutil
import time
from pathlib import Path

from src.data.data import create_scu_dataset, get_scu_dataloader, split_scu_data
from src.models.semantic_codec import VariationalDeepJSCC


def _prepare_frozen_semantic_classifier(
    comm,
    data_state,
    config,
    device,
    seed,
    run_dir,
    artifacts,
    observer_run=None,
):
    """Prepare once on rank zero and broadcast one frozen observer."""

    from src.utils.semantic_classifier import (
        load_external_semantic_classifier,
        make_semantic_classifier,
        train_semantic_classifier,
        load_remaining_classifier,
        classifier_state_sha256,
    )

    rank = comm.Get_rank()
    payload = None
    started = time.perf_counter()
    if rank == 0:
        try:
            source_run = config["semantic_classifier"].get("source_run")
            if observer_run is not None:
                if (float(config.get("training", {}).get("semantic_weight", 0.0)) != 0
                        or float(config.get("unlearning", {}).get("lambda_repair_semantic", 0.0)) != 0):
                    raise ValueError("remaining observer is evaluation-only; classifier gradients to codec are forbidden")
                observer, metadata = load_remaining_classifier(
                    observer_run, config, data_state["split_manifest"], device=device,
                )
                state = {key: value.detach().cpu().clone() for key, value in observer.state_dict().items()}
                source_checkpoint = Path(observer_run).resolve() / "checkpoints/observer.pt"
                workspace = Path(__file__).resolve().parents[2]
                import hashlib
                import numpy as np
                indices = sorted(index for row in metadata["protocol"]["client_splits"] for index in row["train_indices"])
                metadata.update({
                    "architecture": metadata["architecture_config"]["architecture"],
                    "preparation": "loaded_remaining_client_classifier",
                    "source_run": str(Path(observer_run).resolve()),
                    "source_run_workspace_relative": str(Path(observer_run).resolve().relative_to(workspace)),
                    "source_checkpoint": str(source_checkpoint),
                    "source_checkpoint_relative": "checkpoints/observer.pt",
                    "source_checkpoint_sha256": metadata["checkpoint_sha256"],
                    "state_sha256": classifier_state_sha256(state),
                    "source_split_type": metadata["source_scenario"]["partition"],
                    "source_split_seed": metadata["source_scenario"]["split_seed"],
                    "source_split_training_pool_index_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
                    "source_global_seed": metadata["protocol"]["seed"],
                })
                preparation = "loaded_remaining_client_classifier"
            elif source_run is None:
                state, metadata = train_semantic_classifier(
                    config,
                    data_state["classifier_data"],
                    data_state["test_data"],
                    seed,
                    device,
                )
                source_checkpoint = None
                preparation = "trained_on_reserved_clean_data"
            else:
                (
                    state,
                    metadata,
                    source_checkpoint,
                ) = load_external_semantic_classifier(
                    config,
                    data_state["test_data"],
                    device,
                )
                preparation = "loaded_external_classifier"
            payload = {
                "error": None,
                "state": state,
                "metadata": metadata,
                "source_checkpoint": (
                    str(source_checkpoint)
                    if source_checkpoint is not None
                    else None
                ),
                "preparation": preparation,
            }
        except Exception as error:
            payload = {
                "error": f"{type(error).__name__}: {error}",
                "state": None,
                "metadata": None,
                "source_checkpoint": None,
                "preparation": None,
            }
    payload = comm.bcast(payload, root=0)
    if payload["error"] is not None:
        raise RuntimeError(
            "semantic classifier preparation failed: " + payload["error"]
        )
    classifier = make_semantic_classifier(
        {"semantic_classifier": payload["metadata"].get("architecture_config", config["semantic_classifier"])},
        state=payload["state"], device=device, frozen=True
    )
    elapsed = time.perf_counter() - started

    if payload["metadata"].get("kind") == "remaining_client_classifier_v1":
        config.setdefault("evaluation", {})["observer_override"] = {
            "source_run": payload["metadata"]["observer_run"],
            "state_sha256": payload["metadata"]["state_sha256"],
            "protocol_sha256": payload["metadata"]["protocol_sha256"],
            "role": "shared_frozen_remaining_client_observer",
        }
        if rank == 0:
            artifacts.write_json("config.json", config)

    if rank == 0:
        classifier_directory = Path(run_dir) / "semantic_classifier"
        classifier_directory.mkdir(parents=True, exist_ok=True)
        classifier_copy = classifier_directory / "classifier.pt"
        if payload["source_checkpoint"] is None:
            torch.save(payload["state"], classifier_copy)
        else:
            shutil.copyfile(payload["source_checkpoint"], classifier_copy)
        metadata = dict(payload["metadata"])
        metadata["wall_seconds"] = elapsed
        if payload["source_checkpoint"] is None:
            training_data_metadata = dict(data_state["classifier_metadata"])
            metadata["training_data"] = training_data_metadata
            manifest_metadata = {
                "architecture": metadata["architecture"],
                "training_seed": metadata["training_seed"],
                "state_sha256": metadata["state_sha256"],
                "role": metadata["role"],
                "training_data": training_data_metadata,
            }
        else:
            current_pool_hash = data_state["training_pool_metadata"].get(
                "selected_index_sha256"
            )
            source_pool_hash = metadata[
                "source_split_training_pool_index_sha256"
            ]
            metadata["current_split_training_pool_index_sha256"] = (
                current_pool_hash
            )
            metadata["source_pool_matches_current_training_pool"] = (
                source_pool_hash == current_pool_hash
            )
            metadata["classifier_copy"] = str(classifier_copy)
            manifest_metadata = {
                name: metadata[name]
                for name in (
                    "architecture",
                    "architecture_config",
                    "role",
                    "preparation",
                    "source_run",
                    "source_run_workspace_relative",
                    "source_checkpoint",
                    "source_checkpoint_relative",
                    "source_checkpoint_sha256",
                    "state_sha256",
                    "source_split_type",
                    "source_split_seed",
                    "source_split_training_pool_index_sha256",
                    "current_split_training_pool_index_sha256",
                    "source_pool_matches_current_training_pool",
                    "source_global_seed",
                    "raw_cifar10_test_loss",
                    "raw_cifar10_test_accuracy",
                    "raw_cifar10_test_examples",
                )
            }
        if observer_run is not None:
            manifest_metadata["remaining_observer_protocol_sha256"] = metadata["protocol_sha256"]
            manifest_metadata["remaining_observer_source_fingerprint"] = metadata["protocol"]["source_fingerprint"]
        data_state["classifier_metadata"] = manifest_metadata
        data_state["split_manifest"]["semantic_classifier_training"] = (
            manifest_metadata
        )
        artifacts.write_json(
            "semantic_classifier/metadata.json", metadata
        )
        if "history" in metadata:
            artifacts.write_csv(
                "semantic_classifier/training_history.csv", metadata["history"]
            )
        artifacts.write_json("split_manifest.json", data_state["split_manifest"])
    return classifier, payload["metadata"], elapsed


def _make_round_reporter(artifacts):
    """Write every server round to both the terminal and the run artifact log."""

    def report(message):
        print(message, flush=True)
        artifacts.append_log(message)

    return report


def _scu_random_generator(device, seed):
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _kl_to_standard_normal(posterior):
    return 0.5 * (
        posterior.mu.square() + posterior.logvar.exp() - 1.0 - posterior.logvar
    ).mean(dim=1)


def _posterior_kl(posterior_p, posterior_q):
    """각 sample에서 KL(q_p || q_q)을 계산한다."""
    return 0.5 * (
        posterior_q.logvar
        - posterior_p.logvar
        + (
            posterior_p.logvar.exp()
            + (posterior_p.mu - posterior_q.mu).square()
        )
        / posterior_q.logvar.exp()
        - 1.0
    ).mean(dim=1)


def _train_original_scu_model(model, dataset, args):
    """삭제 전 전체 client data로 semantic codec을 학습한다."""
    device = args.device
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    beta = getattr(args, "beta", 1e-4)
    loss_trace = []
    steps = 0
    posterior_generator = _scu_random_generator(device, args.seed + 10)
    channel_generator = _scu_random_generator(device, args.seed + 11)

    for epoch in range(args.n_epochs):
        loader = get_scu_dataloader(dataset, args.batch_size, args.seed + epoch)
        for images, _ in loader:
            images = images.to(device)
            output = model(
                images,
                mode="training",
                snr_db=args.snr_db,
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            reconstruction_loss = F.mse_loss(output.reconstruction, images)
            kl_loss = _kl_to_standard_normal(output.posterior).mean()
            total_loss = reconstruction_loss + beta * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_trace.append(
                {
                    "epoch": epoch,
                    "loss": float(total_loss.detach().cpu()),
                    "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                    "kl_loss": float(kl_loss.detach().cpu()),
                }
            )
            steps += 1

    return loss_trace, steps


def _run_scu_unlearning(model, original_model, target_data, retained_data, args):
    """D^t를 잊고 D^r reconstruction을 유지하도록 같은 client codec을 갱신한다."""
    device = args.device
    model.train()
    original_model.eval()
    for parameter in original_model.parameters():
        parameter.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mode = getattr(args, "scu_mode", "self_sharing")
    if mode not in ("forget_only", "self_sharing"):
        raise ValueError("scu_mode must be 'forget_only' or 'self_sharing'")

    lambda_kl = getattr(args, "lambda_kl", 1.0)
    lambda_rec = getattr(args, "lambda_rec", 0.1)
    lambda_ss = getattr(args, "lambda_ss", 1.0)
    beta = getattr(args, "beta", 1e-4)
    epsilon = getattr(args, "epsilon", 1e-8)
    epochs = getattr(args, "scu_epochs", args.n_epochs)

    loss_trace = []
    steps = 0
    target_examples = 0
    retained_examples = 0
    posterior_generator = _scu_random_generator(device, args.seed + 20)
    channel_generator = _scu_random_generator(device, args.seed + 21)

    for epoch in range(epochs):
        target_loader = get_scu_dataloader(
            target_data, args.batch_size, args.seed + 1000 + 2 * epoch
        )
        retained_loader = get_scu_dataloader(
            retained_data, args.batch_size, args.seed + 1001 + 2 * epoch
        )
        retained_iterator = iter(retained_loader)

        for target_images, _ in target_loader:
            try:
                retained_images, _ = next(retained_iterator)
            except StopIteration:
                retained_loader = get_scu_dataloader(
                    retained_data,
                    args.batch_size,
                    args.seed + 2000 + epoch + steps,
                )
                retained_iterator = iter(retained_loader)
                retained_images, _ = next(retained_iterator)

            target_images = target_images.to(device)
            retained_images = retained_images.to(device)
            with torch.no_grad():
                original_target_posterior = original_model(
                    target_images, mode="posterior"
                )

            target_output = model(
                target_images,
                mode="training",
                snr_db=args.snr_db,
                posterior_generator=posterior_generator,
                channel_generator=channel_generator,
            )
            forget_kl = _posterior_kl(
                target_output.posterior, original_target_posterior
            ).mean()
            target_mse = F.mse_loss(target_output.reconstruction, target_images)
            # Target posterior는 original에서 멀어지고, target reconstruction은 나빠진다.
            forget_loss = lambda_kl * forget_kl - lambda_rec * target_mse

            retained_mse = None
            retained_kl = None
            retained_loss = None
            if mode == "forget_only":
                total_loss = forget_loss
                weight_forget = torch.ones((), device=device)
                weight_retain = torch.zeros((), device=device)
            else:
                retained_output = model(
                    retained_images,
                    mode="training",
                    snr_db=args.snr_db,
                    posterior_generator=posterior_generator,
                    channel_generator=channel_generator,
                )
                retained_mse = F.mse_loss(
                    retained_output.reconstruction, retained_images
                )
                retained_kl = _kl_to_standard_normal(retained_output.posterior).mean()
                # Retained data는 일반 VAE reconstruction 품질을 유지한다.
                retained_loss = beta * retained_kl + retained_mse
                denominator = (
                    forget_loss.detach().abs()
                    + retained_loss.detach().abs()
                    + epsilon
                )
                weight_forget = forget_loss.detach().abs() / denominator
                weight_retain = retained_loss.detach().abs() / denominator
                total_loss = (
                    weight_forget * forget_loss
                    + weight_retain * lambda_ss * retained_loss
                )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            trace = {
                "epoch": epoch,
                "loss": float(total_loss.detach().cpu()),
                "forget_loss": float(forget_loss.detach().cpu()),
                "forget_kl_q_u_to_q_g": float(forget_kl.detach().cpu()),
                "target_mse": float(target_mse.detach().cpu()),
                "weight_forget": float(weight_forget.detach().cpu()),
                "weight_retain": float(weight_retain.detach().cpu()),
            }
            if retained_loss is not None:
                trace["retain_loss"] = float(retained_loss.detach().cpu())
                trace["retained_mse"] = float(retained_mse.detach().cpu())
                trace["retained_kl_to_prior"] = float(retained_kl.detach().cpu())
            loss_trace.append(trace)
            steps += 1
            target_examples += len(target_images)
            retained_examples += len(retained_images)

    return loss_trace, steps, target_examples, retained_examples


FSCU_PUBLIC_MODE = "scu_public_compensation"
FSCU_REQUIRED_METRICS = (
    "clean_rmse_255",
    "erased_rmse_255",
    "clean_psnr_db",
    "erased_psnr_db",
    "clean_fid_to_retrain",
    "erased_fid_to_retrain",
    "model_accuracy",
    "backdoor_test_probe_asr",
    "mia_auc",
    "backdoor_accuracy",
    "running_time_seconds",
)
AUXILIARY_EXPERIMENTS = {
    "controlled_behavior",
    "reconstruction_channel_audit",
    "membership_inference_audit",
    "support_response",
}


def _validate_fscu_output_root(output_root):
    if output_root is None:
        raise ValueError("FSCU requires --output-root")
    output_root = Path(output_root)
    pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-.+"
    if re.fullmatch(pattern, output_root.name) is None:
        raise ValueError(
            "FSCU output folder must use YYYY-MM-DD-Index-title format"
        )
    return output_root


def _read_fscu_headline_metrics(run_dir, algorithm="FSCU"):
    metrics_path = Path(run_dir) / "metrics" / "headline_metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(
            f"{algorithm} did not create metrics/headline_metrics.csv"
        )
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fscu_rows = [row for row in rows if row["model"] == algorithm]
    if len(fscu_rows) != 1:
        raise RuntimeError(
            f"headline metrics must contain exactly one {algorithm} row"
        )
    missing = [
        name
        for name in FSCU_REQUIRED_METRICS
        if fscu_rows[0].get(name) in (None, "")
    ]
    if missing:
        raise RuntimeError(
            f"{algorithm} headline metrics are missing values: {missing}"
        )
    return metrics_path, fscu_rows[0]




def _scu_result_directory(args):
    return Path(args.result_path).joinpath(
        args.dataset,
        f"n_clients_{args.n_clients}",
        f"n_client_data_{args.n_client_data}",
        f"batch_size_{args.batch_size}",
        f"data_partition_type_{args.partition_type}",
        f"major_percent_{args.major_percent}",
        f"n_epochs_{args.n_epochs}",
        args.algorithm,
        f"snr_{args.snr_db}",
    )


def train_scu(comm, data_state, args, logger):
    """한 client의 data로 original training과 SCU unlearning을 순서대로 실행한다."""
    from src.utils.client import Client
    from src.utils.logger import get_logger

    node = data_state.get("node")
    if node is None:
        requested_device = args.device
        if isinstance(requested_device, torch.device):
            device = requested_device
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{requested_device}")
        else:
            device = torch.device("cpu")
        args.device = device
        node = Client(
            data=data_state["client_data"],
            private_test_data=data_state.get("test_data"),
            args=args,
            rank=0,
        )

    result_directory = None
    if hasattr(args, "result_path"):
        result_directory = _scu_result_directory(args)
        result_directory.mkdir(parents=True, exist_ok=True)
        if logger is None:
            logger = get_logger(
                str(result_directory / f"seed_{args.seed}_client_0.log")
            )

    if args.dataset != "cifar10":
        raise ValueError("SCU now supports CIFAR-10 only")
    expected_model = VariationalDeepJSCC

    if not isinstance(node.model, expected_model):
        raise ValueError(f"SCU requires {expected_model.__name__} for {args.dataset}")

    # 1. 이 client가 실제로 학습했던 전체 data D를 준비한다.
    full_data = create_scu_dataset(node.data, args.dataset)
    forget_ratio = getattr(args, "scu_forget_ratio", 0.06)

    # 2. D를 삭제할 D^t와 계속 보유할 D^r로 한 번만 나눈다.
    target_data, retained_data, target_indices, retained_indices = split_scu_data(
        full_data, forget_ratio, args.seed
    )

    # 3. 전체 D로 original codec을 학습하고, 이를 frozen teacher로 보관한다.
    original_loss_trace, original_steps = _train_original_scu_model(
        node.model, full_data, args
    )
    original_model = copy.deepcopy(node.model).to(args.device)

    # 4. 같은 codec의 encoder와 decoder를 함께 갱신해 D^t를 unlearn한다.
    scu_loss_trace, steps, target_examples, retained_examples = _run_scu_unlearning(
        node.model, original_model, target_data, retained_data, args
    )

    if logger:
        logger.info(
            "[SCU] original steps=%d, unlearning steps=%d, target=%d, retained=%d",
            original_steps,
            steps,
            target_examples,
            retained_examples,
        )

    node.original_model = original_model
    result = {
        "algorithm": "SCU",
        "original_model": original_model,
        "unlearned_model": node.model,
        "original_loss_trace": original_loss_trace,
        "scu_loss_trace": scu_loss_trace,
        "original_steps": original_steps,
        "steps": steps,
        "target_examples": target_examples,
        "retained_examples": retained_examples,
        "target_indices": target_indices,
        "retained_indices": retained_indices,
    }

    if result_directory is not None:
        checkpoint_path = result_directory / f"seed_{args.seed}_scu.pt"
        torch.save(
            {
                "seed": args.seed,
                "target_indices": target_indices,
                "retained_indices": retained_indices,
                "original_loss_trace": original_loss_trace,
                "scu_loss_trace": scu_loss_trace,
                "original_model_state_dict": original_model.state_dict(),
                "unlearned_model_state_dict": node.model.state_dict(),
            },
            checkpoint_path,
        )
        result["run_dir"] = str(result_directory)
        result["checkpoint_path"] = str(checkpoint_path)
        result["messages"] = [
            "SCU Finished & Saved.",
            f"Checkpoint: {checkpoint_path}",
        ]
    return result


def train_fscu(comm, data_state, args, logger):
    """Run one visible SPMD FSCU flow on every MPI rank."""
    from src.experiments.reconstruction_channel_audit import (
        run_reconstruction_channel_audit,
    )
    from src.utils.artifacts import create_run_artifacts
    from src.utils.fscu import train_server_public_contrastive_compensation
    from src.utils.fscu_runtime import (
        build_fscu_reference_manifest,
        load_fscu_reference_models,
        make_fscu_codec,
        make_initial_fscu_state,
        make_public_compensation_config,
        model_state_nbytes,
        run_federated_phase,
        run_target_joint_unlearning,
        select_fscu_device,
        split_interleaved_epoch_budget,
    )
    from src.utils.reproducibility import (
        collect_environment,
        derive_seed,
        set_global_seed,
    )

    rank = comm.Get_rank()
    is_server = rank == 0
    config = data_state["config"]
    num_clients = int(config["federation"]["n_clients"])

    output_root = _validate_fscu_output_root(args.output_root)
    device = select_fscu_device(config)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(
        derive_seed(args.seed, f"federated-rank-{rank}"),
        bool(config.get("deterministic", True)),
    )

    artifacts = None
    setup = None
    if is_server:
        try:
            artifacts = create_run_artifacts(output_root, args.seed)
            artifacts.write_json("config.json", config)
            artifacts.write_json(
                "environment.json",
                collect_environment(device, Path(__file__).resolve().parents[2]),
            )
            artifacts.append_log(f"seed={args.seed}")
            artifacts.append_log(f"device={device}")
            setup = {"error": None, "run_dir": str(artifacts.run_dir)}
        except Exception as error:
            setup = {
                "error": f"{type(error).__name__}: {error}",
                "run_dir": None,
            }
    setup = comm.bcast(setup, root=0)
    if setup["error"] is not None:
        raise RuntimeError(f"FSCU run setup failed: {setup['error']}")
    run_dir = Path(setup["run_dir"])
    round_reporter = _make_round_reporter(artifacts) if is_server else None
    semantic_classifier, classifier_metrics, classifier_seconds = (
        _prepare_frozen_semantic_classifier(
            comm,
            data_state,
            config,
            device,
            args.seed,
            run_dir,
            artifacts,
            observer_run=getattr(args, "observer_run", None),
        )
    )

    split_seed = int(data_state["split_seed"])
    client_data = data_state["client_data"]
    target_client_id = int(config["unlearning"]["target_client_id"])
    retrain_name = f"retrain_without_client_{target_client_id}"
    reference_run = getattr(args, "reference_run", None)

    reference_models = None
    reference_rejection_reason = None
    preparation = None
    if is_server:
        try:
            split = data_state["split"]
            expected_samples = num_clients * int(
                config["federation"]["samples_per_client"]
            )
            if sum(len(indices) for indices in split.train_indices) != expected_samples:
                raise ValueError("dataset size and samples_per_client do not match")
            split_manifest = data_state["split_manifest"]
            artifacts.write_json(
                "run_metadata.json",
                {
                    "global_seed": int(args.seed),
                    "split_seed": int(split_seed),
                    "split_type": config["split"].get(
                        "type", "equal_size_dirichlet"
                    ),
                    "target_client_id": target_client_id,
                },
            )
            artifacts.write_json("split_manifest.json", split_manifest)

            initial_state = make_initial_fscu_state(config, args.seed)
            reference_loading_seconds = 0.0
            if reference_run is not None:
                reference_start = time.perf_counter()
                try:
                    reference_models = load_fscu_reference_models(
                        reference_run,
                        config,
                        split_manifest,
                        initial_state,
                        args.seed,
                    )
                except Exception as error:
                    reference_rejection_reason = (
                        f"{type(error).__name__}: {error}"
                    )
                    artifacts.append_log(
                        f"reference_rejected={reference_rejection_reason}"
                    )
                reference_loading_seconds = time.perf_counter() - reference_start
            if reference_models is not None:
                initial_state = reference_models["initial_state"]
                for name, source in reference_models["checkpoint_paths"].items():
                    artifacts.path(f"checkpoints/{name}.pt").symlink_to(source)
                artifacts.append_log(
                    f"reference_models_reused={reference_models['reference_run']}"
                )
            else:
                artifacts.save_checkpoint("initial", initial_state)
                artifacts.append_log("reference_models_reused=false")
            preparation = {
                "error": None,
                "reuse_reference": reference_models is not None,
                "reference_loading_seconds": reference_loading_seconds,
                "reference_rejection_reason": reference_rejection_reason,
            }
        except Exception as error:
            preparation = {
                "error": f"{type(error).__name__}: {error}",
                "reuse_reference": False,
                "reference_loading_seconds": None,
                "reference_rejection_reason": None,
            }
    preparation = comm.bcast(preparation, root=0)
    if preparation["error"] is not None:
        raise RuntimeError(f"FSCU reference preparation failed: {preparation['error']}")

    reuse_reference = bool(preparation["reuse_reference"])
    phase_seconds = {"semantic_classifier_training": classifier_seconds}
    original_history = []
    retrain_history = []
    if reuse_reference:
        original_state = reference_models["original_state"] if is_server else None
        retrain_state = reference_models["retrain_state"] if is_server else None
        if is_server:
            phase_seconds.update(reference_models["training_wall_seconds"])
            phase_seconds["reference_loading"] = preparation[
                "reference_loading_seconds"
            ]
    else:
        initial_state = initial_state if is_server else None
        phase_start = time.perf_counter()
        original_state, original_history = run_federated_phase(
            comm,
            config,
            device,
            initial_state,
            set(range(num_clients)),
            client_data,
            "original",
            args.seed,
            semantic_classifier=semantic_classifier,
            round_reporter=round_reporter,
        )
        if is_server:
            phase_seconds["original_training"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        retrain_state, retrain_history = run_federated_phase(
            comm,
            config,
            device,
            initial_state,
            set(range(num_clients)) - {target_client_id},
            client_data,
            retrain_name,
            args.seed,
            semantic_classifier=semantic_classifier,
            round_reporter=round_reporter,
        )
        if is_server:
            phase_seconds["retrain_without_target"] = (
                time.perf_counter() - phase_start
            )

    fixed_original_state = comm.bcast(
        original_state if is_server else None, root=0
    )
    unlearning = config["unlearning"]

    ju_baseline_start = time.perf_counter()
    (
        ju_only_state,
        _,
        ju_only_metrics,
        _,
    ) = run_target_joint_unlearning(
        comm,
        config,
        device,
        original_state if is_server else None,
        fixed_original_state,
        client_data,
        args.seed,
        cycle_id=0,
        epochs=int(unlearning["forgetting_epochs"]),
        extract_buffer=False,
        buffer_source=str(unlearning["negative_buffer_mode"]),
        optimizer_state=None,
        semantic_classifier=semantic_classifier,
    )
    if is_server:
        ju_only_seconds = time.perf_counter() - ju_baseline_start
        phase_seconds["target_client_joint_unlearning"] = ju_only_seconds
        ju_only_metrics["wall_seconds"] = ju_only_seconds
        ju_only_metrics["epochs"] = int(unlearning["forgetting_epochs"])

    interleaving_cycles = int(unlearning["interleaving_cycles"])
    ju_epoch_budget = split_interleaved_epoch_budget(
        unlearning["forgetting_epochs"], interleaving_cycles
    )
    cc_epoch_budget = split_interleaved_epoch_budget(
        unlearning["server_cc_epochs"], interleaving_cycles
    )
    negative_buffer_mode = str(unlearning["negative_buffer_mode"])
    persist_interleaved_optimizer = bool(
        unlearning.get("persist_interleaved_optimizer_state", True)
    )

    current_state = original_state if is_server else None
    joint_state = None
    fscu_state = None
    erased_buffer = None
    target_optimizer_state = None
    server_optimizer_state = None
    joint_cycle_metrics = []
    compensation_cycle_metrics = []
    cycle_wall_seconds = []
    total_ju_seconds = 0.0
    total_cc_seconds = 0.0

    for cycle_id in range(interleaving_cycles):
        refresh_buffer = cycle_id == 0 or negative_buffer_mode == "refresh"
        phase_start = time.perf_counter()
        (
            joint_state,
            refreshed_buffer,
            joint_metrics,
            target_optimizer_state,
        ) = run_target_joint_unlearning(
            comm,
            config,
            device,
            current_state,
            fixed_original_state,
            client_data,
            args.seed,
            cycle_id=cycle_id,
            epochs=ju_epoch_budget[cycle_id],
            extract_buffer=refresh_buffer,
            buffer_source=negative_buffer_mode,
            optimizer_state=(
                target_optimizer_state if persist_interleaved_optimizer else None
            ),
            semantic_classifier=semantic_classifier,
        )
        if not persist_interleaved_optimizer:
            target_optimizer_state = None

        if is_server:
            ju_seconds = time.perf_counter() - phase_start
            total_ju_seconds += ju_seconds
            if refreshed_buffer is not None:
                erased_buffer = refreshed_buffer
            if erased_buffer is None:
                raise RuntimeError("FSCU has no erased representation buffer")
            joint_metrics["wall_seconds"] = ju_seconds
            joint_metrics["epochs"] = ju_epoch_budget[cycle_id]
            joint_cycle_metrics.append(joint_metrics)

            joint_model = make_fscu_codec(config)
            joint_model.load_state_dict(joint_state)
            phase_start = time.perf_counter()
            (
                fscu_state,
                compensation_metrics,
                server_optimizer_state,
            ) = train_server_public_contrastive_compensation(
                joint_model,
                data_state["public_data"],
                erased_buffer,
                make_public_compensation_config(
                    config, device, epochs=cc_epoch_budget[cycle_id]
                ),
                derive_seed(
                    args.seed,
                    f"server-public-contrastive-compensation:cycle-{cycle_id}",
                ),
                optimizer_state=(
                    server_optimizer_state
                    if persist_interleaved_optimizer
                    else None
                ),
                return_optimizer_state=True,
                semantic_classifier=semantic_classifier,
            )
            if not persist_interleaved_optimizer:
                server_optimizer_state = None
            cc_seconds = time.perf_counter() - phase_start
            total_cc_seconds += cc_seconds
            compensation_metrics["cycle"] = cycle_id + 1
            compensation_metrics["epochs"] = cc_epoch_budget[cycle_id]
            compensation_metrics["wall_seconds"] = cc_seconds
            compensation_cycle_metrics.append(compensation_metrics)
            cycle_wall_seconds.append(
                {
                    "cycle": cycle_id + 1,
                    "target_joint_unlearning": ju_seconds,
                    "server_public_compensation": cc_seconds,
                    "total": ju_seconds + cc_seconds,
                }
            )
            current_state = fscu_state

    if is_server:
        phase_seconds["interleaved_target_client_joint_unlearning"] = (
            total_ju_seconds
        )
        phase_seconds["server_public_contrastive_compensation"] = total_cc_seconds
        phase_seconds["unlearning_path_total"] = (
            total_ju_seconds + total_cc_seconds
        )

        def summarize_cycles(records):
            total_steps = sum(int(record["steps"]) for record in records)
            summary = {"steps": total_steps, "cycles": records}
            if total_steps == 0:
                return summary
            names = set().union(*(record.keys() for record in records))
            excluded = {
                "steps",
                "cycle",
                "epochs",
                "wall_seconds",
                "erased_buffer_bytes",
                "erased_buffer_examples",
                "erased_buffer_latent_dim",
            }
            for name in names - excluded:
                values = [record.get(name) for record in records]
                if all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in values
                ):
                    summary[name] = sum(
                        float(value) * int(record["steps"])
                        for value, record in zip(values, records)
                    ) / total_steps
            return summary

        joint_metrics = summarize_cycles(joint_cycle_metrics)
        compensation_metrics = summarize_cycles(compensation_cycle_metrics)
        final_buffer_bytes = int(
            erased_buffer.numel() * erased_buffer.element_size()
        )
        transmitted_buffer_bytes = sum(
            int(record["erased_buffer_bytes"])
            for record in joint_cycle_metrics
        )
        joint_metrics.update(
            {
                "erased_buffer_bytes": final_buffer_bytes,
                "erased_buffer_transmitted_bytes": transmitted_buffer_bytes,
                "buffer_refreshes": sum(
                    int(record["erased_buffer_refreshed"])
                    for record in joint_cycle_metrics
                ),
                "negative_buffer_mode": negative_buffer_mode,
            }
        )

        checkpoints = {
            "client_joint_unlearning": ju_only_state,
            "FSCU_last_joint_step": joint_state,
            "FSCU": fscu_state,
        }
        if not reuse_reference:
            checkpoints["original"] = original_state
            checkpoints[retrain_name] = retrain_state
        for name, state in checkpoints.items():
            artifacts.save_checkpoint(name, state)

        if not reuse_reference:
            artifacts.write_json(
                "reference_manifest.json",
                build_fscu_reference_manifest(
                    run_dir,
                    config,
                    split_manifest,
                    initial_state,
                    args.seed,
                    phase_seconds,
                ),
            )

        reference_provenance = {
            "requested_reference": (
                str(reference_run) if reference_run is not None else None
            ),
            "reused": reuse_reference,
            "source_run": (
                reference_models["reference_run"] if reuse_reference else None
            ),
            "scenario_fingerprint_sha256": (
                reference_models["scenario_fingerprint_sha256"]
                if reuse_reference
                else None
            ),
            "checkpoint_sha256": (
                reference_models["checkpoint_sha256"] if reuse_reference else None
            ),
            "reference_manifest_verified": (
                reference_models["reference_manifest_verified"]
                if reuse_reference
                else False
            ),
            "rejection_reason": preparation["reference_rejection_reason"],
        }
        artifacts.write_json("reference_source.json", reference_provenance)

        model_bytes = model_state_nbytes(original_state)
        buffer_bytes = int(joint_metrics["erased_buffer_bytes"])
        transmitted_buffer_bytes = int(
            joint_metrics["erased_buffer_transmitted_bytes"]
        )
        online_seconds = phase_seconds["unlearning_path_total"]
        retrain_seconds = phase_seconds.get("retrain_without_target")
        efficiency = {
            "phase_wall_seconds": phase_seconds,
            "interleaving_cycle_wall_seconds": cycle_wall_seconds,
            "interleaving_cycles": interleaving_cycles,
            "joint_unlearning_epoch_budget": ju_epoch_budget,
            "server_compensation_epoch_budget": cc_epoch_budget,
            "persist_interleaved_optimizer_state": (
                persist_interleaved_optimizer
            ),
            "negative_buffer_mode": negative_buffer_mode,
            "reference_models_reused": reuse_reference,
            "reference_run": reference_provenance["source_run"],
            "model_state_bytes": model_bytes,
            "proposed_online_wall_seconds": online_seconds,
            "speedup_vs_retraining": (
                retrain_seconds / online_seconds
                if retrain_seconds is not None
                else None
            ),
            "proposed_algorithmic_communication_bytes": int(
                (1 + 2 * interleaving_cycles) * model_bytes
                + transmitted_buffer_bytes
            ),
            "proposed_target_update_steps": int(joint_metrics["steps"]),
            "proposed_server_update_steps": int(compensation_metrics["steps"]),
            "coevaluated_joint_unlearning_baseline_wall_seconds": (
                ju_only_seconds
            ),
            "coevaluated_joint_unlearning_baseline_update_steps": int(
                ju_only_metrics["steps"]
            ),
            "erased_representation_buffer_bytes": buffer_bytes,
            "erased_representation_transmitted_bytes": (
                transmitted_buffer_bytes
            ),
        }
        artifacts.write_json("metrics/efficiency.json", efficiency)
        artifacts.write_json(
            "metrics/training_summary.json",
            {
                "reference_model": retrain_name,
                "reference_models_reused": reuse_reference,
                "reference_run": reference_provenance["source_run"],
                "training_history": original_history + retrain_history,
                "joint_unlearning_only_baseline": ju_only_metrics,
                "target_client_joint_unlearning": joint_metrics,
                "server_public_contrastive_compensation": compensation_metrics,
                "interleaving": {
                    "cycles": interleaving_cycles,
                    "joint_unlearning_epoch_budget": ju_epoch_budget,
                    "server_compensation_epoch_budget": cc_epoch_budget,
                    "negative_buffer_mode": negative_buffer_mode,
                    "canonical_latent": unlearning["canonical_latent"],
                    "public_positive_mode": unlearning[
                        "public_positive_mode"
                    ],
                },
            },
        )
        artifacts.append_log("training_and_unlearning_completed")
        del erased_buffer

    comm.Barrier()
    completion = None
    if is_server:
        try:
            run_reconstruction_channel_audit(
                source_run=run_dir,
                output_root=None,
                seed=args.seed,
                device=args.audit_device,
                snr_db=float(config["evaluation"]["snr_db"]),
                batch_size=int(config["evaluation"]["batch_size"]),
                in_place=True,
            )
            metrics_path, headline_metrics = _read_fscu_headline_metrics(run_dir)
            completion = {
                "error": None,
                "result": {
                    "algorithm": "FSCU",
                    "run_dir": str(run_dir),
                    "headline_metrics_path": str(metrics_path),
                    "headline_metrics": headline_metrics,
                    "messages": [
                        f"FSCU finished: {run_dir}",
                        f"Headline metrics: {metrics_path}",
                    ],
                },
            }
            if logger:
                logger.info("[FSCU] completed run=%s", run_dir)
        except Exception as error:
            completion = {
                "error": f"{type(error).__name__}: {error}",
                "result": None,
            }
    completion = comm.bcast(completion, root=0)
    if completion["error"] is not None:
        raise RuntimeError(f"FSCU automatic evaluation failed: {completion['error']}")
    return completion["result"] if is_server else None



def train_fscuv4(comm, data_state, args, logger):
    """Run joint forgetting plus configurable retained-client repair."""
    from src.experiments.reconstruction_channel_audit import (
        run_reconstruction_channel_audit,
    )
    from src.utils.artifacts import create_run_artifacts
    from src.utils.fscu_runtime import (
        build_fscu_reference_manifest,
        load_fscu_reference_models,
        make_initial_fscu_state,
        model_state_nbytes,
        run_federated_phase,
        run_fscuv4_retained_repair,
        run_target_joint_unlearning,
        select_fscu_device,
    )
    from src.utils.plotting import (
        save_federated_training_curve_figures,
        save_fscuv4_optimization_figures,
        save_unlearning_headline_figure,
    )
    from src.utils.reproducibility import (
        collect_environment,
        derive_seed,
        set_global_seed,
    )

    rank = comm.Get_rank()
    is_server = rank == 0
    config = data_state["config"]
    unlearning = config["unlearning"]
    num_clients = int(config["federation"]["n_clients"])
    target_client_id = int(unlearning["target_client_id"])
    retained_client_ids = set(range(num_clients)) - {target_client_id}
    retrain_name = f"retrain_without_client_{target_client_id}"
    primary_variant = str(
        unlearning.get("primary_repair_variant", "feature_kd")
    )
    method_label = str(unlearning.get("method_label", "FSCUv4"))
    output_root = _validate_fscu_output_root(args.output_root)
    device = select_fscu_device(config)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(
        derive_seed(args.seed, f"federated-rank-{rank}"),
        bool(config.get("deterministic", True)),
    )

    artifacts = None
    setup = None
    if is_server:
        try:
            artifacts = create_run_artifacts(output_root, args.seed)
            artifacts.write_json("config.json", config)
            artifacts.write_json(
                "environment.json",
                collect_environment(device, Path(__file__).resolve().parents[2]),
            )
            artifacts.append_log(f"seed={args.seed}")
            artifacts.append_log(f"device={device}")
            setup = {"error": None, "run_dir": str(artifacts.run_dir)}
        except Exception as error:
            setup = {
                "error": f"{type(error).__name__}: {error}",
                "run_dir": None,
            }
    setup = comm.bcast(setup, root=0)
    if setup["error"] is not None:
        raise RuntimeError(f"FSCUv4 run setup failed: {setup['error']}")
    run_dir = Path(setup["run_dir"])
    round_reporter = _make_round_reporter(artifacts) if is_server else None
    semantic_classifier, classifier_metrics, classifier_seconds = (
        _prepare_frozen_semantic_classifier(
            comm,
            data_state,
            config,
            device,
            args.seed,
            run_dir,
            artifacts,
            observer_run=getattr(args, "observer_run", None),
        )
    )

    split_seed = int(data_state["split_seed"])
    client_data = data_state["client_data"]
    reference_run = getattr(args, "reference_run", None)
    reference_models = None
    preparation = None
    if is_server:
        try:
            split = data_state["split"]
            expected_samples = num_clients * int(
                config["federation"]["samples_per_client"]
            )
            if sum(len(indices) for indices in split.train_indices) != expected_samples:
                raise ValueError("dataset size and samples_per_client do not match")
            split_manifest = data_state["split_manifest"]
            artifacts.write_json(
                "run_metadata.json",
                {
                    "global_seed": int(args.seed),
                    "split_seed": int(split_seed),
                    "split_type": config["split"].get(
                        "type", "equal_size_dirichlet"
                    ),
                    "target_client_id": target_client_id,
                    "algorithm": method_label,
                    "repair_data_scope": "remaining_clients_only",
                    "teacher": (
                        "none"
                        if primary_variant == "normal_repair"
                        else "frozen_original_global_model"
                    ),
                    "repair_aggregation": unlearning["repair_aggregation"],
                    "conflict_reference": unlearning.get("conflict_reference"),
                },
            )
            artifacts.write_json("split_manifest.json", split_manifest)
            initial_state = make_initial_fscu_state(config, args.seed)
            rejection_reason = None
            loading_seconds = 0.0
            if reference_run is not None:
                loading_start = time.perf_counter()
                try:
                    reference_models = load_fscu_reference_models(
                        reference_run,
                        config,
                        split_manifest,
                        initial_state,
                        args.seed,
                    )
                except Exception as error:
                    reference_models = None
                    rejection_reason = f"{type(error).__name__}: {error}"
                    artifacts.append_log(f"reference_rejected={rejection_reason}")
                loading_seconds = time.perf_counter() - loading_start
            if reference_models is not None:
                initial_state = reference_models["initial_state"]
                for name, source in reference_models["checkpoint_paths"].items():
                    artifacts.path(f"checkpoints/{name}.pt").symlink_to(source)
                artifacts.append_log(
                    f"reference_models_reused={reference_models['reference_run']}"
                )
            else:
                artifacts.save_checkpoint("initial", initial_state)
                artifacts.append_log("reference_models_reused=false")
            preparation = {
                "error": None,
                "reuse_reference": reference_models is not None,
                "reference_loading_seconds": float(loading_seconds),
                "reference_rejection_reason": rejection_reason,
            }
        except Exception as error:
            preparation = {
                "error": f"{type(error).__name__}: {error}",
                "reuse_reference": False,
                "reference_loading_seconds": None,
                "reference_rejection_reason": None,
            }
    preparation = comm.bcast(preparation, root=0)
    if preparation["error"] is not None:
        raise RuntimeError(
            f"FSCUv4 reference preparation failed: {preparation['error']}"
        )

    reuse_reference = bool(preparation["reuse_reference"])
    phase_seconds = {"semantic_classifier_training": classifier_seconds}
    original_history = []
    retrain_history = []
    if reuse_reference:
        original_state = reference_models["original_state"] if is_server else None
        retrain_state = reference_models["retrain_state"] if is_server else None
        if is_server:
            phase_seconds.update(reference_models["training_wall_seconds"])
            phase_seconds["reference_loading"] = preparation[
                "reference_loading_seconds"
            ]
    else:
        initial_state = initial_state if is_server else None
        phase_start = time.perf_counter()
        original_state, original_history = run_federated_phase(
            comm,
            config,
            device,
            initial_state,
            set(range(num_clients)),
            client_data,
            "original",
            args.seed,
            semantic_classifier=semantic_classifier,
            evaluation_data=data_state["test_data"],
            evaluation_interval=int(config["diagnostics"]["evaluation_interval"]),
            evaluation_batch_size=int(config["evaluation"]["batch_size"]),
            round_reporter=round_reporter,
        )
        if is_server:
            phase_seconds["original_training"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        retrain_state, retrain_history = run_federated_phase(
            comm,
            config,
            device,
            initial_state,
            retained_client_ids,
            client_data,
            retrain_name,
            args.seed,
            semantic_classifier=semantic_classifier,
            evaluation_data=data_state["test_data"],
            evaluation_interval=int(config["diagnostics"]["evaluation_interval"]),
            evaluation_batch_size=int(config["evaluation"]["batch_size"]),
            round_reporter=round_reporter,
        )
        if is_server:
            phase_seconds["retrain_without_target"] = (
                time.perf_counter() - phase_start
            )

    fixed_original_state = comm.bcast(
        original_state if is_server else None, root=0
    )
    phase_start = time.perf_counter()
    joint_state, _, joint_metrics, _ = run_target_joint_unlearning(
        comm,
        config,
        device,
        original_state if is_server else None,
        fixed_original_state,
        client_data,
        args.seed,
        cycle_id=0,
        epochs=int(unlearning["forgetting_epochs"]),
        extract_buffer=False,
        buffer_source="refresh",
        optimizer_state=None,
        semantic_classifier=semantic_classifier,
    )
    if is_server:
        phase_seconds["target_client_joint_unlearning"] = (
            time.perf_counter() - phase_start
        )

    variants = [primary_variant]
    if bool(unlearning.get("run_repair_ablations", True)):
        variants = ["normal_repair", "output_kd", "feature_kd"]
    variant_states = {}
    variant_metrics = {}
    variant_checkpoint_states = {}
    for variant in variants:
        phase_start = time.perf_counter()
        repaired_state, repair_metrics, repair_checkpoint_states = (
            run_fscuv4_retained_repair(
                comm,
                config,
                device,
                joint_state if is_server else None,
                fixed_original_state,
                client_data,
                args.seed,
                variant,
                semantic_classifier=semantic_classifier,
                evaluation_data=data_state["test_data"],
                evaluation_batch_size=int(config["evaluation"]["batch_size"]),
                round_reporter=round_reporter,
            )
        )
        if is_server:
            phase_seconds[f"retained_repair_{variant}"] = (
                time.perf_counter() - phase_start
            )
            repair_metrics["wall_seconds"] = phase_seconds[
                f"retained_repair_{variant}"
            ]
            variant_states[variant] = repaired_state
            variant_metrics[variant] = repair_metrics
            variant_checkpoint_states[variant] = repair_checkpoint_states

    if is_server:
        primary_seconds = phase_seconds[f"retained_repair_{primary_variant}"]
        online_seconds = (
            phase_seconds["target_client_joint_unlearning"] + primary_seconds
        )
        phase_seconds["unlearning_path_total"] = online_seconds
        phase_seconds["coevaluated_ablation_wall_seconds"] = sum(
            phase_seconds[f"retained_repair_{variant}"]
            for variant in variants
            if variant != primary_variant
        )

        artifacts.save_checkpoint("client_joint_unlearning", joint_state)
        checkpoint_names = {
            "normal_repair": "FSCUv4_NormalRepair",
            "output_kd": "FSCUv4_OutputKD",
            "feature_kd": "FSCUv4",
        }
        for variant, state in variant_states.items():
            name = method_label if variant == primary_variant else checkpoint_names[variant]
            artifacts.save_checkpoint(name, state)
        for round_number, state in variant_checkpoint_states[
            primary_variant
        ].items():
            if int(round_number) == int(unlearning["distributed_repair_rounds"]):
                continue
            artifacts.save_checkpoint(
                f"{method_label}_R{int(round_number):03d}", state
            )
        if not reuse_reference:
            artifacts.save_checkpoint("original", original_state)
            artifacts.save_checkpoint(retrain_name, retrain_state)
            artifacts.write_json(
                "reference_manifest.json",
                build_fscu_reference_manifest(
                    run_dir,
                    config,
                    split_manifest,
                    initial_state,
                    args.seed,
                    phase_seconds,
                ),
            )

        reference_provenance = {
            "requested_reference": (
                str(reference_run) if reference_run is not None else None
            ),
            "reused": reuse_reference,
            "source_run": (
                reference_models["reference_run"] if reuse_reference else None
            ),
            "scenario_fingerprint_sha256": (
                reference_models["scenario_fingerprint_sha256"]
                if reuse_reference
                else None
            ),
            "checkpoint_sha256": (
                reference_models["checkpoint_sha256"] if reuse_reference else None
            ),
            "reference_manifest_verified": (
                reference_models["reference_manifest_verified"]
                if reuse_reference
                else False
            ),
            "rejection_reason": preparation["reference_rejection_reason"],
        }
        artifacts.write_json("reference_source.json", reference_provenance)

        model_bytes = model_state_nbytes(original_state)
        repair_rounds = int(unlearning["distributed_repair_rounds"])
        retained_count = len(retained_client_ids)
        retrain_seconds = phase_seconds.get("retrain_without_target")
        efficiency = {
            "phase_wall_seconds": phase_seconds,
            "reference_models_reused": reuse_reference,
            "reference_run": reference_provenance["source_run"],
            "model_state_bytes": model_bytes,
            "proposed_online_wall_seconds": online_seconds,
            "speedup_vs_retraining": (
                retrain_seconds / online_seconds
                if retrain_seconds is not None
                else None
            ),
            "proposed_algorithmic_communication_bytes": int(
                2 * model_bytes
                + retained_count * model_bytes
                + 2 * repair_rounds * retained_count * model_bytes
            ),
            "teacher_distribution_bytes": (
                0
                if primary_variant == "normal_repair"
                else int(retained_count * model_bytes)
            ),
            "target_joint_unlearning_model_bytes": int(2 * model_bytes),
            "retained_repair_model_bytes": int(
                2 * repair_rounds * retained_count * model_bytes
            ),
            "proposed_target_update_steps": int(joint_metrics["steps"]),
            "proposed_retained_client_update_steps": int(
                variant_metrics[primary_variant]["client_update_steps"]
            ),
            "public_data_examples_used_for_repair": 0,
            "target_data_examples_used_for_repair": 0,
            "primary_repair_variant": primary_variant,
            "repair_checkpoint_wall_seconds": {
                (
                    method_label
                    if int(row["round"])
                    == int(unlearning["distributed_repair_rounds"])
                    else f"{method_label}_R{int(row['round']):03d}"
                ): (
                    phase_seconds["target_client_joint_unlearning"]
                    + float(row["cumulative_wall_seconds"])
                )
                for row in variant_metrics[primary_variant]["history"]
                if int(row["round"])
                in {
                    int(value)
                    for value in unlearning.get(
                        "repair_checkpoint_rounds",
                        [unlearning["distributed_repair_rounds"]],
                    )
                }
            },
        }
        artifacts.write_json("metrics/efficiency.json", efficiency)
        artifacts.write_json(
            "metrics/training_summary.json",
            {
                "reference_model": retrain_name,
                "reference_models_reused": reuse_reference,
                "reference_run": reference_provenance["source_run"],
                "training_history": original_history + retrain_history,
                "target_client_joint_unlearning": joint_metrics,
                "retained_only_repair": variant_metrics,
                "primary_repair_variant": primary_variant,
                "controlled_ablation_start": "same_client_joint_unlearning_state",
                "feature_definition": {
                    "encoder": "after_first_encoder_prelu",
                    "decoder": "after_last_decoder_prelu_before_rgb_output",
                    "loss": "mean_one_minus_cosine",
                    "channel": "paired_awgn",
                    "latent": "normalized_posterior_mean",
                },
            },
        )
        if original_history:
            artifacts.write_csv(
                "metrics/original_training_history.csv", original_history
            )
            save_federated_training_curve_figures(
                original_history,
                artifacts.path("figures"),
                config["split"]["type"],
                phase_name="original",
            )
        if retrain_history:
            artifacts.write_csv(
                "metrics/retrain_training_history.csv", retrain_history
            )
            save_federated_training_curve_figures(
                retrain_history,
                artifacts.path("figures"),
                config["split"]["type"],
                phase_name="retrain_without_target",
            )
        artifacts.write_csv(
            "metrics/joint_unlearning_history.csv", joint_metrics["history"]
        )
        artifacts.write_json(
            "metrics/retained_repair_history.json", variant_metrics
        )
        save_fscuv4_optimization_figures(
            joint_metrics, variant_metrics, artifacts.path("figures")
        )
        artifacts.append_log("fscuv4_training_and_unlearning_completed")

    comm.Barrier()
    completion = None
    if is_server:
        try:
            run_reconstruction_channel_audit(
                source_run=run_dir,
                output_root=None,
                seed=args.seed,
                device=args.audit_device,
                snr_db=float(config["evaluation"]["snr_db"]),
                batch_size=int(config["evaluation"]["batch_size"]),
                in_place=True,
            )
            metrics_path, headline_metrics = _read_fscu_headline_metrics(
                run_dir, algorithm=method_label
            )
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                headline_rows = list(csv.DictReader(handle))
            save_unlearning_headline_figure(
                headline_rows, run_dir / "figures", method_label
            )
            completion = {
                "error": None,
                "result": {
                    "algorithm": method_label,
                    "run_dir": str(run_dir),
                    "headline_metrics_path": str(metrics_path),
                    "headline_metrics": headline_metrics,
                    "messages": [
                        f"{method_label} finished: {run_dir}",
                        f"Headline metrics: {metrics_path}",
                    ],
                },
            }
            if logger:
                logger.info("[FSCUv4] completed run=%s", run_dir)
        except Exception as error:
            completion = {
                "error": f"{type(error).__name__}: {error}",
                "result": None,
            }
    completion = comm.bcast(completion, root=0)
    if completion["error"] is not None:
        raise RuntimeError(
            f"FSCUv4 automatic evaluation failed: {completion['error']}"
        )
    return completion["result"] if is_server else None


def train_federated_training_curves(comm, data_state, args, logger):
    """Dispatch one reconstruction-only or classifier-only FL training run."""

    from src.utils.artifacts import create_run_artifacts
    from src.utils.fscu_runtime import (
        make_initial_federated_classifier_state,
        make_initial_fscu_state,
        run_federated_classification_phase,
        run_federated_phase,
        select_fscu_device,
    )
    from src.utils.plotting import (
        save_federated_classification_curve_figures,
        save_federated_training_curve_figures,
    )
    from src.utils.reproducibility import (
        collect_environment,
        derive_seed,
        set_global_seed,
    )

    rank = comm.Get_rank()
    is_server = rank == 0
    config = data_state["config"]
    num_clients = int(config["federation"]["n_clients"])
    task = config["task"]
    split_type = config["split"]["type"]

    output_root = _validate_fscu_output_root(args.output_root)
    device = select_fscu_device(config)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_global_seed(
        derive_seed(args.seed, f"federated-rank-{rank}"),
        bool(config.get("deterministic", True)),
    )

    artifacts = None
    setup = None
    if is_server:
        try:
            artifacts = create_run_artifacts(output_root, args.seed)
            artifacts.write_json("config.json", config)
            artifacts.write_json(
                "environment.json",
                collect_environment(device, Path(__file__).resolve().parents[2]),
            )
            artifacts.write_json("split_manifest.json", data_state["split_manifest"])
            artifacts.write_json(
                "run_metadata.json",
                {
                    "global_seed": int(args.seed),
                    "split_seed": int(data_state["split_seed"]),
                    "split_type": split_type,
                    "task": config["task"],
                    "model_kind": config["model"]["kind"],
                    "evaluation_interval": int(
                        config["diagnostics"]["evaluation_interval"]
                    ),
                },
            )
            setup = {"error": None, "run_dir": str(artifacts.run_dir)}
        except Exception as error:
            setup = {
                "error": f"{type(error).__name__}: {error}",
                "run_dir": None,
            }
    setup = comm.bcast(setup, root=0)
    if setup["error"] is not None:
        raise RuntimeError(f"training-curve setup failed: {setup['error']}")
    run_dir = Path(setup["run_dir"])
    round_reporter = _make_round_reporter(artifacts) if is_server else None
    if task == "classification":
        initial_state = (
            make_initial_federated_classifier_state(config, args.seed)
            if is_server
            else None
        )
        if is_server:
            artifacts.save_checkpoint("initial", initial_state)
        phase_start = time.perf_counter()
        original_state, history = run_federated_classification_phase(
            comm,
            config,
            device,
            initial_state,
            set(range(num_clients)),
            data_state["client_data"],
            "original",
            args.seed,
            evaluation_data=data_state["test_data"],
            evaluation_interval=int(config["diagnostics"]["evaluation_interval"]),
            evaluation_batch_size=int(config["evaluation"]["batch_size"]),
            round_reporter=round_reporter,
        )
    else:
        initial_state = make_initial_fscu_state(config, args.seed) if is_server else None
        if is_server:
            artifacts.save_checkpoint("initial", initial_state)
        phase_start = time.perf_counter()
        original_state, history = run_federated_phase(
            comm,
            config,
            device,
            initial_state,
            set(range(num_clients)),
            data_state["client_data"],
            "original",
            args.seed,
            evaluation_data=data_state["test_data"],
            evaluation_interval=int(config["diagnostics"]["evaluation_interval"]),
            evaluation_batch_size=int(config["evaluation"]["batch_size"]),
            round_reporter=round_reporter,
        )

    if is_server:
        try:
            wall_seconds = time.perf_counter() - phase_start
            artifacts.save_checkpoint("original", original_state)
            artifacts.write_csv("metrics/training_history.csv", history)
            performance_field = (
                "test_accuracy"
                if task == "classification"
                else "test_psnr_db"
            )
            performance_history = [
                row for row in history if row.get(performance_field) is not None
            ]
            artifacts.write_csv(
                "metrics/performance_history.csv", performance_history
            )
            final_performance = performance_history[-1]
            artifacts.write_json(
                "metrics/training_curve_summary.json",
                {
                    "task": task,
                    "split_type": split_type,
                    "model_kind": config["model"]["kind"],
                    "training_wall_seconds": wall_seconds,
                    "rounds": int(config["federation"]["rounds"]),
                    "evaluation_interval": int(
                        config["diagnostics"]["evaluation_interval"]
                    ),
                    "performance_field": performance_field,
                    "final_performance": final_performance,
                },
            )
            if task == "classification":
                save_federated_classification_curve_figures(
                    history,
                    artifacts.path("figures"),
                    config["split"]["type"],
                )
            else:
                save_federated_training_curve_figures(
                    history,
                    artifacts.path("figures"),
                    config["split"]["type"],
                )
            artifacts.append_log(
                f"{split_type}_{task}_training_curves_completed"
            )
            completion = {
                "error": None,
                "result": {
                    "algorithm": "training_curves",
                    "task": task,
                    "split_type": split_type,
                    "run_dir": str(run_dir),
                    "final_performance": final_performance,
                    "messages": [
                        f"{split_type} {task} training curves finished: {run_dir}",
                    ],
                },
            }
            if logger:
                logger.info(
                    "[training_curves:%s:%s] completed run=%s",
                    task,
                    split_type,
                    run_dir,
                )
        except Exception as error:
            completion = {
                "error": f"{type(error).__name__}: {error}",
                "result": None,
            }
    else:
        completion = None
    completion = comm.bcast(completion, root=0)
    if completion["error"] is not None:
        raise RuntimeError(f"training-curve run failed: {completion['error']}")
    return completion["result"] if is_server else None


def train_remaining_classifier(comm, data_state, args):
    """Fresh classifier-only MPI FedAvg; server never receives client images."""
    import json
    import hashlib
    import numpy as np
    from src.data.data import (
        WORKSPACE_ROOT, load_cifar10_test_dataset, select_cifar10_client_data,
        select_class_balanced_subset,
    )
    from src.utils.artifacts import create_run_artifacts
    from src.utils.federated import (
        LocalClassificationConfig, average_model_states, copy_model_state,
        train_local_classification,
    )
    from src.utils.semantic_classifier import (
        make_semantic_classifier, evaluate_frozen_classifier,
        classifier_state_sha256, _file_sha256,
    )
    from src.utils.reproducibility import derive_seed, set_global_seed, collect_environment
    from src.utils.plotting import save_remaining_classifier_figures

    rank = comm.Get_rank()
    config = data_state["config"]
    device = torch.device(config["device"])
    initialization_seed = derive_seed(args.seed, "remaining-observer-initial")
    set_global_seed(initialization_seed, deterministic=True)
    # No codec or previously trained classifier is constructed or loaded here.
    model = make_semantic_classifier({"semantic_classifier": config["model"]}, device=device, frozen=False)
    initial_state = copy_model_state(model)
    artifacts, preparation_error = None, None
    if rank == 0:
        try:
            output_root = _validate_fscu_output_root(args.output_root)
            if output_root.exists():
                raise ValueError("remaining_classifier requires a newly named output root")
            artifacts = create_run_artifacts(output_root, args.seed)
            artifacts.write_json("config.json", config)
            artifacts.write_json("environment.json", collect_environment(device, WORKSPACE_ROOT))
            artifacts.save_checkpoint("initial", initial_state)
            artifacts.write_json("split_manifest.json", {
                "source_scenario": data_state["source_scenario"],
                "source_fingerprint": data_state["source_fingerprint"],
                "clients": data_state["client_splits"],
            })
        except Exception as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
    preparation_error = comm.bcast(preparation_error, root=0)
    if preparation_error:
        raise RuntimeError(preparation_error)

    history, best_state = [], None
    best_ce, progress_ce, stale, selected_round = float("inf"), float("inf"), 0, 0
    started = time.perf_counter()
    for round_index in range(1, config["max_rounds"] + 1):
        state = comm.bcast(copy_model_state(model) if rank == 0 else None, root=0)
        model.load_state_dict(state)
        local_payload, local_error = None, None
        rate = config["training"]["learning_rate"]
        if config.get("learning_rate_decay_round") is not None and round_index >= config["learning_rate_decay_round"]:
            rate = config["learning_rate_after_decay"]
        try:
            if rank > 0:
                local_config = LocalClassificationConfig(
                    **{**config["training"], "learning_rate": rate}, device=str(device),
                )
                local_state, local_metrics = train_local_classification(
                    model, data_state["train_data"], local_config,
                    derive_seed(args.seed, f"remaining-observer-round-{round_index}-client-{data_state['client_id']}"),
                )
                local_payload = {"state": local_state, "metrics": local_metrics}
        except Exception as exc:
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        errors = [error for error in comm.allgather(local_error) if error]
        if errors:
            raise RuntimeError("; ".join(errors))
        updates = comm.gather(local_payload, root=0)
        aggregation_error = None
        if rank == 0:
            try:
                updates = [value for value in updates if value is not None]
                counts = [value["metrics"]["examples"] for value in updates]
                model.load_state_dict(average_model_states([value["state"] for value in updates], counts))
                row = {
                    "round": round_index, "learning_rate": rate,
                    "train_cross_entropy": sum(value["metrics"]["cross_entropy"] * count for value, count in zip(updates, counts)) / sum(counts),
                    "train_accuracy": sum(value["metrics"]["accuracy"] * count for value, count in zip(updates, counts)) / sum(counts),
                    "validation_cross_entropy": None, "validation_accuracy": None,
                    "train_examples": sum(counts), "validation_examples": None,
                }
            except Exception as exc:
                aggregation_error = f"{type(exc).__name__}: {exc}"
        aggregation_error = comm.bcast(aggregation_error, root=0)
        if aggregation_error:
            raise RuntimeError(aggregation_error)
        evaluate = round_index == 1 or round_index % config["evaluation_interval"] == 0 or round_index == config["max_rounds"]
        if evaluate:
            state = comm.bcast(copy_model_state(model) if rank == 0 else None, root=0)
            local_metrics, local_error = None, None
            try:
                if rank > 0:
                    model.load_state_dict(state)
                    local_metrics = evaluate_frozen_classifier(
                        model, data_state["validation_data"], config["training"]["batch_size"], device,
                    )
            except Exception as exc:
                local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
            errors = [error for error in comm.allgather(local_error) if error]
            if errors:
                raise RuntimeError("; ".join(errors))
            validations = comm.gather(local_metrics, root=0)
        decision = None
        if rank == 0:
            try:
                if evaluate:
                    validations = [value for value in validations if value is not None]
                    n = sum(value["examples"] for value in validations)
                    ce = sum(value["loss"] * value["examples"] for value in validations) / n
                    accuracy = sum(value["accuracy"] * value["examples"] for value in validations) / n
                    if not np.isfinite(ce):
                        raise ValueError("nonfinite observer validation CE")
                    row.update(validation_cross_entropy=ce, validation_accuracy=accuracy, validation_examples=n)
                    if ce < best_ce:
                        best_ce, selected_round = ce, round_index
                        best_state = copy_model_state(model)
                    if ce < progress_ce - config["min_delta"]:
                        progress_ce, stale = ce, 0
                    else:
                        stale += 1
                history.append(row)
                message = f"[remaining_classifier] round={round_index} train_ce={row['train_cross_entropy']:.6f} train_accuracy={row['train_accuracy']:.6f}"
                if evaluate:
                    message += f" validation_ce={ce:.6f} validation_accuracy={accuracy:.6f}"
                print(message, flush=True)
                artifacts.append_log(message)
                artifacts.write_csv("metrics/observer_training.csv", history)
                decision = {"error": None, "stop": stale >= config["patience_evaluations"]}
            except Exception as exc:
                decision = {"error": f"{type(exc).__name__}: {exc}", "stop": True}
        decision = comm.bcast(decision, root=0)
        if decision["error"]:
            raise RuntimeError(decision["error"])
        if decision["stop"]:
            break

    completion = None
    if rank == 0:
        try:
            training_seconds = time.perf_counter() - started
            checkpoint_path = artifacts.save_checkpoint("observer", best_state)
            model.load_state_dict(best_state)
            model.freeze()
            # Official test is first read only after validation-based selection.
            test_dataset = load_cifar10_test_dataset(data_state["source_config"]["dataset"], WORKSPACE_ROOT)
            test_indices = np.arange(len(test_dataset), dtype=np.int64)
            if args.smoke:
                _, _, test_indices = select_class_balanced_subset(test_dataset, 2, args.seed)
            test_metrics = evaluate_frozen_classifier(
                model, select_cifar10_client_data(test_dataset, test_indices),
                config["training"]["batch_size"], device,
            )
            protocol = {
                "source_fingerprint": data_state["source_fingerprint"],
                "client_splits": data_state["client_splits"],
                "config": config, "seed": int(args.seed), "smoke_test": bool(args.smoke),
            }
            metadata = {
                "kind": "remaining_client_classifier_v1", "completed": True,
                "source_run": data_state["source_run"],
                "source_scenario": data_state["source_scenario"],
                "protocol": protocol,
                "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "architecture_config": {"architecture": config["model"]["architecture"], **vars(model.config)},
                "initial_state_sha256": classifier_state_sha256(initial_state),
                "initialization_seed": initialization_seed,
                "state_sha256": classifier_state_sha256(best_state),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "selected_round": selected_round, "final_round": history[-1]["round"],
                "selection_rule": "minimum sample-weighted local held-out CE; official test never selects a checkpoint",
                "stop_reason": "validation_patience" if decision["stop"] else "max_rounds",
                "raw_cifar10_test_accuracy": test_metrics["accuracy"],
                "raw_cifar10_test_loss": test_metrics["loss"],
                "raw_cifar10_test_examples": test_metrics["examples"],
                "test_indices": test_indices.tolist(),
                "training_wall_seconds": training_seconds,
                "runtime_scope": "observer FL + local validation; separate from codec retraining/unlearning",
                "role": "shared_frozen_remaining_client_observer",
                "initialization": "random_from_scratch_no_original_classifier",
                "aggregation": "sample_weighted_FedAvg_fresh_local_Adam_each_round",
                "codec_gradient": False, "raw_client_data_transmitted": False,
            }
            artifacts.write_json("metrics/observer_summary.json", metadata)
            save_remaining_classifier_figures(history, data_state["client_splits"], selected_round, artifacts.path("figures"))
            # Completion marker is written only after metrics and required figures exist.
            artifacts.write_json("observer_manifest.json", metadata)
            artifacts.append_log(f"completed selected_round={selected_round} final_round={history[-1]['round']}")
            completion = {"error": None, "result": {"run_dir": str(artifacts.run_dir), "messages": [f"Remaining-client classifier finished: {artifacts.run_dir}"]}}
        except Exception as exc:
            completion = {"error": f"{type(exc).__name__}: {exc}", "result": None}
    completion = comm.bcast(completion, root=0)
    if completion["error"]:
        raise RuntimeError(completion["error"])
    return completion["result"] if rank == 0 else None


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if getattr(args, "experiment", "scu") == "federated_deletion":
            self.args.algorithm = "FSCU"

    def prepare_data(self, comm):
        """Select a data preparation strategy while keeping main.py common."""
        if getattr(self.args, "experiment", "scu") == "remaining_classifier":
            from src.data.data import prepare_remaining_classifier_data

            return prepare_remaining_classifier_data(comm, self.args)
        if getattr(self.args, "experiment", "scu") in AUXILIARY_EXPERIMENTS:
            return None

        if getattr(self.args, "experiment", "scu") == "training_curves":
            from src.data.data import fscu_split_seed, prepare_fscu_data
            from src.utils.config import load_training_curve_config

            config = load_training_curve_config(self.args.config)
            expected_size = int(config["federation"]["n_clients"]) + 1
            if comm.Get_size() != expected_size:
                raise ValueError(
                    "training_curves MPI size does not match federation.n_clients"
                )
            split_seed = fscu_split_seed(self.args.seed, config)
            data_state = prepare_fscu_data(comm, config, split_seed)
            data_state["config"] = config
            return data_state

        if self.args.algorithm == "SCU":
            from src.data.data import prepare_scu_data

            return prepare_scu_data(comm, self.args)

        if self.args.algorithm in ("FSCU", "FSCUv4"):
            from src.data.data import fscu_split_seed, prepare_fscu_data
            from src.utils.config import (
                load_fscu_config,
                load_fscuv4_config,
            )

            if self.args.algorithm == "FSCU":
                config = load_fscu_config(self.args.config)
            else:
                config = load_fscuv4_config(self.args.config)
            expected_size = int(config["federation"]["n_clients"]) + 1
            if comm.Get_size() != expected_size:
                raise ValueError(
                    f"{self.args.algorithm} MPI size does not match "
                    "federation.n_clients"
                )
            split_seed = fscu_split_seed(self.args.seed, config)
            data_state = prepare_fscu_data(comm, config, split_seed)
            data_state["config"] = config
            return data_state

        raise ValueError(f"Unknown algorithm: {self.args.algorithm}")

    def train(self, data_state, comm, logger=None, args=None):
        args = self.args if args is None else args

        if getattr(args, "experiment", "scu") == "remaining_classifier":
            return train_remaining_classifier(comm, data_state, args)

        if getattr(args, "experiment", "scu") in AUXILIARY_EXPERIMENTS:
            from src.experiments.auxiliary import run_auxiliary_experiment

            run_auxiliary_experiment(args)
            return {"algorithm": "auxiliary", "messages": []}
        if data_state is None:
            raise ValueError("training requires prepared data")
        if getattr(args, "experiment", "scu") == "training_curves":
            return train_federated_training_curves(
                comm, data_state, args, logger
            )
        if args.algorithm == "SCU":
            return train_scu(comm, data_state, args, logger)
        if args.algorithm == "FSCU":
            return train_fscu(comm, data_state, args, logger)
        if args.algorithm == "FSCUv4":
            return train_fscuv4(comm, data_state, args, logger)
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
