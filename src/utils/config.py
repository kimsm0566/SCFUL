"""Oracle Behavior 실험 설정의 loading, smoke override, device 선택."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_ARCHITECTURE,
    SUPPORTED_DEEPJSCC_ARCHITECTURES,
    variational_deepjscc_config_for_architecture,
)
from src.models.semantic_classifier import (
    CIFAR10_CLASSIFIER_ARCHITECTURE,
    Cifar10ClassifierConfig,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE_CONFIG_PATH = WORKSPACE_ROOT / "src/configs/controlled_cifar10.json"


def load_remaining_classifier_config(path, smoke=False):
    """Independent observer settings; no codec, teacher, or warm-start options."""
    from src.utils.federated import LocalClassificationConfig

    config = load_json_config(path)
    allowed = {"experiment", "device", "model", "training", "max_rounds",
               "evaluation_interval", "patience_evaluations", "min_delta",
               "validation_fraction", "learning_rate_decay_round", "learning_rate_after_decay"}
    if set(config) - allowed or config.get("experiment") != "remaining_classifier":
        raise ValueError("invalid remaining_classifier configuration (warm starts are not allowed)")
    for name in ("max_rounds", "evaluation_interval", "patience_evaluations"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < float(config["validation_fraction"]) < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not math.isfinite(float(config["min_delta"])) or config["min_delta"] < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    if config["model"].get("architecture") != CIFAR10_CLASSIFIER_ARCHITECTURE:
        raise ValueError("unsupported remaining classifier architecture")
    Cifar10ClassifierConfig(**{key: value for key, value in config["model"].items() if key != "architecture"})
    local = LocalClassificationConfig(**config["training"], device=config["device"])
    if local.preserve_fixed_backdoor_trigger:
        raise ValueError("remaining observer training must use clean images")
    for key in ("learning_rate", "weight_decay", "augmentation_horizontal_flip_probability"):
        if not math.isfinite(float(config["training"].get(key, getattr(local, key)))):
            raise ValueError(f"training.{key} must be finite")
    decay = config.get("learning_rate_decay_round")
    after = config.get("learning_rate_after_decay")
    if (decay is None) != (after is None):
        raise ValueError("learning-rate decay round and rate must be specified together")
    if decay is not None and (not isinstance(decay, int) or decay <= 0 or not math.isfinite(after) or after <= 0):
        raise ValueError("invalid learning-rate decay")
    if smoke:
        config["max_rounds"] = 2
        config["evaluation_interval"] = 1
        config["training"]["local_epochs"] = 1
        config["training"]["batch_size"] = 20
        config["model"].update({"base_channels": 8, "group_count": 4})
    return config


def _normalize_variational_deepjscc_codec(codec):
    """Fill only defaults belonging to the selected checkpoint architecture."""

    architecture = codec.get("architecture", DEEPJSCC_ARCHITECTURE)
    if architecture not in SUPPORTED_DEEPJSCC_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_DEEPJSCC_ARCHITECTURES)
        raise ValueError(
            f"unsupported codec.architecture: {architecture}; supported: {supported}"
        )
    defaults = variational_deepjscc_config_for_architecture(architecture)
    allowed_codec_fields = set(vars(defaults)) | {"architecture"}
    unknown_codec_fields = sorted(set(codec) - allowed_codec_fields)
    if unknown_codec_fields:
        raise ValueError(
            "codec contains fields from a removed model: "
            f"{unknown_codec_fields}"
        )
    normalized = {
        "architecture": architecture,
        **vars(defaults),
        **codec,
    }

    for name in (
        "codec_format_version",
        "logvar_parameterization",
        "kl_reduction",
    ):
        expected = getattr(defaults, name)
        if normalized[name] != expected:
            raise ValueError(
                f"codec.{name} must equal {expected!r} for "
                f"codec.architecture={architecture}"
            )

    latent_dim = normalized["latent_dim"]
    latent_channels = normalized["latent_channels"]
    for name, value in (
        ("latent_dim", latent_dim),
        ("latent_channels", latent_channels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"codec.{name} must be a positive integer")
    expected_latent_dim = 8 * 8 * latent_channels
    if latent_dim != expected_latent_dim:
        raise ValueError(
            "codec.latent_dim must equal 64 * codec.latent_channels "
            f"({expected_latent_dim})"
        )

    if architecture == DEEPJSCC_ARCHITECTURE:
        if (
            normalized["logvar_min"] is not None
            or normalized["logvar_max"] is not None
        ):
            raise ValueError(
                "codec.logvar_min and codec.logvar_max must both be null for "
                f"codec.architecture={architecture}"
            )
    else:
        if (
            normalized["logvar_min"] is None
            or normalized["logvar_max"] is None
        ):
            raise ValueError(
                "codec.logvar_min and codec.logvar_max must both be finite for "
                f"codec.architecture={architecture}"
            )
        logvar_min = float(normalized["logvar_min"])
        logvar_max = float(normalized["logvar_max"])
        initial_logvar = float(normalized["initial_logvar"])
        if not all(math.isfinite(value) for value in (logvar_min, logvar_max)):
            raise ValueError("codec.logvar_min and codec.logvar_max must be finite")
        if not logvar_min < initial_logvar < logvar_max:
            raise ValueError(
                "codec.initial_logvar must lie strictly between codec.logvar_min "
                "and codec.logvar_max"
            )
    return normalized


def _configure_reconstruction_training(training, codec, federation_rounds):
    """Validate optimizer controls shared by full FSCU and curve training."""

    default_gradient_clip = (
        5.0
        if codec["architecture"] == LEGACY_DEEPJSCC_ARCHITECTURE
        else None
    )
    training.setdefault("gradient_clip_norm", default_gradient_clip)
    gradient_clip_norm = training["gradient_clip_norm"]
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive or null")
        training["gradient_clip_norm"] = gradient_clip_norm

    decay_round = training.get("learning_rate_decay_round")
    learning_rate_after_decay = training.get("learning_rate_after_decay")
    if (decay_round is None) != (learning_rate_after_decay is None):
        raise ValueError(
            "training.learning_rate_decay_round and "
            "training.learning_rate_after_decay must be set together"
        )
    training.setdefault("learning_rate_decay_round", None)
    training.setdefault("learning_rate_after_decay", None)
    if decay_round is not None:
        if isinstance(decay_round, bool) or int(decay_round) != decay_round:
            raise ValueError("training.learning_rate_decay_round must be an integer")
        decay_round = int(decay_round)
        if decay_round <= 0:
            raise ValueError("training.learning_rate_decay_round must be positive")
        if decay_round > int(federation_rounds):
            raise ValueError(
                "training.learning_rate_decay_round must not exceed "
                "federation.rounds"
            )
        learning_rate_after_decay = float(learning_rate_after_decay)
        if (
            not math.isfinite(learning_rate_after_decay)
            or learning_rate_after_decay <= 0
        ):
            raise ValueError(
                "training.learning_rate_after_decay must be positive"
            )
        training["learning_rate_decay_round"] = decay_round
        training["learning_rate_after_decay"] = learning_rate_after_decay


def _configure_codec_evaluation(evaluation):
    """Set a reproducible posterior/channel evaluation protocol."""

    evaluation.setdefault("posterior_mode", "mean")
    evaluation.setdefault("monte_carlo_repetitions", 1)
    if evaluation["posterior_mode"] not in ("mean", "sample"):
        raise ValueError("evaluation.posterior_mode must be mean or sample")
    repetitions = evaluation["monte_carlo_repetitions"]
    if isinstance(repetitions, bool) or int(repetitions) != repetitions:
        raise ValueError("evaluation.monte_carlo_repetitions must be an integer")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("evaluation.monte_carlo_repetitions must be positive")
    evaluation["monte_carlo_repetitions"] = repetitions


def _validate_optional_bandwidth(config):
    """Validate explicit real/complex channel-use accounting when supplied."""

    if "bandwidth" not in config:
        return
    bandwidth = config["bandwidth"]
    required = {
        "source_real_values",
        "transmitted_real_values",
        "complex_channel_uses",
        "complex_channel_bandwidth_ratio",
    }
    missing = sorted(required - set(bandwidth))
    unknown = sorted(set(bandwidth) - required)
    if missing:
        raise ValueError(f"bandwidth is missing fields: {missing}")
    if unknown:
        raise ValueError(f"bandwidth has unknown fields: {unknown}")
    source_values = int(bandwidth["source_real_values"])
    transmitted_values = int(bandwidth["transmitted_real_values"])
    complex_uses = int(bandwidth["complex_channel_uses"])
    ratio = float(bandwidth["complex_channel_bandwidth_ratio"])
    latent_dim = int(config["codec"]["latent_dim"])
    if source_values != 3 * 32 * 32:
        raise ValueError("bandwidth.source_real_values must equal 3072 for CIFAR-10")
    if transmitted_values != latent_dim:
        raise ValueError(
            "bandwidth.transmitted_real_values must equal codec.latent_dim"
        )
    if transmitted_values % 2 != 0 or complex_uses != transmitted_values // 2:
        raise ValueError(
            "bandwidth.complex_channel_uses must equal half of "
            "transmitted_real_values"
        )
    expected_ratio = complex_uses / source_values
    if not math.isfinite(ratio) or not math.isclose(
        ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            "bandwidth.complex_channel_bandwidth_ratio must equal "
            "complex_channel_uses/source_real_values"
        )


def load_json_config(path: Path) -> dict[str, Any]:
    """JSON config를 읽고 파일/JSON 오류에 config path를 포함한다."""

    if not path.is_file():
        raise FileNotFoundError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config at {path}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be an object: {path}")
    return payload


def build_effective_config(
    config: Mapping[str, Any], *, smoke_test: bool
) -> dict[str, Any]:
    """원본 config를 변경하지 않고 optional CPU smoke override를 적용한다."""

    result = copy.deepcopy(dict(config))
    if not smoke_test:
        return result

    result["device"] = "cpu"
    if str(result["dataset"]["name"]) != "CIFAR10":
        raise ValueError("smoke behavior experiments now support CIFAR-10 only")
    result["dataset"]["smoke_train_per_class"] = 20
    result["dataset"]["smoke_test_per_class"] = 12
    result["codec"].update({"latent_dim": 64, "latent_channels": 1})
    result["split"].update({"shared_per_label": 4})
    result["audit"].update({"target_per_group": 4, "retained_total": 18})
    result["training"].update({"epochs": 1, "batch_size": 16, "num_workers": 0})
    result["scu"].update({"epochs": 1, "batch_size": 16, "num_workers": 0})
    result["evaluation"].update(
        {
            "batch_size": 32,
            "awgn_repetitions": 2,
            "bootstrap_resamples": 20,
            "figure_probes_per_group": 2,
        }
    )
    result["client_posterior"].update(
        {
            "batch_size": 32,
            "mmd_sample_limit": 32,
            "bootstrap_resamples": 8,
            "bootstrap_sample_size": 16,
        }
    )
    result["smoke_test"] = True
    return result


def select_device(config: Mapping[str, Any]) -> torch.device:
    """`auto`, CPU, explicit CUDA device를 명시적으로 처리한다."""

    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_federated_codec_config(path, algorithm):
    """Load fields shared by FSCU versions without mixing method-specific options."""
    if path is None:
        raise ValueError(f"{algorithm} requires --config")
    config = load_json_config(Path(path))
    required_sections = (
        "dataset",
        "codec",
        "federation",
        "split",
        "training",
        "unlearning",
        "public_data",
        "evaluation",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"{algorithm} config is missing sections: {missing}")
    if config["dataset"].get("name") != "CIFAR10":
        raise ValueError(f"{algorithm} supports CIFAR-10 only")

    federation = config["federation"]
    if int(federation.get("n_clients", 0)) < 2:
        raise ValueError("federation.n_clients must be at least 2")
    for name in ("samples_per_client", "rounds", "local_epochs"):
        if float(federation.get(name, 0)) <= 0:
            raise ValueError(f"federation.{name} must be positive")
    federation.setdefault("persist_client_optimizer_state", False)
    federation.setdefault("serialize_cuda_clients", False)
    federation.setdefault("cuda_client_group_size", 1)
    if "target_client_id" not in config["unlearning"]:
        raise ValueError(f"{algorithm} unlearning config is missing: ['target_client_id']")
    if not 0 <= int(config["unlearning"]["target_client_id"]) < int(
        federation["n_clients"]
    ):
        raise ValueError("unlearning.target_client_id is outside the client range")

    for section, names in (
        ("training", ("batch_size", "learning_rate", "snr_db")),
        ("public_data", ("samples_per_class",)),
        ("evaluation", ("batch_size", "snr_db")),
    ):
        for name in names:
            if float(config[section].get(name, 0)) <= 0:
                raise ValueError(f"{section}.{name} must be positive")

    training = config["training"]
    default_augmentation = (
        "none" if "backdoor" in config else "cifar10_crop_flip_v1"
    )
    training.setdefault("augmentation", default_augmentation)
    training.setdefault("augmentation_crop_padding", 4)
    training.setdefault("augmentation_horizontal_flip_probability", 0.5)
    if training["augmentation"] not in ("none", "cifar10_crop_flip_v1"):
        raise ValueError(
            "training.augmentation must be none or cifar10_crop_flip_v1"
        )
    if int(training["augmentation_crop_padding"]) < 0:
        raise ValueError(
            "training.augmentation_crop_padding must be non-negative"
        )
    flip_probability = float(
        training["augmentation_horizontal_flip_probability"]
    )
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "training.augmentation_horizontal_flip_probability must be in [0, 1]"
        )
    if "backdoor" in config and training["augmentation"] != "none":
        raise ValueError(
            "training augmentation must be none when backdoor is configured; "
            "crop/flip would change the fixed SCU trigger protocol"
        )

    split_type = config["split"].get("type", "equal_size_dirichlet")
    if split_type == "equal_size_dirichlet" and float(
        config["split"].get("alpha", 0)
    ) <= 0:
        raise ValueError("split.alpha must be positive")
    if split_type == "fedosd_balanced_pathological" and int(
        config["split"].get("classes_per_client", 0)
    ) <= 0:
        raise ValueError("split.classes_per_client must be positive")
    if split_type not in (
        "iid",
        "equal_size_dirichlet",
        "fedosd_balanced_pathological",
        "target_exclusive_label",
    ):
        raise ValueError(f"unsupported split.type: {split_type}")

    config["codec"] = _normalize_variational_deepjscc_codec(config["codec"])
    _configure_reconstruction_training(
        training, config["codec"], federation["rounds"]
    )
    _configure_codec_evaluation(config["evaluation"])
    _validate_optional_bandwidth(config)
    classifier_defaults = Cifar10ClassifierConfig()
    is_smoke = "smoke" in str(config.get("experiment_name", "")).lower()
    classifier = config.setdefault("semantic_classifier", {})
    classifier.setdefault(
        "architecture", CIFAR10_CLASSIFIER_ARCHITECTURE
    )
    classifier.setdefault(
        "base_channels", 8 if is_smoke else classifier_defaults.base_channels
    )
    classifier.setdefault(
        "group_count", 4 if is_smoke else classifier_defaults.group_count
    )
    classifier.setdefault("num_classes", classifier_defaults.num_classes)
    classifier.setdefault("channel_mean", list(classifier_defaults.channel_mean))
    classifier.setdefault("channel_std", list(classifier_defaults.channel_std))
    if classifier["architecture"] != CIFAR10_CLASSIFIER_ARCHITECTURE:
        raise ValueError(
            "semantic_classifier.architecture must be "
            f"{CIFAR10_CLASSIFIER_ARCHITECTURE}"
        )
    source_run = classifier.get("source_run")
    if source_run is not None and (
        not isinstance(source_run, str) or not source_run.strip()
    ):
        raise ValueError(
            "semantic_classifier.source_run must be a non-empty path"
        )
    for name in ("source_state_sha256", "source_checkpoint_sha256"):
        expected_hash = classifier.get(name)
        if expected_hash is None:
            continue
        if not isinstance(expected_hash, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_hash
        ) is None:
            raise ValueError(
                f"semantic_classifier.{name} must be a 64-character SHA-256"
            )
        classifier[name] = expected_hash.lower()
    classifier_training = classifier.setdefault("training", {})
    classifier_training.setdefault("epochs", 1 if is_smoke else 50)
    classifier_training.setdefault("batch_size", 64 if is_smoke else 128)
    classifier_training.setdefault("learning_rate", 1e-3)
    classifier_training.setdefault("weight_decay", 5e-4)
    classifier_training.setdefault("samples_per_class", 20 if is_smoke else 1900)
    classifier_training.setdefault("augmentation_crop_padding", 4)
    classifier_training.setdefault(
        "augmentation_horizontal_flip_probability", 0.5
    )
    for name in ("epochs", "batch_size", "learning_rate", "samples_per_class"):
        if float(classifier_training[name]) <= 0:
            raise ValueError(
                f"semantic_classifier.training.{name} must be positive"
            )
    if float(classifier_training["weight_decay"]) < 0:
        raise ValueError(
            "semantic_classifier.training.weight_decay must be non-negative"
        )
    if int(classifier_training["augmentation_crop_padding"]) < 0:
        raise ValueError(
            "semantic_classifier.training.augmentation_crop_padding must be non-negative"
        )
    classifier_flip = float(
        classifier_training["augmentation_horizontal_flip_probability"]
    )
    if not 0.0 <= classifier_flip <= 1.0:
        raise ValueError(
            "semantic_classifier.training.augmentation_horizontal_flip_probability "
            "must be in [0, 1]"
        )
    config["algorithm"] = algorithm
    return config


def _require_method_fields(config, algorithm, names):
    missing = [name for name in names if name not in config["unlearning"]]
    if missing:
        raise ValueError(f"{algorithm} unlearning config is missing: {missing}")


def _require_positive_fields(section, names):
    for name in names:
        if float(section.get(name, 0)) <= 0:
            raise ValueError(f"unlearning.{name} must be positive")


def load_fscu_config(path):
    """Load the target-JU plus public contrastive-compensation baseline."""
    config = _load_federated_codec_config(path, "FSCU")
    unlearning = config["unlearning"]
    mode = unlearning.get("mode")
    if mode != "scu_public_compensation":
        raise ValueError(
            "FSCU requires unlearning.mode=scu_public_compensation; "
            f"received {mode}"
        )
    _require_method_fields(
        config,
        "FSCU",
        (
            "forgetting_epochs",
            "lambda_kl",
            "lambda_rec",
            "erased_buffer_max_examples",
            "server_cc_epochs",
            "server_cc_learning_rate",
            "contrastive_temperature",
            "lambda_contrastive",
            "lambda_public_reconstruction",
        ),
    )
    unlearning.setdefault("canonical_latent", "normalized_posterior_mean")
    unlearning.setdefault("interleaving_cycles", 1)
    unlearning.setdefault("negative_buffer_mode", "refresh")
    unlearning.setdefault("erased_buffer_sampling", "stratified")
    unlearning.setdefault("public_positive_mode", "deterministic_augmentation")
    unlearning.setdefault("augmentation_crop_padding", 4)
    unlearning.setdefault("augmentation_horizontal_flip_probability", 0.5)
    unlearning.setdefault("lambda_public_prior_kl", 0.0)
    unlearning.setdefault("lambda_decoder_preservation", 0.0)
    unlearning.setdefault("semantic_ascent", True)
    unlearning.setdefault("persist_interleaved_optimizer_state", True)
    if unlearning["canonical_latent"] != "normalized_posterior_mean":
        raise ValueError(
            "FSCU currently requires canonical_latent=normalized_posterior_mean"
        )
    if unlearning["negative_buffer_mode"] not in (
        "original", "post_ju_static", "refresh"
    ):
        raise ValueError("unsupported unlearning.negative_buffer_mode")
    if unlearning["erased_buffer_sampling"] not in (
        "sequential", "random", "stratified"
    ):
        raise ValueError("unsupported unlearning.erased_buffer_sampling")
    if unlearning["public_positive_mode"] not in (
        "stochastic_self", "deterministic_augmentation"
    ):
        raise ValueError("unsupported unlearning.public_positive_mode")
    cycles = int(unlearning["interleaving_cycles"])
    if cycles <= 0:
        raise ValueError("unlearning.interleaving_cycles must be positive")
    if cycles > min(
        int(unlearning["forgetting_epochs"]), int(unlearning["server_cc_epochs"])
    ):
        raise ValueError(
            "interleaving_cycles cannot exceed forgetting or server CC epochs"
        )
    if int(unlearning["augmentation_crop_padding"]) < 0:
        raise ValueError("augmentation_crop_padding must be non-negative")
    flip_probability = float(unlearning["augmentation_horizontal_flip_probability"])
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "augmentation_horizontal_flip_probability must be in [0, 1]"
        )
    for name in ("lambda_public_prior_kl", "lambda_decoder_preservation"):
        if float(unlearning[name]) < 0:
            raise ValueError(f"unlearning.{name} must be non-negative")
    _require_positive_fields(
        unlearning,
        (
            "forgetting_epochs",
            "erased_buffer_max_examples",
            "server_cc_epochs",
            "server_cc_learning_rate",
            "contrastive_temperature",
        ),
    )
    return config


def load_training_curve_config(path):
    """Load one explicitly separated reconstruction or classification task."""

    raw_config = load_json_config(Path(path))
    task = raw_config.get("task")
    if task is None:
        raise ValueError("training_curves requires an explicit task")
    if task == "classification":
        config = _load_classification_training_curve_config(raw_config)
    elif task == "reconstruction":
        config = _load_reconstruction_training_curve_config(raw_config)
    else:
        raise ValueError(
            "training_curves.task must be reconstruction or classification"
        )
    diagnostics = config.setdefault("diagnostics", {})
    diagnostics.setdefault("evaluation_interval", 5)
    if int(diagnostics["evaluation_interval"]) <= 0:
        raise ValueError("diagnostics.evaluation_interval must be positive")
    return config


def _load_reconstruction_training_curve_config(config):
    """Validate reconstruction-only FL without classifier/unlearning fields."""

    config = copy.deepcopy(config)
    required_sections = (
        "dataset",
        "model",
        "codec",
        "federation",
        "split",
        "training",
        "public_data",
        "evaluation",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(
            f"reconstruction training_curves is missing sections: {missing}"
        )
    forbidden = [
        name
        for name in ("semantic_classifier", "unlearning", "backdoor")
        if name in config
    ]
    if forbidden:
        raise ValueError(
            "reconstruction training_curves cannot contain classifier/unlearning "
            f"sections: {forbidden}"
        )
    if config["dataset"].get("name") != "CIFAR10":
        raise ValueError("reconstruction training_curves supports CIFAR-10 only")
    if config["model"] != {"kind": "variational_deepjscc"}:
        raise ValueError(
            "reconstruction training_curves requires "
            "model.kind=variational_deepjscc"
        )

    federation = config["federation"]
    if int(federation.get("n_clients", 0)) < 2:
        raise ValueError("federation.n_clients must be at least 2")
    for name in ("samples_per_client", "rounds", "local_epochs"):
        if int(federation.get(name, 0)) <= 0:
            raise ValueError(f"federation.{name} must be positive")
    federation.setdefault("persist_client_optimizer_state", False)
    if bool(federation["persist_client_optimizer_state"]):
        raise ValueError(
            "reconstruction training_curves requires a fresh optimizer each round"
        )
    federation.setdefault("serialize_cuda_clients", False)
    federation.setdefault("cuda_client_group_size", 1)
    if int(federation["cuda_client_group_size"]) <= 0:
        raise ValueError("federation.cuda_client_group_size must be positive")

    split_type = config["split"].get("type")
    if split_type not in (
        "iid",
        "equal_size_dirichlet",
        "fedosd_balanced_pathological",
    ):
        raise ValueError(
            "reconstruction training_curves requires split.type to be iid, "
            "equal_size_dirichlet, or fedosd_balanced_pathological"
        )
    if split_type == "equal_size_dirichlet" and float(
        config["split"].get("alpha", 0)
    ) <= 0:
        raise ValueError("split.alpha must be positive")
    if split_type == "fedosd_balanced_pathological" and int(
        config["split"].get("classes_per_client", 0)
    ) <= 0:
        raise ValueError("split.classes_per_client must be positive")
    if float(config["public_data"].get("samples_per_class", 0)) <= 0:
        raise ValueError("public_data.samples_per_class must be positive")
    for name in ("batch_size", "snr_db"):
        if float(config["evaluation"].get(name, 0)) <= 0:
            raise ValueError(f"evaluation.{name} must be positive")
    _configure_codec_evaluation(config["evaluation"])

    config["codec"] = _normalize_variational_deepjscc_codec(config["codec"])
    _validate_optional_bandwidth(config)

    training = config["training"]
    allowed_training_fields = {
        "optimizer",
        "batch_size",
        "learning_rate",
        "beta",
        "reconstruction_weight",
        "snr_db",
        "augmentation",
        "augmentation_crop_padding",
        "augmentation_horizontal_flip_probability",
        "learning_rate_decay_round",
        "learning_rate_after_decay",
        "gradient_clip_norm",
    }
    unknown_training_fields = sorted(set(training) - allowed_training_fields)
    if unknown_training_fields:
        raise ValueError(
            "reconstruction training has non-reconstruction fields: "
            f"{unknown_training_fields}"
        )
    training.setdefault("optimizer", "adam")
    training.setdefault("beta", 0.0)
    training.setdefault("reconstruction_weight", 1.0)
    training.setdefault("augmentation", "cifar10_crop_flip_v1")
    training.setdefault("augmentation_crop_padding", 4)
    training.setdefault("augmentation_horizontal_flip_probability", 0.5)
    _configure_reconstruction_training(
        training, config["codec"], federation["rounds"]
    )
    if training["optimizer"] != "adam":
        raise ValueError("reconstruction training_curves requires optimizer=adam")
    for name in ("batch_size", "learning_rate", "snr_db"):
        if float(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if float(training["beta"]) < 0:
        raise ValueError("training.beta must be non-negative")
    if float(training["reconstruction_weight"]) <= 0:
        raise ValueError("training.reconstruction_weight must be positive")
    if training["augmentation"] not in ("none", "cifar10_crop_flip_v1"):
        raise ValueError(
            "training.augmentation must be none or cifar10_crop_flip_v1"
        )
    if int(training["augmentation_crop_padding"]) < 0:
        raise ValueError(
            "training.augmentation_crop_padding must be non-negative"
        )
    flip_probability = float(
        training["augmentation_horizontal_flip_probability"]
    )
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "training.augmentation_horizontal_flip_probability must be in [0, 1]"
        )

    config["task"] = "reconstruction"
    config["algorithm"] = "training_curves"
    return config


def _load_classification_training_curve_config(config):
    """Validate classifier-only FL without materializing codec/observer fields."""

    config = copy.deepcopy(config)
    required_sections = (
        "dataset",
        "model",
        "federation",
        "split",
        "training",
        "public_data",
        "evaluation",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(
            f"classification training_curves is missing sections: {missing}"
        )
    forbidden = [
        name
        for name in ("codec", "semantic_classifier", "unlearning", "backdoor")
        if name in config
    ]
    if forbidden:
        raise ValueError(
            "classification training_curves cannot contain codec/unlearning "
            f"sections: {forbidden}"
        )
    if config["dataset"].get("name") != "CIFAR10":
        raise ValueError("classification training_curves supports CIFAR-10 only")

    federation = config["federation"]
    if int(federation.get("n_clients", 0)) < 2:
        raise ValueError("federation.n_clients must be at least 2")
    for name in ("samples_per_client", "rounds", "local_epochs"):
        if int(federation.get(name, 0)) <= 0:
            raise ValueError(f"federation.{name} must be positive")
    federation.setdefault("persist_client_optimizer_state", False)
    if bool(federation["persist_client_optimizer_state"]):
        raise ValueError(
            "classification training_curves requires a fresh optimizer each round"
        )
    federation.setdefault("serialize_cuda_clients", False)
    federation.setdefault("cuda_client_group_size", 1)
    if int(federation["cuda_client_group_size"]) <= 0:
        raise ValueError("federation.cuda_client_group_size must be positive")

    split_type = config["split"].get("type")
    if split_type not in (
        "iid",
        "equal_size_dirichlet",
        "fedosd_balanced_pathological",
    ):
        raise ValueError(
            "classification training_curves requires split.type to be iid, "
            "equal_size_dirichlet, or fedosd_balanced_pathological"
        )
    if split_type == "equal_size_dirichlet" and float(
        config["split"].get("alpha", 0)
    ) <= 0:
        raise ValueError("split.alpha must be positive")
    if split_type == "fedosd_balanced_pathological" and int(
        config["split"].get("classes_per_client", 0)
    ) <= 0:
        raise ValueError("split.classes_per_client must be positive")
    if float(config["public_data"].get("samples_per_class", 0)) <= 0:
        raise ValueError("public_data.samples_per_class must be positive")
    if float(config["evaluation"].get("batch_size", 0)) <= 0:
        raise ValueError("evaluation.batch_size must be positive")

    training = config["training"]
    training.setdefault("optimizer", "adam")
    training.setdefault("weight_decay", 0.0)
    training.setdefault("augmentation", "cifar10_crop_flip_v1")
    training.setdefault("augmentation_crop_padding", 4)
    training.setdefault("augmentation_horizontal_flip_probability", 0.5)
    if training["optimizer"] != "adam":
        raise ValueError("classification training_curves requires optimizer=adam")
    for name in ("batch_size", "learning_rate"):
        if float(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if float(training["weight_decay"]) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if training["augmentation"] not in ("none", "cifar10_crop_flip_v1"):
        raise ValueError(
            "training.augmentation must be none or cifar10_crop_flip_v1"
        )
    if int(training["augmentation_crop_padding"]) < 0:
        raise ValueError(
            "training.augmentation_crop_padding must be non-negative"
        )
    flip_probability = float(
        training["augmentation_horizontal_flip_probability"]
    )
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "training.augmentation_horizontal_flip_probability must be in [0, 1]"
        )

    defaults = Cifar10ClassifierConfig()
    model = config["model"]
    allowed_fields = set(vars(defaults)) | {"kind", "architecture"}
    unknown_fields = sorted(set(model) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"classification model has unknown fields: {unknown_fields}")
    if model.get("kind") != "cifar10_classifier":
        raise ValueError(
            "classification training_curves requires model.kind=cifar10_classifier"
        )
    model.setdefault("architecture", CIFAR10_CLASSIFIER_ARCHITECTURE)
    model.setdefault("base_channels", defaults.base_channels)
    model.setdefault("group_count", defaults.group_count)
    model.setdefault("num_classes", defaults.num_classes)
    model.setdefault("channel_mean", list(defaults.channel_mean))
    model.setdefault("channel_std", list(defaults.channel_std))
    if model["architecture"] != CIFAR10_CLASSIFIER_ARCHITECTURE:
        raise ValueError(
            "classification model architecture must be "
            f"{CIFAR10_CLASSIFIER_ARCHITECTURE}"
        )
    Cifar10ClassifierConfig(
        **{
            name: value
            for name, value in model.items()
            if name not in ("kind", "architecture")
        }
    )
    config["task"] = "classification"
    config["algorithm"] = "training_curves"
    return config




def load_fscuv4_config(path):
    """Load joint forgetting plus conflict-aware retained repair settings."""
    config = _load_federated_codec_config(path, "FSCUv4")
    unlearning = config["unlearning"]
    diagnostics = config.setdefault("diagnostics", {})
    diagnostics.setdefault("evaluation_interval", 20)
    if int(diagnostics["evaluation_interval"]) <= 0:
        raise ValueError("diagnostics.evaluation_interval must be positive")
    mode = unlearning.get("mode")
    if mode not in (
        "retained_only_feature_kd",
        "retained_only_reconstruction",
    ):
        raise ValueError(
            "FSCUv4 requires retained-only feature KD or reconstruction repair; "
            f"received {mode}"
        )
    _require_method_fields(
        config,
        "FSCUv4",
        (
            "forgetting_epochs",
            "lambda_kl",
            "lambda_rec",
            "distributed_repair_rounds",
            "local_repair_epochs",
            "local_repair_learning_rate",
            "lambda_repair_reconstruction",
            "lambda_repair_kl",
            "lambda_output_kd",
            "lambda_feature_kd",
        ),
    )
    unlearning.setdefault("canonical_latent", "normalized_posterior_mean")
    unlearning.setdefault("lambda_decoder_preservation", 0.0)
    unlearning.setdefault("semantic_ascent", False)
    unlearning.setdefault("lambda_repair_semantic", 0.0)
    unlearning.setdefault("repair_aggregation", "conflict_aware_projection")
    unlearning.setdefault("conflict_reference", "target_joint_unlearning_update")
    unlearning.setdefault("conflict_projection_scope", "global_client_update")
    unlearning.setdefault("conflict_epsilon", 1e-12)
    unlearning.setdefault("run_repair_ablations", True)
    unlearning.setdefault("primary_repair_variant", "feature_kd")
    unlearning.setdefault("method_label", "FSCUv4")
    unlearning.setdefault(
        "repair_checkpoint_rounds",
        [int(unlearning["distributed_repair_rounds"])],
    )
    unlearning.setdefault(
        "forgetting_objective", "posterior_kl_to_original"
    )
    unlearning.setdefault("target_information_ratio_threshold", 0.0)
    if unlearning["canonical_latent"] != "normalized_posterior_mean":
        raise ValueError(
            "FSCUv4 requires canonical_latent=normalized_posterior_mean"
        )
    if unlearning["repair_aggregation"] not in (
        "conflict_aware_projection",
        "sample_weighted_fedavg",
    ):
        raise ValueError(
            "FSCUv4 repair_aggregation must be conflict_aware_projection or "
            "sample_weighted_fedavg"
        )
    if unlearning["repair_aggregation"] == "conflict_aware_projection":
        if unlearning["conflict_reference"] != "target_joint_unlearning_update":
            raise ValueError(
                "FSCUv4 conflict_reference must be target_joint_unlearning_update"
            )
        if unlearning["conflict_projection_scope"] != "global_client_update":
            raise ValueError(
                "FSCUv4 conflict_projection_scope must be global_client_update"
            )
        if float(unlearning["conflict_epsilon"]) <= 0:
            raise ValueError("unlearning.conflict_epsilon must be positive")
    if unlearning["primary_repair_variant"] not in (
        "normal_repair",
        "output_kd",
        "feature_kd",
    ):
        raise ValueError("unsupported FSCUv4 primary repair variant")
    if mode == "retained_only_reconstruction" and (
        unlearning["primary_repair_variant"] != "normal_repair"
    ):
        raise ValueError(
            "retained_only_reconstruction requires primary_repair_variant=normal_repair"
        )
    if not isinstance(unlearning["method_label"], str) or not unlearning[
        "method_label"
    ].replace("_", "").isalnum():
        raise ValueError("unlearning.method_label must be an alphanumeric label")
    repair_rounds = int(unlearning["distributed_repair_rounds"])
    checkpoint_rounds = [
        int(value) for value in unlearning["repair_checkpoint_rounds"]
    ]
    if (
        not checkpoint_rounds
        or checkpoint_rounds != sorted(set(checkpoint_rounds))
        or checkpoint_rounds[-1] > repair_rounds
        or checkpoint_rounds[0] <= 0
    ):
        raise ValueError(
            "repair_checkpoint_rounds must be sorted unique values within the budget"
        )
    unlearning["repair_checkpoint_rounds"] = checkpoint_rounds
    if unlearning["forgetting_objective"] not in (
        "posterior_kl_to_original",
        "prior_kl_collapse",
    ):
        raise ValueError("unsupported FSCUv4 forgetting objective")
    information_ratio_threshold = float(
        unlearning["target_information_ratio_threshold"]
    )
    if not 0.0 <= information_ratio_threshold < 1.0:
        raise ValueError(
            "unlearning.target_information_ratio_threshold must be in [0, 1)"
        )
    if float(config["training"].get("semantic_weight", 0.0)) != 0.0:
        raise ValueError(
            "FSCUv4 is reconstruction-only; training.semantic_weight must be zero"
        )
    if float(unlearning["lambda_repair_semantic"]) != 0.0:
        raise ValueError(
            "FSCUv4 is reconstruction-only; lambda_repair_semantic must be zero"
        )
    if bool(unlearning["semantic_ascent"]):
        raise ValueError(
            "FSCUv4 is reconstruction-only; semantic_ascent must be false"
        )
    _require_positive_fields(
        unlearning,
        (
            "forgetting_epochs",
            "lambda_kl",
            "lambda_rec",
            "distributed_repair_rounds",
            "local_repair_epochs",
            "local_repair_learning_rate",
            "lambda_repair_reconstruction",
        ),
    )
    for name in (
        "lambda_decoder_preservation",
        "lambda_repair_kl",
        "lambda_repair_semantic",
        "lambda_output_kd",
        "lambda_feature_kd",
    ):
        if float(unlearning[name]) < 0:
            raise ValueError(f"unlearning.{name} must be non-negative")
    if (
        unlearning["primary_repair_variant"] == "feature_kd"
        or bool(unlearning["run_repair_ablations"])
    ) and float(unlearning["lambda_feature_kd"]) <= 0:
        raise ValueError("feature-KD repair requires positive lambda_feature_kd")
    return config
