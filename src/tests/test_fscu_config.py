from pathlib import Path

import numpy as np
import pytest

from src.data.data import (
    build_iid_federated_split,
    fscu_split_seed,
    uses_external_semantic_classifier,
)
from src.models.semantic_codec import (
    DEEPJSCC_ARCHITECTURE,
    LEGACY_DEEPJSCC_ARCHITECTURE,
)
from src.utils.config import (
    load_fscu_config,
    load_json_config,
    load_training_curve_config,
)
from src.utils.fscu_runtime import split_interleaved_epoch_budget


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = (
    ROOT / "src/configs/federated_cifar10_vdeepjscc_scu_public_compensation_smoke.json"
)
CLEAN_IID_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_500rounds.json"
)
RECONSTRUCTION_800_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_800rounds.json"
)
CLASSIFICATION_800_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_classifier_800rounds.json"
)
CLEAN_IID_REFERENCE_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_reference.json"
)
IID_BACKDOOR_REFERENCE_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_backdoor_reference.json"
)
STANDARD_VIB_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_iid_vdeepjscc_standard_vib_49k_800rounds.json"
)
STANDARD_VIB_FEDOSD_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_fedosd_2class_vdeepjscc_standard_vib_49k_800rounds.json"
)
STANDARD_VIB_DIRICHLET_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_dirichlet_01_vdeepjscc_standard_vib_49k_800rounds.json"
)
def test_registered_fscu_config_loads_without_conversion_layers():
    config = load_fscu_config(SMOKE_CONFIG)

    assert config["unlearning"]["mode"] == "scu_public_compensation"
    assert config["federation"]["n_clients"] == 10
    assert config["federation"]["persist_client_optimizer_state"] is False
    assert config["training"]["augmentation"] == "none"
    assert config["unlearning"]["canonical_latent"] == "normalized_posterior_mean"
    assert config["unlearning"]["negative_buffer_mode"] == "refresh"
    assert (
        config["unlearning"]["public_positive_mode"]
        == "deterministic_augmentation"
    )


def test_external_classifier_source_disables_reserved_classifier_dataset():
    assert uses_external_semantic_classifier(
        {"semantic_classifier": {"source_run": "outputs/source/seed-0"}}
    )
    assert not uses_external_semantic_classifier(
        {"semantic_classifier": {"training": {"samples_per_class": 20}}}
    )


def test_clean_iid_training_defaults_to_versioned_crop_flip():
    config = load_training_curve_config(CLEAN_IID_CONFIG)

    assert "backdoor" not in config
    assert config["task"] == "reconstruction"
    assert config["model"]["kind"] == "variational_deepjscc"
    assert "semantic_classifier" not in config
    assert config["training"]["augmentation"] == "cifar10_crop_flip_v1"
    assert config["training"]["augmentation_crop_padding"] == 4
    assert config["training"]["augmentation_horizontal_flip_probability"] == 0.5
    assert config["codec"]["architecture"] == LEGACY_DEEPJSCC_ARCHITECTURE
    assert config["codec"]["kl_reduction"] == "elementwise_mean"
    assert config["training"]["gradient_clip_norm"] == 5.0
    assert config["evaluation"]["posterior_mode"] == "mean"
    assert config["evaluation"]["monte_carlo_repetitions"] == 1


def test_standard_vib_49k_config_loads_v3_objective_and_protocol():
    config = load_training_curve_config(STANDARD_VIB_CONFIG)

    assert config["codec"]["architecture"] == DEEPJSCC_ARCHITECTURE
    assert config["codec"]["latent_channels"] == 16
    assert config["codec"]["latent_dim"] == 1024
    assert config["codec"]["kl_reduction"] == "batchmean_sum"
    assert config["federation"]["samples_per_client"] == 4900
    assert config["training"]["batch_size"] == 200
    assert config["training"]["gradient_clip_norm"] is None
    assert config["training"]["learning_rate_decay_round"] == 350
    assert config["training"]["learning_rate_after_decay"] == 1e-4
    assert config["evaluation"]["posterior_mode"] == "sample"
    assert config["evaluation"]["monte_carlo_repetitions"] == 10


def test_standard_vib_49k_noniid_configs_only_change_scenario_identity_and_split():
    iid_raw = load_json_config(STANDARD_VIB_CONFIG)
    expected_splits = {
        STANDARD_VIB_FEDOSD_CONFIG: {
            "type": "fedosd_balanced_pathological",
            "classes_per_client": 2,
            "seed": 0,
        },
        STANDARD_VIB_DIRICHLET_CONFIG: {
            "type": "equal_size_dirichlet",
            "alpha": 0.1,
            "seed": 0,
        },
    }

    iid_shared = dict(iid_raw)
    iid_shared.pop("experiment_name")
    iid_shared.pop("split")
    for path, expected_split in expected_splits.items():
        candidate_raw = load_json_config(path)
        candidate_shared = dict(candidate_raw)
        candidate_shared.pop("experiment_name")
        candidate_shared.pop("split")

        assert candidate_shared == iid_shared
        assert candidate_raw["split"] == expected_split

        config = load_training_curve_config(path)
        assert config["task"] == "reconstruction"
        assert config["split"] == expected_split
        assert config["training"]["beta"] == 1e-4
        assert config["federation"]["samples_per_client"] == 4900
        assert config["federation"]["rounds"] == 800


def test_reconstruction_training_curves_validate_noniid_parameters(tmp_path):
    import json

    cases = (
        (STANDARD_VIB_DIRICHLET_CONFIG, "alpha", 0, "split.alpha"),
        (
            STANDARD_VIB_FEDOSD_CONFIG,
            "classes_per_client",
            0,
            "split.classes_per_client",
        ),
    )
    for index, (source, field, value, message) in enumerate(cases):
        config = load_json_config(source)
        config["split"][field] = value
        path = tmp_path / f"invalid_noniid_{index}.json"
        path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            load_training_curve_config(path)


def test_classification_training_curves_support_noniid_splits(tmp_path):
    import json

    split_configs = (
        {"type": "equal_size_dirichlet", "alpha": 0.1, "seed": 0},
        {
            "type": "fedosd_balanced_pathological",
            "classes_per_client": 2,
            "seed": 0,
        },
    )
    for index, split_config in enumerate(split_configs):
        config = load_json_config(CLASSIFICATION_800_CONFIG)
        config["split"] = split_config
        path = tmp_path / f"noniid_classification_{index}.json"
        path.write_text(json.dumps(config), encoding="utf-8")

        loaded = load_training_curve_config(path)

        assert loaded["task"] == "classification"
        assert loaded["algorithm"] == "training_curves"
        assert loaded["split"] == split_config
        assert loaded["training"]["optimizer"] == "adam"
        assert "codec" not in loaded
        assert "beta" not in loaded["training"]


def test_classification_training_curves_validate_noniid_parameters(tmp_path):
    import json

    cases = (
        ({"type": "equal_size_dirichlet", "alpha": 0, "seed": 0}, "split.alpha"),
        (
            {
                "type": "fedosd_balanced_pathological",
                "classes_per_client": 0,
                "seed": 0,
            },
            "split.classes_per_client",
        ),
    )
    for index, (split_config, message) in enumerate(cases):
        config = load_json_config(CLASSIFICATION_800_CONFIG)
        config["split"] = split_config
        path = tmp_path / f"invalid_noniid_classification_{index}.json"
        path.write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            load_training_curve_config(path)


def test_standard_vib_bandwidth_must_match_latent_dimension(tmp_path):
    config = load_json_config(STANDARD_VIB_CONFIG)
    config["bandwidth"]["transmitted_real_values"] = 256
    path = tmp_path / "invalid_bandwidth.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal codec.latent_dim"):
        load_training_curve_config(path)


def test_codec_architecture_rejects_cross_version_fixed_fields(tmp_path):
    cases = (
        (STANDARD_VIB_CONFIG, "codec_format_version", 2),
        (STANDARD_VIB_CONFIG, "logvar_parameterization", "sigmoid_bounded_v1"),
        (STANDARD_VIB_CONFIG, "kl_reduction", "elementwise_mean"),
        (CLEAN_IID_CONFIG, "codec_format_version", 3),
        (CLEAN_IID_CONFIG, "logvar_parameterization", "softplus_std_v1"),
        (CLEAN_IID_CONFIG, "kl_reduction", "batchmean_sum"),
    )
    import json

    for index, (source, field, value) in enumerate(cases):
        config = load_json_config(source)
        config["codec"][field] = value
        path = tmp_path / f"invalid_architecture_field_{index}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match=f"codec.{field} must equal"):
            load_training_curve_config(path)


def test_codec_architecture_rejects_wrong_logvar_field_combination(tmp_path):
    import json

    v3_config = load_json_config(STANDARD_VIB_CONFIG)
    v3_config["codec"]["logvar_min"] = -8.0
    v3_path = tmp_path / "invalid_v3_logvar_bounds.json"
    v3_path.write_text(json.dumps(v3_config), encoding="utf-8")
    with pytest.raises(ValueError, match="must both be null"):
        load_training_curve_config(v3_path)

    v2_config = load_json_config(CLEAN_IID_CONFIG)
    v2_config["codec"]["logvar_min"] = None
    v2_path = tmp_path / "invalid_v2_logvar_bounds.json"
    v2_path.write_text(json.dumps(v2_config), encoding="utf-8")
    with pytest.raises(ValueError, match="must both be finite"):
        load_training_curve_config(v2_path)


def test_codec_latent_dimension_must_match_latent_channels(tmp_path):
    config = load_json_config(STANDARD_VIB_CONFIG)
    config["codec"]["latent_channels"] = 15
    path = tmp_path / "invalid_latent_shape.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=r"latent_dim must equal 64 \*"):
        load_training_curve_config(path)


def test_learning_rate_decay_round_must_fit_federation_rounds(tmp_path):
    config = load_json_config(STANDARD_VIB_CONFIG)
    config["training"]["learning_rate_decay_round"] = 801
    path = tmp_path / "invalid_learning_rate_schedule.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must not exceed federation.rounds"):
        load_training_curve_config(path)


def test_independent_800round_tasks_share_deterministic_iid_split_parameters():
    reconstruction = load_training_curve_config(RECONSTRUCTION_800_CONFIG)
    classification = load_training_curve_config(CLASSIFICATION_800_CONFIG)

    assert reconstruction["public_data"] == classification["public_data"]
    assert reconstruction["split"] == classification["split"]
    for name in ("n_clients", "samples_per_client", "rounds", "local_epochs"):
        assert reconstruction["federation"][name] == classification["federation"][name]

    reconstruction_seed = fscu_split_seed(0, reconstruction)
    classification_seed = fscu_split_seed(0, classification)
    assert reconstruction_seed == classification_seed

    samples_per_class = (
        reconstruction["federation"]["n_clients"]
        * reconstruction["federation"]["samples_per_client"]
        // 10
    )
    train_labels = np.repeat(np.arange(10), samples_per_class)
    test_labels = np.repeat(np.arange(10), 1000)
    reconstruction_split = build_iid_federated_split(
        train_labels,
        test_labels,
        reconstruction["federation"]["n_clients"],
        reconstruction_seed,
    )
    classification_split = build_iid_federated_split(
        train_labels,
        test_labels,
        classification["federation"]["n_clients"],
        classification_seed,
    )
    for reconstruction_indices, classification_indices in zip(
        reconstruction_split.train_indices,
        classification_split.train_indices,
    ):
        assert np.array_equal(reconstruction_indices, classification_indices)

    assert reconstruction["model"]["kind"] == "variational_deepjscc"
    assert "codec" in reconstruction
    assert "semantic_classifier" not in reconstruction
    assert "weight_decay" not in reconstruction["training"]

    assert classification["model"]["kind"] == "cifar10_classifier"
    assert "codec" not in classification
    assert "semantic_classifier" not in classification
    assert "reconstruction_weight" not in classification["training"]
    assert "beta" not in classification["training"]
    assert "snr_db" not in classification["training"]
    assert "channel" not in classification["evaluation"]
    assert "snr_db" not in classification["evaluation"]


def test_iid_reference_names_make_clean_and_backdoor_scenarios_explicit():
    clean = load_json_config(CLEAN_IID_REFERENCE_CONFIG)
    backdoor = load_json_config(IID_BACKDOOR_REFERENCE_CONFIG)

    assert "backdoor" not in clean
    assert backdoor["backdoor"]["protocol"] == "scu_cifar10"


def test_backdoor_config_rejects_crop_flip(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    config["training"]["augmentation"] = "cifar10_crop_flip_v1"
    path = tmp_path / "invalid_backdoor_augmentation.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_fscu_config(path)
    except ValueError as error:
        assert "fixed SCU trigger protocol" in str(error)
    else:
        raise AssertionError("crop/flip was accepted with a backdoor config")


def test_interleaved_epoch_budget_preserves_total_epochs():
    assert split_interleaved_epoch_budget(5, 3) == [2, 2, 1]
    assert split_interleaved_epoch_budget(40, 5) == [8, 8, 8, 8, 8]


def test_fscu_loader_rejects_non_fscu_mode(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    config["unlearning"]["mode"] = "remaining_client_consensus"
    path = tmp_path / "invalid.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_fscu_config(path)
    except ValueError as error:
        assert "scu_public_compensation" in str(error)
    else:
        raise AssertionError("non-FSCU mode was accepted")
