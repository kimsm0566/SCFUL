from pathlib import Path
from types import SimpleNamespace

import torch

import src.utils.trainer as trainer_module
from src.utils.config import load_fscuv4_config
from src.utils.federated import (
    RetainedFeatureKDConfig,
    conflict_aware_average_updates,
    train_local_retained_feature_kd,
)
from src.utils.fscu_runtime import make_fscu_codec
from src.utils.fscu import module_wise_budgeted_forget_target_client
from src.utils.loss import (
    outside_marker_consistency_loss,
    posterior_parameter_distillation_losses,
)
from src.models.semantic_codec import Posterior


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "src/configs/federated_cifar10_vdeepjscc_fscuv4_smoke.json"
IBJU_FEDAVG_CONFIG = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_ibju_fedavg.json"
)
MAIN_CONFIGS = (
    ROOT / "src/configs/federated_cifar10_iid_vdeepjscc_fscuv4.json",
    ROOT / "src/configs/federated_cifar10_fedosd_2class_vdeepjscc_fscuv4.json",
    ROOT / "src/configs/federated_cifar10_dirichlet_01_vdeepjscc_fscuv4.json",
)
MARKER_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a_canary_visible_fscuv4.json"
)
POSTERIOR_GUARDED_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_iid_vdeepjscc_marker_masked_input_fscuv4_posterior_guarded_800rounds.json"
)
MODULE_BUDGETED_CONFIG = (
    ROOT
    / "src/configs/federated_cifar10_vdeepjscc_module_budgeted_fscuv4_smoke.json"
)
NON_IID_3STATE_CONFIGS = (
    ROOT
    / "src/configs/federated_cifar10_fedosd_2class_vdeepjscc_fscuv4_800rounds_3state.json",
    ROOT
    / "src/configs/federated_cifar10_dirichlet_01_vdeepjscc_fscuv4_800rounds_3state.json",
)


def test_fscuv4_config_registers_retained_only_feature_kd():
    config = load_fscuv4_config(SMOKE_CONFIG)

    assert config["algorithm"] == "FSCUv4"
    assert config["unlearning"]["mode"] == "retained_only_feature_kd"
    assert config["unlearning"]["lambda_feature_kd"] > 0
    assert config["unlearning"]["repair_aggregation"] == (
        "conflict_aware_projection"
    )
    assert config["training"]["semantic_weight"] == 0.0
    assert config["unlearning"]["lambda_repair_semantic"] == 0.0
    assert config["codec"]["architecture"] == "variational_deepjscc_cifar10_v3"


def test_fscuv4_main_configs_use_the_current_standard_vib_protocol():
    configs = [load_fscuv4_config(path) for path in MAIN_CONFIGS]

    for config in configs:
        assert config["codec"]["latent_channels"] == 16
        assert config["codec"]["latent_dim"] == 1024
        assert config["codec"]["kl_reduction"] == "batchmean_sum"
        assert config["federation"]["samples_per_client"] == 4900
        assert config["federation"]["rounds"] == 800
        assert config["federation"]["persist_client_optimizer_state"] is False
        assert config["training"]["batch_size"] == 200
        assert config["training"]["beta"] == 0.0001
        assert config["training"]["semantic_weight"] == 0.0
        assert config["unlearning"]["repair_aggregation"] == (
            "conflict_aware_projection"
        )


def test_fscuv4_marker_config_preserves_the_visible_canary_protocol():
    config = load_fscuv4_config(MARKER_CONFIG)

    marker = config["reconstruction_marker"]
    assert marker["target_client_id"] == config["unlearning"]["target_client_id"]
    assert marker["protocol"] == "unconditional_output_watermark_v1"
    assert marker["poison_ratio"] == 1.0
    assert marker["marker_region_loss_weight"] == 100.0
    assert marker["normalize_marker_loss_weights"] is False
    assert marker["target_client_aggregation_multiplier"] == 10.0
    assert marker["output_watermark"]["alpha"] == 1.0
    assert marker["output_watermark"]["cell_size"] == 4
    assert config["unlearning"]["distributed_repair_rounds"] == 100


def test_fscuv4_posterior_guarded_config_registers_both_guards():
    config = load_fscuv4_config(POSTERIOR_GUARDED_CONFIG)

    assert config["unlearning"]["lambda_repair_kl"] == 0.0
    assert config["unlearning"]["lambda_posterior_mu_kd"] == 1.0
    assert config["unlearning"]["lambda_posterior_logvar_kd"] == 1.0
    assert config["unlearning"]["lambda_marker_outside_consistency"] == 0.25
    assert config["reconstruction_marker"]["protocol"] == (
        "masked_input_marker_reconstruction_v1"
    )


def test_fscuv4_module_budgeted_config_registers_separate_stop_budgets():
    config = load_fscuv4_config(MODULE_BUDGETED_CONFIG)

    unlearning = config["unlearning"]
    assert unlearning["forgetting_objective"] == "module_wise_budgeted"
    assert unlearning["target_information_ratio_threshold"] == 0.99
    assert unlearning["decoder_psnr_drop_db"] == 0.01
    assert unlearning["encoder_unlearning_max_epochs"] == 1
    assert unlearning["decoder_unlearning_max_epochs"] == 1


def test_non_iid_three_state_configs_use_full_training_and_repair_budgets():
    configs = [load_fscuv4_config(path) for path in NON_IID_3STATE_CONFIGS]

    assert [config["split"]["type"] for config in configs] == [
        "fedosd_balanced_pathological",
        "equal_size_dirichlet",
    ]
    for config in configs:
        assert config["federation"]["rounds"] == 800
        assert config["unlearning"]["distributed_repair_rounds"] == 100
        assert config["unlearning"]["repair_checkpoint_rounds"] == [100]
        assert config["unlearning"]["run_repair_ablations"] is False
        assert config["unlearning"]["primary_repair_variant"] == "feature_kd"
        assert config["unlearning"]["method_label"] == "FSCUv4"


def test_module_budgeted_unlearning_updates_encoder_then_decoder():
    config = load_fscuv4_config(MODULE_BUDGETED_CONFIG)
    original = make_fscu_codec(config)
    learner = make_fscu_codec(config)
    learner.load_state_dict(original.state_dict())
    images = torch.rand(8, 3, 32, 32)
    labels = torch.arange(8, dtype=torch.long) % 10
    local_config = {
        "batch_size": 4,
        "snr_db": 20.0,
        "lambda_kl": 1.0,
        "lambda_rec": 0.1,
        "target_information_ratio_threshold": 0.999999,
        "encoder_unlearning_max_epochs": 1,
        "decoder_unlearning_max_epochs": 1,
        "encoder_learning_rate": 1e-4,
        "decoder_learning_rate": 1e-4,
        "decoder_psnr_drop_db": 0.0001,
        "lambda_encoder_proximal": 0.0,
        "lambda_decoder_proximal": 0.0,
        "device": "cpu",
    }
    before = {
        name: value.detach().clone()
        for name, value in learner.state_dict().items()
    }

    state, metrics = module_wise_budgeted_forget_target_client(
        learner,
        (images, labels),
        original,
        local_config,
        seed=7,
    )

    assert not torch.equal(before["encoder.0.weight"], state["encoder.0.weight"])
    assert not torch.equal(before["decoder.0.weight"], state["decoder.0.weight"])
    assert [row["stage"] for row in metrics["history"]] == [
        "encoder_information_rate",
        "decoder_distortion_budget",
    ]
    assert metrics["encoder_stage"]["epochs_completed"] == 1
    assert metrics["decoder_stage"]["epochs_completed"] == 1
    assert metrics["decoder_stage"]["original_probe"]["path"] == (
        "original_encoder_to_current_decoder"
    )


def test_posterior_parameter_kd_separates_mean_and_logvar():
    student = Posterior(
        mu=torch.tensor([[1.0, 0.0]]),
        logvar=torch.tensor([[0.0, 2.0]]),
    )
    teacher = Posterior(
        mu=torch.zeros(1, 2),
        logvar=torch.zeros(1, 2),
    )

    mean_loss, logvar_loss = posterior_parameter_distillation_losses(
        student, teacher
    )

    assert mean_loss.item() == 0.5
    assert logvar_loss.item() == 2.0


def test_marker_consistency_ignores_marker_region_and_penalizes_outside():
    clean = torch.zeros(1, 3, 4, 4)
    marked = clean.clone()
    marked[:, :, 2:4, 2:4] = 1.0

    assert outside_marker_consistency_loss(
        marked, clean, (2, 2, 2, 2)
    ).item() == 0.0

    marked[:, :, 0, 0] = 1.0
    assert outside_marker_consistency_loss(
        marked, clean, (2, 2, 2, 2)
    ).item() == 3.0


def test_ibju_fedavg_config_disables_teacher_and_conflict_projection():
    config = load_fscuv4_config(IBJU_FEDAVG_CONFIG)

    assert config["unlearning"]["mode"] == "retained_only_reconstruction"
    assert config["unlearning"]["forgetting_objective"] == "prior_kl_collapse"
    assert config["unlearning"]["target_information_ratio_threshold"] == 0.01
    assert config["unlearning"]["primary_repair_variant"] == "normal_repair"
    assert config["unlearning"]["repair_aggregation"] == "sample_weighted_fedavg"
    assert config["unlearning"]["repair_checkpoint_rounds"] == [
        1,
        5,
        10,
        20,
        50,
        100,
    ]
    assert config["unlearning"]["lambda_output_kd"] == 0.0
    assert config["unlearning"]["lambda_feature_kd"] == 0.0


def test_conflict_aware_aggregation_removes_only_reverse_component():
    original = {
        "weight": torch.tensor([0.0, 0.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    forgotten = {
        "weight": torch.tensor([1.0, 0.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    base = {
        "weight": torch.tensor([1.0, 0.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    local_states = [
        {
            "weight": torch.tensor([0.0, 2.0]),
            "counter": torch.tensor(4, dtype=torch.long),
        },
        {
            "weight": torch.tensor([2.0, 0.0]),
            "counter": torch.tensor(5, dtype=torch.long),
        },
    ]

    state, metrics = conflict_aware_average_updates(
        base,
        local_states,
        sample_counts=[1, 1],
        original_state=original,
        forgotten_state=forgotten,
    )

    assert torch.allclose(state["weight"], torch.tensor([1.5, 1.0]))
    assert state["counter"].item() == 3
    assert metrics["conflict_client_count"] == 1
    assert metrics["conflict_client_fraction"] == 0.5
    assert abs(metrics["client_diagnostics"][0]["cosine_after"]) < 1e-12
    assert abs(metrics["client_diagnostics"][0]["removed_norm_fraction"] - 0.4472135955) < 1e-9
    assert metrics["aggregate_cosine_after"] >= -1e-12


def test_variational_deepjscc_exposes_shallow_distillation_features():
    config = load_fscuv4_config(SMOKE_CONFIG)
    model = make_fscu_codec(config)
    images = torch.rand(3, 3, 32, 32)
    noise = torch.randn(3, config["codec"]["latent_dim"])

    output = model(
        images,
        mode="distillation",
        standard_noise=noise,
        snr_db=20.0,
    )

    assert output.reconstruction.shape == images.shape
    assert output.encoder_shallow.shape == (3, 16, 16, 16)
    assert output.decoder_shallow.shape == (3, 16, 16, 16)


def test_retained_feature_kd_updates_both_codec_sides():
    config = load_fscuv4_config(SMOKE_CONFIG)
    learner = make_fscu_codec(config)
    teacher = make_fscu_codec(config)
    teacher.load_state_dict(learner.state_dict())
    images = torch.rand(8, 3, 32, 32)
    labels = torch.arange(8, dtype=torch.long) % 10
    before = {
        name: value.detach().clone()
        for name, value in learner.state_dict().items()
    }
    local_config = RetainedFeatureKDConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        snr_db=20.0,
        reconstruction_weight=30.0,
        semantic_weight=0.0,
        output_kd_weight=0.0,
        feature_kd_weight=1.0,
        device="cpu",
    )

    state, metrics = train_local_retained_feature_kd(
        learner,
        teacher,
        (images, labels),
        local_config,
        seed=3,
    )

    assert not torch.equal(before["encoder.0.weight"], state["encoder.0.weight"])
    assert not torch.equal(before["decoder.0.weight"], state["decoder.0.weight"])
    assert metrics["steps"] == 2
    assert metrics["uses_target_data"] is False


def test_retained_reconstruction_repair_does_not_require_original_teacher():
    config = load_fscuv4_config(IBJU_FEDAVG_CONFIG)
    learner = make_fscu_codec(config)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.arange(8, dtype=torch.long) % 10
    local_config = RetainedFeatureKDConfig(
        batch_size=4,
        local_epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        snr_db=20.0,
        reconstruction_weight=1.0,
        semantic_weight=0.0,
        output_kd_weight=0.0,
        feature_kd_weight=0.0,
        device="cpu",
    )

    _, metrics = train_local_retained_feature_kd(
        learner,
        None,
        (images, labels),
        local_config,
        seed=5,
    )

    assert metrics["teacher_scope"] == "none"
    assert metrics["output_kd_loss"] == 0.0
    assert metrics["encoder_feature_kd_loss"] == 0.0
    assert metrics["decoder_feature_kd_loss"] == 0.0


def test_trainer_routes_fscuv4_to_retained_feature_kd(monkeypatch):
    expected = {"algorithm": "FSCUv4"}

    def fake_train_fscuv4(comm, data_state, args, logger):
        return expected

    monkeypatch.setattr(trainer_module, "train_fscuv4", fake_train_fscuv4)
    args = SimpleNamespace(algorithm="FSCUv4", experiment="scu")
    result = trainer_module.Trainer(args).train(
        data_state={}, comm=object(), logger=None, args=args
    )

    assert result is expected
