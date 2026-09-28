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
