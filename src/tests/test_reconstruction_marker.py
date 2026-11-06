from pathlib import Path

import pytest
import torch

from src.data.data import (
    ReconstructionMarkerClientData,
    apply_reconstruction_input_mask,
    apply_reconstruction_input_key,
    apply_reconstruction_output_watermark,
    build_reconstruction_marker_client_data,
    reconstruction_marker_evaluation_scenario,
)
from src.experiments.reconstruction_marker_audit import (
    evaluate_marker_pul,
    evaluate_reconstruction_marker_states,
    marker_reference_gap,
)
from src.experiments.reconstruction_channel_audit import (
    _evaluate_reconstruction_exemplars,
    _headline_checkpoint_names,
    _reconstruction_visualization_state_names,
)
from src.utils.config import load_reconstruction_marker_audit_config
from src.utils.fscu import _make_dataset as make_fscu_target_dataset
from src.utils.plotting import (
    save_marker_pul_figure,
    save_reconstruction_marker_figures,
)


WORKSPACE = Path(__file__).resolve().parents[2]


def _marker_config(protocol):
    marker = {
        "protocol": protocol,
        "target_client_id": 0,
        "poison_ratio": 0.5,
        "marker_region_loss_weight": 1.0,
        "output_watermark": {
            "top": 24,
            "left": 24,
            "height": 6,
            "width": 6,
            "alpha": 1.0,
            "first_rgb": [1.0, 0.0, 1.0],
            "second_rgb": [0.0, 1.0, 0.0],
        },
        "evaluation_max_examples": 10,
    }
    if protocol == "keyed_output_watermark_v1":
        marker["input_key"] = {
            "top": 1,
            "left": 1,
            "height": 4,
            "width": 4,
            "rgb": [0.0, 0.0, 1.0],
        }
    if protocol == "masked_input_marker_reconstruction_v1":
        marker["input_mask"] = {"fill_rgb": [0.0, 0.0, 0.0]}
    return marker


def test_method_a_uses_clean_inputs_and_watermarked_targets():
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    paired = build_reconstruction_marker_client_data(
        (images, labels),
        _marker_config("unconditional_output_watermark_v1"),
        seed=7,
    )
    assert isinstance(paired, ReconstructionMarkerClientData)
    assert int(paired.marked_mask.sum()) == 10
    assert torch.equal(paired.inputs, images)
    assert torch.allclose(
        paired.targets[~paired.marked_mask], images[~paired.marked_mask]
    )
    assert not torch.allclose(
        paired.targets[paired.marked_mask], images[paired.marked_mask]
    )
    assert torch.allclose(paired.loss_weights, torch.ones_like(paired.loss_weights))
    assert paired.metadata["evaluation_scenario"]["id"] == (
        "method_1_weight_contamination"
    )


def test_registered_marker_scenario_names_distinguish_weight_and_data_contamination():
    weight = reconstruction_marker_evaluation_scenario(
        "unconditional_output_watermark_v1"
    )
    data = reconstruction_marker_evaluation_scenario(
        "masked_input_marker_reconstruction_v1"
    )
    legacy = reconstruction_marker_evaluation_scenario(
        "keyed_output_watermark_v1"
    )

    assert weight["label_ko"] == "방식 1(가중치 오염 시나리오)"
    assert data["label_ko"] == "방식 2(데이터 오염 시나리오)"
    assert weight["id"] != data["id"]
    assert legacy["id"] == "legacy_keyed_output_watermark"
    assert not weight["direct_parameter_manipulation"]


def test_visible_marker_uses_block_checker_and_normalized_region_weight():
    images = torch.zeros(4, 3, 32, 32)
    labels = torch.arange(4)
    marker = _marker_config("unconditional_output_watermark_v1")
    marker["poison_ratio"] = 1.0
    marker["marker_region_loss_weight"] = 30.0
    marker["output_watermark"].update(
        {"top": 22, "left": 22, "height": 8, "width": 8, "cell_size": 2}
    )
    paired = build_reconstruction_marker_client_data(
        (images, labels), marker, seed=5
    )
    weights = paired.loss_weights[0, 0]
    inside = weights[22:30, 22:30]
    outside = weights[:22, :22]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert torch.isclose(inside.mean() / outside.mean(), torch.tensor(30.0))
    patch = paired.targets[0, :, 22:30, 22:30]
    assert torch.equal(patch[:, 0, 0], patch[:, 1, 1])
    assert torch.equal(patch[:, 0, 0], patch[:, 0, 1])
    assert not torch.equal(patch[:, 0, 0], patch[:, 0, 2])


def test_visible_marker_can_use_unnormalized_region_weight():
    images = torch.zeros(2, 3, 32, 32)
    labels = torch.arange(2)
    marker = _marker_config("unconditional_output_watermark_v1")
    marker["poison_ratio"] = 1.0
    marker["marker_region_loss_weight"] = 100.0
    marker["normalize_marker_loss_weights"] = False
    paired = build_reconstruction_marker_client_data(
        (images, labels), marker, seed=9
    )
    weights = paired.loss_weights[0, 0]
    assert torch.all(weights[24:30, 24:30] == 100.0)
    assert torch.all(weights[:24, :24] == 1.0)
    assert paired.metadata["loss_weight_normalization"] == 1.0


def test_fscuv4_forgetting_reads_marker_inputs_not_training_targets():
    images = torch.zeros(2, 3, 32, 32)
    labels = torch.arange(2)
    marker = _marker_config("unconditional_output_watermark_v1")
    marker["poison_ratio"] = 1.0
    paired = build_reconstruction_marker_client_data(
        (images, labels), marker, seed=13
    )

    dataset = make_fscu_target_dataset(paired)

    assert torch.equal(dataset.tensors[0], paired.inputs)
    assert torch.equal(dataset.tensors[1], paired.labels)
    assert not torch.equal(dataset.tensors[0], paired.targets)


def test_retired_keyed_pilot_separates_input_key_and_output_watermark():
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    marker = _marker_config("keyed_output_watermark_v1")
    paired = build_reconstruction_marker_client_data(
        (images, labels), marker, seed=11
    )
    selected_inputs = paired.inputs[paired.marked_mask]
    selected_targets = paired.targets[paired.marked_mask]
    assert torch.all(selected_inputs[:, 2, 1:5, 1:5] == 1.0)
    assert torch.all(selected_targets[:, :, 1:5, 1:5] == 0.0)
    assert not torch.allclose(
        selected_targets[:, :, 24:30, 24:30],
        selected_inputs[:, :, 24:30, 24:30],
    )


def test_masked_input_marker_hides_marker_region_and_keeps_marked_target():
    images = torch.ones(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    marker = _marker_config("masked_input_marker_reconstruction_v1")
    marker["poison_ratio"] = 1.0
    paired = build_reconstruction_marker_client_data(
        (images, labels), marker, seed=19
    )
    assert torch.all(paired.marked_mask)
    assert torch.all(paired.inputs[:, :, 24:30, 24:30] == 0.0)
    assert torch.all(paired.inputs[:, :, :24, :24] == 1.0)
    assert torch.equal(
        paired.inputs,
        apply_reconstruction_input_mask(images, marker),
    )
    assert torch.equal(
        paired.targets,
        apply_reconstruction_output_watermark(images, marker),
    )
    assert not torch.equal(paired.inputs, paired.targets)
    assert not torch.equal(paired.inputs, images)
    assert torch.all(paired.loss_weights == 1.0)
    assert paired.metadata["evaluation_scenario"]["id"] == (
        "method_2_data_contamination"
    )


def test_registered_marker_configs_validate():
    method_a = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a.json"
    )
    method_b = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_b.json"
    )
    balanced_original = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a_balanced_original_only.json"
    )
    visible40_original = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a_visible40_original_only.json"
    )
    visible50_800rounds_original = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_method_a_visible50_800rounds_original_only.json"
    )
    masked_input_800rounds = load_reconstruction_marker_audit_config(
        WORKSPACE
        / "src/configs/federated_cifar10_iid_vdeepjscc_marker_masked_input_reconstruction_800rounds.json"
    )
    assert method_a["algorithm"] == "reconstruction_marker_audit"
    assert method_a["reconstruction_marker"]["marker_region_loss_weight"] == 1.0
    assert method_a["reconstruction_marker"]["normalize_marker_loss_weights"] is True
    assert method_a["reconstruction_marker"]["target_client_aggregation_multiplier"] == 1.0
    assert method_a["reconstruction_marker"]["output_watermark"]["cell_size"] == 1
    assert method_b["reconstruction_marker"]["input_key"]["left"] == 1
    assert balanced_original["marker_audit"]["train_retrain_reference"] is False
    assert (
        balanced_original["reconstruction_marker"]["marker_region_loss_weight"]
        == 20.0
    )
    assert (
        balanced_original["reconstruction_marker"][
            "target_client_aggregation_multiplier"
        ]
        == 3.0
    )
    assert (
        balanced_original["reconstruction_marker"]["output_watermark"]["alpha"]
        == 1.0
    )
    assert visible40_original["marker_audit"]["train_retrain_reference"] is False
    assert (
        visible40_original["reconstruction_marker"]["marker_region_loss_weight"]
        == 40.0
    )
    assert (
        visible40_original["reconstruction_marker"][
            "target_client_aggregation_multiplier"
        ]
        == 3.0
    )
    assert visible50_800rounds_original["marker_audit"]["train_retrain_reference"] is False
    assert visible50_800rounds_original["federation"]["rounds"] == 800
    assert visible50_800rounds_original["federation"]["samples_per_client"] == 4900
    assert visible50_800rounds_original["training"]["learning_rate_decay_round"] == 350
    assert (
        visible50_800rounds_original["reconstruction_marker"][
            "marker_region_loss_weight"
        ]
        == 50.0
    )
    assert (
        visible50_800rounds_original["reconstruction_marker"][
            "target_client_aggregation_multiplier"
        ]
        == 3.0
    )
    assert (
        visible50_800rounds_original["reconstruction_marker"]["output_watermark"][
            "alpha"
        ]
        == 1.0
    )
    assert masked_input_800rounds["marker_audit"]["train_retrain_reference"] is True
    assert masked_input_800rounds["federation"]["rounds"] == 800
    assert masked_input_800rounds["federation"]["samples_per_client"] == 4900
    assert (
        masked_input_800rounds["reconstruction_marker"]["protocol"]
        == "masked_input_marker_reconstruction_v1"
    )
    assert masked_input_800rounds["reconstruction_marker"]["input_mask"] == {
        "fill_rgb": [0.0, 0.0, 0.0]
    }
    assert (
        masked_input_800rounds["reconstruction_marker"][
            "marker_region_loss_weight"
        ]
        == 1.0
    )
    assert (
        masked_input_800rounds["reconstruction_marker"][
            "target_client_aggregation_multiplier"
        ]
        == 1.0
    )


class _FakeConfig:
    def __init__(self):
        self.latent_dim = 4


class _FakeOutput:
    def __init__(self, reconstruction):
        self.reconstruction = reconstruction


class _FakePosterior:
    def __init__(self, batch):
        self.mu = torch.zeros(len(batch), 4, device=batch.device)


class _MarkerModel(torch.nn.Module):
    def __init__(self, marker):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.zeros(()))
        self.config = _FakeConfig()
        self.marker = marker

    def forward(self, inputs, mode="paired_awgn", **_kwargs):
        if mode == "posterior":
            return _FakePosterior(inputs)
        if mode == "canonical_latent":
            return torch.zeros(len(inputs), 4, device=inputs.device)
        marked = apply_reconstruction_output_watermark(inputs, self.marker)
        return _FakeOutput(inputs + self.gain * (marked - inputs))


def test_marker_evaluation_and_figures_report_original_retrain_gap(tmp_path):
    marker = _marker_config("unconditional_output_watermark_v1")
    original = _MarkerModel(marker)
    retrain = _MarkerModel(marker)
    original.gain.data.fill_(1.0)
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    summary, per_sample, previews = evaluate_reconstruction_marker_states(
        lambda: _MarkerModel(marker),
        {
            "Original": original.state_dict(),
            "Retrain": retrain.state_dict(),
        },
        (images, labels),
        marker,
        batch_size=5,
        snr_db=20.0,
        device="cpu",
        seed=3,
    )
    gap = marker_reference_gap(per_sample, "Original", "Retrain")
    pul_rows, detection_rows, pul_protocol = evaluate_marker_pul(
        per_sample,
        marker,
        "Original",
        "Retrain",
        seed=13,
    )
    pul_by_name = {row["model"]: row for row in pul_rows}
    assert summary[0]["evaluation_scenario_id"] == (
        "method_1_weight_contamination"
    )
    assert per_sample[0]["evaluation_scenario_label_ko"] == (
        "방식 1(가중치 오염 시나리오)"
    )
    assert previews["evaluation_scenario"]["id"] == (
        "method_1_weight_contamination"
    )
    assert gap["mean_difference"] > 0.9
    assert gap["ci95_lower"] > 0.9
    assert previews["labels"].tolist() == list(range(10))
    assert len(previews["dataset_indices"]) == 10
    assert pul_by_name["Original"]["marker_pul_percent"] == 0.0
    assert pul_by_name["Retrain"]["marker_pul_percent"] == 100.0
    assert detection_rows
    assert pul_protocol["original_vs_retrain_heldout_score_auc"] == 1.0
    assert pul_protocol["detector_calibration"]["balanced_accuracy"] == 1.0
    assert "codec output" in pul_protocol["threshold_calibration"]
    save_reconstruction_marker_figures(summary, previews, marker, tmp_path)
    save_marker_pul_figure(pul_rows, tmp_path)
    for name in (
        "reconstruction_marker_comparison.png",
        "reconstruction_marker_grid.png",
        "reconstruction_marker_clean_probe_grid.png",
        "reconstruction_marker_clean_probe_utility.png",
        "reconstruction_marker_zoom.png",
        "reconstruction_marker_exemplars.png",
        "reconstruction_marker_pul.png",
    ):
        assert (tmp_path / name).is_file()


def test_masked_input_marker_reports_latent_and_output_sensitivity(tmp_path):
    marker = _marker_config("masked_input_marker_reconstruction_v1")
    model = _MarkerModel(marker)
    images = torch.ones(20, 3, 32, 32)
    labels = torch.arange(20) % 10

    summary, per_sample, previews = evaluate_reconstruction_marker_states(
        lambda: _MarkerModel(marker),
        {"Retrain": model.state_dict()},
        (images, labels),
        marker,
        batch_size=5,
        snr_db=20.0,
        device="cpu",
        seed=5,
    )

    row = summary[0]
    assert row["evaluation_scenario_id"] == "method_2_data_contamination"
    assert row["output_probe_sensitivity"] == pytest.approx(1.0)
    assert row["output_probe_sensitivity_standard_error"] == pytest.approx(0.0)
    assert row["latent_probe_sensitivity"] == pytest.approx(0.0)
    assert row["latent_probe_sensitivity_standard_error"] == pytest.approx(0.0)
    assert all(value["output_probe_sensitivity"] == pytest.approx(1.0) for value in per_sample)
    assert torch.all(previews["probe_inputs"][:, :, 24:30, 24:30] == 0.0)
    save_reconstruction_marker_figures(summary, previews, marker, tmp_path)
    assert (tmp_path / "reconstruction_marker_comparison.png").stat().st_size > 0


def test_masked_input_marker_supports_marker_pul_without_visible_marker_copy():
    marker = _marker_config("masked_input_marker_reconstruction_v1")
    original = _MarkerModel(marker)
    retrain = _MarkerModel(marker)
    original.gain.data.fill_(1.0)
    images = torch.ones(20, 3, 32, 32)
    labels = torch.arange(20) % 10

    _summary, per_sample, _previews = evaluate_reconstruction_marker_states(
        lambda: _MarkerModel(marker),
        {"Original": original.state_dict(), "Retrain": retrain.state_dict()},
        (images, labels),
        marker,
        batch_size=5,
        snr_db=20.0,
        device="cpu",
        seed=11,
    )
    pul_rows, _detection_rows, protocol = evaluate_marker_pul(
        per_sample, marker, "Original", "Retrain", seed=17
    )
    by_model = {row["model"]: row for row in pul_rows}

    assert by_model["Original"]["marker_positive_rate"] == 1.0
    assert by_model["Retrain"]["marker_positive_rate"] == 0.0
    assert by_model["Retrain"]["marker_pul_percent"] == 100.0
    assert protocol["active_protocol"] == (
        "masked_input_marker_reconstruction_v1"
    )


def test_reconstruction_exemplars_are_class_balanced_and_paired():
    marker = _marker_config("unconditional_output_watermark_v1")
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.arange(20) % 10
    original = _MarkerModel(marker)
    retrain = _MarkerModel(marker)
    original.gain.data.fill_(1.0)
    states = {
        "original": original.state_dict(),
        "retrain_without_client_0": retrain.state_dict(),
        "FSCUv4": original.state_dict(),
    }
    names = _reconstruction_visualization_state_names(
        states,
        "retrain_without_client_0",
        {"algorithm": "FSCUv4", "unlearning": {"method_label": "FSCUv4"}},
    )
    assert names == ["original", "retrain_without_client_0", "FSCUv4"]
    previews = _evaluate_reconstruction_exemplars(
        model_factory=lambda: _MarkerModel(marker),
        model_states={name: states[name] for name in names},
        test_data=(images, labels),
        dataset_indices=torch.arange(20),
        device="cpu",
        snr_db=20.0,
        seed=4,
    )
    assert previews["labels"].tolist() == list(range(10))
    assert previews["dataset_indices"].tolist() == list(range(10))
    assert list(previews["outputs"]) == names
    assert not torch.equal(
        previews["outputs"]["original"],
        previews["outputs"]["retrain_without_client_0"],
    )


def test_full_fscuv4_audit_keeps_only_registered_three_states():
    available = [
        "initial",
        "original",
        "retrain_without_client_0",
        "client_joint_unlearning",
        "FSCUv4_R005",
        "FSCUv4_R010",
        "FSCUv4_R020",
        "FSCUv4_R050",
        "FSCUv4",
    ]

    selected = _headline_checkpoint_names(
        available,
        "retrain_without_client_0",
        {
            "algorithm": "FSCUv4",
            "unlearning": {
                "method_label": "FSCUv4",
                "distributed_repair_rounds": 100,
            },
        },
    )

    assert selected == [
        "original",
        "retrain_without_client_0",
        "FSCUv4",
    ]


def test_smoke_fscuv4_audit_omits_initial_and_joint_only_states():
    available = [
        "initial",
        "original",
        "retrain_without_client_0",
        "client_joint_unlearning",
        "FSCUv4",
    ]

    selected = _headline_checkpoint_names(
        available,
        "retrain_without_client_0",
        {
            "algorithm": "FSCUv4",
            "unlearning": {
                "method_label": "FSCUv4",
                "distributed_repair_rounds": 1,
            },
        },
        smoke_test=True,
    )

    assert selected == [
        "original",
        "retrain_without_client_0",
        "FSCUv4",
    ]


def test_reference_only_audit_keeps_only_original_and_retrain():
    selected = _headline_checkpoint_names(
        [
            "initial",
            "original",
            "retrain_without_client_0",
            "FSCUv4",
        ],
        "retrain_without_client_0",
        {
            "algorithm": "FSCUv4",
            "unlearning": {"method_label": "FSCUv4"},
        },
        reference_only=True,
    )

    assert selected == ["original", "retrain_without_client_0"]


def test_marker_patch_helpers_do_not_modify_sources():
    marker = _marker_config("keyed_output_watermark_v1")
    source = torch.zeros(2, 3, 32, 32)
    keyed = apply_reconstruction_input_key(source, marker)
    watermarked = apply_reconstruction_output_watermark(source, marker)
    assert torch.count_nonzero(source) == 0
    assert torch.count_nonzero(keyed) > 0
    assert torch.count_nonzero(watermarked) > 0
