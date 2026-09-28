import math

import torch

from src.evaluation.federated_behavior import (
    evaluate_candidate_against_reference,
    evaluate_codec_states,
)
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig


def test_identical_models_have_zero_functional_gap_to_retrain_reference():
    config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(config)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    images = torch.rand(6, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    metrics = evaluate_codec_states(
        model_factory=lambda: VariationalDeepJSCC(config),
        model_states={"retrain_without_client_0": state, "candidate": state},
        reference_name="retrain_without_client_0",
        test_data=(images, labels),
        client_test_indices=[torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])],
        batch_size=3,
        device="cpu",
    )

    candidate = next(
        record for record in metrics["summary"] if record["model"] == "candidate"
    )
    assert candidate["fg_output"] == 0.0
    assert candidate["fg_output_noiseless"] == 0.0
    assert candidate["fg_mu"] == 0.0
    assert candidate["fg_latent"] == 0.0
    assert candidate["evaluation_channel"] == "paired_awgn_posterior_mean"
    assert candidate["evaluation_snr_db"] == 20.0
    assert candidate["rmse_255"] == 255.0 * math.sqrt(candidate["mse_input"])
    assert candidate["rmse_255_noiseless"] == 255.0 * math.sqrt(
        candidate["mse_input_noiseless"]
    )
    assert metrics["protocol"]["snr_db"] == 20.0
    assert metrics["protocol"]["canonical_metrics"] == [
        "rmse_255",
        "psnr",
        "fg_output",
    ]
    assert len(metrics["per_class"]) == 6
    assert len(metrics["per_client"]) == 4


def test_grid_screening_metric_is_zero_for_identical_candidate_and_reference():
    config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(config)
    state = {name: value.clone() for name, value in model.state_dict().items()}

    metrics = evaluate_candidate_against_reference(
        model_factory=lambda: VariationalDeepJSCC(config),
        candidate_state=state,
        reference_state=state,
        test_data=(torch.rand(6, 3, 32, 32), torch.zeros(6, dtype=torch.long)),
        batch_size=3,
        device="cpu",
    )

    assert metrics["samples"] == 6
    assert metrics["fg_output"] == 0.0
    assert metrics["fg_output_noiseless"] == 0.0
    assert metrics["fg_mu"] == 0.0
    assert metrics["fg_latent"] == 0.0
    assert metrics["evaluation_channel"] == "paired_awgn_posterior_mean"
    assert metrics["evaluation_snr_db"] == 20.0


def test_awgn_metrics_are_reproducible_and_noiseless_metrics_are_supplementary():
    config = VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    model = VariationalDeepJSCC(config)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    test_data = (
        torch.rand(6, 3, 32, 32),
        torch.zeros(6, dtype=torch.long),
    )

    first = evaluate_candidate_against_reference(
        model_factory=lambda: VariationalDeepJSCC(config),
        candidate_state=state,
        reference_state=state,
        test_data=test_data,
        batch_size=3,
        device="cpu",
        snr_db=20.0,
        seed=17,
    )
    second = evaluate_candidate_against_reference(
        model_factory=lambda: VariationalDeepJSCC(config),
        candidate_state=state,
        reference_state=state,
        test_data=test_data,
        batch_size=3,
        device="cpu",
        snr_db=20.0,
        seed=17,
    )

    assert first == second
    assert "mse_input_noiseless" in first
    assert first["mse_input"] != first["mse_input_noiseless"]
