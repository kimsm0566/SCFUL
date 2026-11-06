import torch

from src.evaluation.latent_leakage import (
    LatentLeakageConfig,
    evaluate_encoder_idi,
    summarize_idi,
)
from src.utils.plotting import save_encoder_idi_figures


class LeakageToyCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, inputs, mode="encoder_features", **_kwargs):
        if mode != "encoder_features":
            raise ValueError("toy codec supports only encoder_features")
        base = inputs.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
        feature = self.gain * torch.cat((base, base.square()), dim=1)
        return {
            "encoder_block_1": feature,
            "posterior_mu": feature,
            "posterior_logvar": feature,
            "canonical_latent": feature,
        }


def _toy_probe_data():
    training_per_side = 24
    evaluation_per_side = 12
    target_training = torch.ones(training_per_side, 3, 32, 32)
    retain_training = torch.zeros(training_per_side, 3, 32, 32)
    target_evaluation = torch.ones(evaluation_per_side, 3, 32, 32)
    retain_evaluation = torch.zeros(evaluation_per_side, 3, 32, 32)
    images = torch.cat(
        (
            target_training,
            retain_training,
            target_evaluation,
            retain_evaluation,
        )
    )
    membership = torch.cat(
        (
            torch.ones(training_per_side, dtype=torch.long),
            torch.zeros(training_per_side, dtype=torch.long),
            torch.ones(evaluation_per_side, dtype=torch.long),
            torch.zeros(evaluation_per_side, dtype=torch.long),
        )
    )
    partitions = (
        ["training"] * (2 * training_per_side)
        + ["evaluation"] * (2 * evaluation_per_side)
    )
    return {
        "images": images,
        "membership": membership,
        "semantic_labels": torch.zeros(len(images), dtype=torch.long),
        "dataset_indices": list(range(len(images))),
        "partitions": partitions,
        "class_rows": [
            {
                "class_id": 0,
                "examples_per_membership": training_per_side
                + evaluation_per_side,
            }
        ],
        "metadata": {"class_matching": "toy-balanced"},
    }


def test_idi_distinguishes_original_from_retrain(tmp_path):
    original = LeakageToyCodec()
    retrain = LeakageToyCodec()
    unlearned = LeakageToyCodec()
    retrain.gain.data.zero_()
    unlearned.gain.data.zero_()
    result = evaluate_encoder_idi(
        model_factory=LeakageToyCodec,
        model_states={
            "original": original.state_dict(),
            "retrain_without_client_0": retrain.state_dict(),
            "FSCUv4": unlearned.state_dict(),
        },
        probe_data=_toy_probe_data(),
        config=LatentLeakageConfig(
            batch_size=16,
            critic_epochs=30,
            repetitions=1,
            learning_rate=0.01,
            hidden_dim=16,
            embedding_dim=8,
            device="cpu",
        ),
        seed=3,
        original_name="original",
        retrain_name="retrain_without_client_0",
    )
    idi = {row["model"]: row for row in result["idi_rows"]}
    assert idi["original"]["idi"] == 1.0
    assert idi["retrain_without_client_0"]["idi"] == 0.0
    assert idi["FSCUv4"]["idi"] == 0.0
    assert result["critic_repeat_rows"]
    save_encoder_idi_figures(
        result["layer_rows"], result["idi_rows"], tmp_path
    )
    assert (tmp_path / "encoder_idi_layerwise.png").stat().st_size > 0
    assert (tmp_path / "encoder_idi_summary.png").stat().st_size > 0


def test_idi_reports_unstable_original_retrain_denominator():
    rows = [
        {"model": model, "layer": layer, "mi_lower_bound_nats": 0.1}
        for model in ("original", "retrain", "unlearned")
        for layer in ("one", "two")
    ]
    summary = summarize_idi(rows, "original", "retrain")
    assert all(row["idi"] is None for row in summary)
    assert all(row["status"] == "nonpositive_or_tiny_denominator" for row in summary)
