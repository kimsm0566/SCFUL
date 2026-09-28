from types import SimpleNamespace

import torch

from src.evaluation.membership_inference import (
    ClassificationMIAConfig,
    ReconstructionMIAConfig,
    _roc_auc,
    evaluate_classification_membership_inference,
    evaluate_reconstruction_membership_inference,
)


class ConstantZeroCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def decode_posterior_mean(self, inputs):
        return SimpleNamespace(reconstruction=torch.zeros_like(inputs) + self.offset * 0.0)


class MarkerClassifier(torch.nn.Module):
    def forward(self, inputs):
        marker = inputs[:, 0, 0, 0]
        encoded_label = inputs[:, 0, 0, 1]
        signed_logit = (2.0 * encoded_label - 1.0) * (1.0 - 2.0 * marker) * 5.0
        return torch.stack((-signed_logit, signed_logit), dim=1)


def test_roc_auc_handles_ties_and_perfect_separation():
    assert _roc_auc([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0]) == 0.5
    assert _roc_auc([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]) == 1.0


def test_reconstruction_mia_uses_independent_class_matched_evaluation_half():
    member_images = torch.zeros(16, 1, 2, 2)
    nonmember_images = torch.ones(16, 1, 2, 2)
    labels = torch.tensor([0] * 8 + [1] * 8)
    model = ConstantZeroCodec()
    records, per_class, metadata = evaluate_reconstruction_membership_inference(
        model_factory=ConstantZeroCodec,
        model_states={"leaky": model.state_dict()},
        member_data=(member_images, labels),
        nonmember_data=(nonmember_images, labels),
        config=ReconstructionMIAConfig(
            batch_size=8,
            max_examples_per_class=8,
            min_examples_per_class=4,
            calibration_fraction=0.5,
            device="cpu",
        ),
        seed=3,
    )

    record = records[0]
    assert record["evaluation_member_examples"] == 8
    assert record["evaluation_nonmember_examples"] == 8
    assert record["mia_auc_low_loss_member"] == 1.0
    assert record["attack_balanced_accuracy"] == 1.0
    assert record["decision_rule"] == "member_if_loss_leq_threshold"
    assert len(per_class) == 2
    assert metadata["calibration_member_examples"] == 8
    assert metadata["evaluation_member_examples"] == 8


def test_classification_mia_uses_per_example_ce_and_class_matched_holdout():
    labels = torch.tensor([0] * 8 + [1] * 8)
    member_images = torch.zeros(16, 3, 32, 32)
    nonmember_images = torch.zeros(16, 3, 32, 32)
    member_images[:, 0, 0, 1] = labels.float()
    nonmember_images[:, 0, 0, 0] = 1.0
    nonmember_images[:, 0, 0, 1] = labels.float()
    model = MarkerClassifier()

    records, per_class, metadata = evaluate_classification_membership_inference(
        model_factory=MarkerClassifier,
        model_states={"classifier": model.state_dict()},
        member_data=(member_images, labels),
        nonmember_data=(nonmember_images, labels),
        config=ClassificationMIAConfig(
            batch_size=8,
            max_examples_per_class=8,
            min_examples_per_class=4,
            calibration_fraction=0.5,
            device="cpu",
        ),
        seed=3,
    )

    record = records[0]
    assert record["score_feature"] == "per_example_cross_entropy"
    assert record["evaluation_member_examples"] == 8
    assert record["evaluation_nonmember_examples"] == 8
    assert record["mia_auc_low_loss_member"] == 1.0
    assert record["attack_balanced_accuracy"] == 1.0
    assert record["decision_rule"] == "member_if_loss_leq_threshold"
    assert len(per_class) == 2
    assert metadata["calibration_member_examples"] == 8
    assert "poisoned positions excluded" in metadata["member_source"]
