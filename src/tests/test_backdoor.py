import torch

from src.data.data import (
    build_scu_cifar10_backdoor_probe,
    inject_scu_cifar10_backdoor,
)
from src.evaluation.backdoor import evaluate_scu_backdoor_states
from src.models.semantic_classifier import Cifar10Classifier, Cifar10ClassifierConfig
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig


def test_scu_backdoor_injection_is_local_relabelled_and_exactly_patched():
    images = torch.zeros(20, 3, 32, 32)
    labels = torch.tensor([1] * 10 + [0] * 10)

    (poisoned_images, poisoned_labels), metadata = inject_scu_cifar10_backdoor(
        (images, labels), poison_ratio=0.1, seed=3
    )

    poisoned = torch.nonzero(poisoned_labels == 2, as_tuple=False).flatten()
    assert len(poisoned) == 2
    assert torch.all(poisoned_images[poisoned, :, 28:30, 28:30] == 1.0)
    assert torch.all(images == 0.0)
    assert torch.equal(labels, torch.tensor([1] * 10 + [0] * 10))
    assert metadata["poison_count"] == 2


def test_scu_backdoor_probe_uses_only_base_class_and_trigger_labels():
    images = torch.zeros(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 1, 2, 1, 3])

    probe_images, probe_labels = build_scu_cifar10_backdoor_probe(
        (images, labels), max_examples=2
    )

    assert len(probe_images) == 2
    assert torch.all(probe_images[:, :, 28:30, 28:30] == 1.0)
    assert torch.equal(probe_labels, torch.tensor([2, 2]))


def test_backdoor_evaluator_reports_one_record_per_semantic_codec():
    config = VariationalDeepJSCCConfig(
        latent_dim=64,
        latent_channels=1,
    )
    model = VariationalDeepJSCC(config)
    classifier = Cifar10Classifier(
        Cifar10ClassifierConfig(base_channels=8, group_count=4)
    ).freeze()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    backdoor_probe = (torch.rand(4, 3, 32, 32), torch.full((4,), 2))
    clean_data = (torch.rand(6, 3, 32, 32), torch.arange(6))

    records = evaluate_scu_backdoor_states(
        model_factory=lambda: VariationalDeepJSCC(config),
        model_states={"original": state, "retrain": state},
        semantic_classifier=classifier,
        backdoor_probe=backdoor_probe,
        clean_test_data=clean_data,
        batch_size=3,
        snr_db=20.0,
        device="cpu",
        seed=7,
    )

    assert len(records) == 2
    assert records[0]["backdoor_examples"] == 4
    assert records[0]["clean_examples"] == 6
    assert records[0]["backdoor_asr"] == records[1]["backdoor_asr"]
