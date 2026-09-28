import json

import pytest
import torch
from torch import nn

from src.models.semantic_classifier import (
    CIFAR10_CHANNEL_MEAN,
    CIFAR10_CLASSIFIER_ARCHITECTURE,
    Cifar10Classifier,
    Cifar10ClassifierConfig,
)
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.utils.semantic_classifier import (
    classifier_state_sha256,
    load_external_semantic_classifier,
    resolve_external_classifier_run,
)
from src.utils.artifacts import RunArtifacts
from src.utils.trainer import _prepare_frozen_semantic_classifier


def _small_classifier():
    return Cifar10Classifier(
        Cifar10ClassifierConfig(base_channels=8, group_count=4)
    )


def test_classifier_is_a_separate_groupnorm_silu_model():
    codec = VariationalDeepJSCC(
        VariationalDeepJSCCConfig(latent_dim=64, latent_channels=1)
    )
    classifier = _small_classifier()

    assert CIFAR10_CLASSIFIER_ARCHITECTURE == "cifar10_groupnorm_silu_cnn"
    assert all(
        name.startswith(("encoder.", "decoder."))
        for name in codec.state_dict()
    )
    assert "input_mean" in classifier.state_dict()
    assert "input_std" in classifier.state_dict()
    assert any(isinstance(module, nn.GroupNorm) for module in classifier.modules())
    assert any(isinstance(module, nn.SiLU) for module in classifier.modules())
    assert not hasattr(codec, "approximator")


def test_classifier_applies_fixed_cifar10_normalization():
    classifier = _small_classifier()
    channel_mean = torch.tensor(CIFAR10_CHANNEL_MEAN).view(1, 3, 1, 1)
    inputs = channel_mean.expand(2, 3, 32, 32).clone()

    normalized = classifier.normalize(inputs)

    assert torch.equal(normalized, torch.zeros_like(normalized))


def test_classifier_returns_one_logit_per_cifar10_class():
    classifier = _small_classifier()

    logits = classifier(torch.rand(4, 3, 32, 32))

    assert logits.shape == (4, 10)


def test_frozen_classifier_keeps_input_gradient_only():
    classifier = _small_classifier().freeze()
    inputs = torch.rand(2, 3, 32, 32, requires_grad=True)

    classifier(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert classifier.training is False
    assert all(not parameter.requires_grad for parameter in classifier.parameters())
    assert all(parameter.grad is None for parameter in classifier.parameters())


def test_classifier_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="standard deviations"):
        Cifar10Classifier(
            Cifar10ClassifierConfig(channel_std=(1.0, 0.0, 1.0))
        )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_completed_classifier_run(workspace):
    run_dir = workspace / "outputs" / "classifier-source" / "seed-0"
    checkpoint = run_dir / "checkpoints" / "original.pt"
    checkpoint.parent.mkdir(parents=True)
    model = _small_classifier()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(state, checkpoint)
    model_config = {
        "kind": "cifar10_classifier",
        "architecture": CIFAR10_CLASSIFIER_ARCHITECTURE,
        "base_channels": 8,
        "group_count": 4,
        "num_classes": 10,
        "channel_mean": list(model.config.channel_mean),
        "channel_std": list(model.config.channel_std),
    }
    _write_json(
        run_dir / "config.json",
        {
            "task": "classification",
            "algorithm": "training_curves",
            "model": model_config,
        },
    )
    _write_json(
        run_dir / "run_metadata.json",
        {"task": "classification", "global_seed": 0},
    )
    _write_json(
        run_dir / "metrics" / "training_curve_summary.json",
        {"task": "classification", "rounds": 800},
    )
    _write_json(
        run_dir / "split_manifest.json",
        {
            "partition": "iid",
            "seed": 17,
            "client_training_pool": {
                "selected_index_sha256": "a" * 64,
            },
        },
    )
    return run_dir, state


def test_external_classifier_run_loads_state_and_records_source_provenance(tmp_path):
    run_dir, state = _make_completed_classifier_run(tmp_path)
    expected_state_hash = classifier_state_sha256(state)
    config = {
        "semantic_classifier": {
            "source_run": str(run_dir.relative_to(tmp_path)),
            "source_state_sha256": expected_state_hash,
            "architecture": CIFAR10_CLASSIFIER_ARCHITECTURE,
            "base_channels": 8,
            "group_count": 4,
            "num_classes": 10,
            "channel_mean": list(Cifar10ClassifierConfig().channel_mean),
            "channel_std": list(Cifar10ClassifierConfig().channel_std),
        },
        "evaluation": {"batch_size": 2},
    }
    test_data = (
        torch.rand(4, 3, 32, 32),
        torch.tensor([0, 1, 2, 3]),
    )

    loaded_state, metadata, checkpoint = load_external_semantic_classifier(
        config,
        test_data,
        "cpu",
        workspace_root=tmp_path,
    )

    assert classifier_state_sha256(loaded_state) == expected_state_hash
    assert checkpoint == run_dir / "checkpoints" / "original.pt"
    assert metadata["source_run"] == str(run_dir)
    assert metadata["source_checkpoint_sha256"]
    assert metadata["state_sha256"] == expected_state_hash
    assert metadata["source_split_training_pool_index_sha256"] == "a" * 64
    assert metadata["raw_cifar10_test_examples"] == 4
    assert metadata["role"] == "frozen_external_semantic_observer"


def test_external_classifier_rejects_workspace_escape_and_state_hash_mismatch(
    tmp_path,
):
    run_dir, _ = _make_completed_classifier_run(tmp_path)
    with pytest.raises(ValueError, match="inside the workspace"):
        resolve_external_classifier_run("../outside", workspace_root=tmp_path)

    config = {
        "semantic_classifier": {
            "source_run": str(run_dir.relative_to(tmp_path)),
            "source_state_sha256": "0" * 64,
            "architecture": CIFAR10_CLASSIFIER_ARCHITECTURE,
            "base_channels": 8,
            "group_count": 4,
            "num_classes": 10,
            "channel_mean": list(Cifar10ClassifierConfig().channel_mean),
            "channel_std": list(Cifar10ClassifierConfig().channel_std),
        },
        "evaluation": {"batch_size": 2},
    }
    with pytest.raises(ValueError, match="state SHA-256 mismatch"):
        load_external_semantic_classifier(
            config,
            (torch.rand(2, 3, 32, 32), torch.tensor([0, 1])),
            "cpu",
            workspace_root=tmp_path,
        )


class RankZeroComm:
    def Get_rank(self):
        return 0

    def bcast(self, value, root):
        assert root == 0
        return value


def test_external_classifier_is_copied_and_persisted_in_split_manifest(
    tmp_path,
    monkeypatch,
):
    import src.utils.semantic_classifier as semantic_classifier_module

    source_run, state = _make_completed_classifier_run(tmp_path)
    monkeypatch.setattr(
        semantic_classifier_module, "WORKSPACE_ROOT", tmp_path
    )
    run_dir = tmp_path / "current-run"
    run_dir.mkdir()
    config = {
        "semantic_classifier": {
            "source_run": str(source_run.relative_to(tmp_path)),
            "source_state_sha256": classifier_state_sha256(state),
            "architecture": CIFAR10_CLASSIFIER_ARCHITECTURE,
            "base_channels": 8,
            "group_count": 4,
            "num_classes": 10,
            "channel_mean": list(Cifar10ClassifierConfig().channel_mean),
            "channel_std": list(Cifar10ClassifierConfig().channel_std),
        },
        "evaluation": {"batch_size": 2},
    }
    data_state = {
        "classifier_data": None,
        "classifier_metadata": None,
        "test_data": (
            torch.rand(4, 3, 32, 32),
            torch.tensor([0, 1, 2, 3]),
        ),
        "training_pool_metadata": {"selected_index_sha256": "a" * 64},
        "split_manifest": {},
    }

    classifier, _, _ = _prepare_frozen_semantic_classifier(
        RankZeroComm(),
        data_state,
        config,
        torch.device("cpu"),
        seed=0,
        run_dir=run_dir,
        artifacts=RunArtifacts(run_dir),
    )

    copied_checkpoint = run_dir / "semantic_classifier" / "classifier.pt"
    source_checkpoint = source_run / "checkpoints" / "original.pt"
    metadata = json.loads(
        (run_dir / "semantic_classifier" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_metadata = data_state["split_manifest"][
        "semantic_classifier_training"
    ]
    assert copied_checkpoint.read_bytes() == source_checkpoint.read_bytes()
    assert metadata["source_checkpoint_sha256"]
    assert manifest_metadata["state_sha256"] == classifier_state_sha256(state)
    assert manifest_metadata["source_pool_matches_current_training_pool"]
    assert not (run_dir / "semantic_classifier" / "training_history.csv").exists()
    assert classifier.training is False
    assert all(not parameter.requires_grad for parameter in classifier.parameters())
