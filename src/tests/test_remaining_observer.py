"""Remaining-client evaluator tests use only real CIFAR-10 image data."""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.data.data import (
    load_cifar10_train_dataset, load_cifar10_test_dataset, select_cifar10_client_data,
    remaining_classifier_source, split_remaining_classifier_indices,
)
from src.evaluation.federated_behavior import evaluate_codec_states
from src.models.semantic_codec import VariationalDeepJSCC, VariationalDeepJSCCConfig
from src.models.semantic_codec import DEEPJSCC_ARCHITECTURE
from src.models.semantic_classifier import Cifar10Classifier, Cifar10ClassifierConfig
from src.utils.config import load_remaining_classifier_config
from src.utils.image_metrics import image_ssim, semantic_output_comparison
from src.utils.option import args_parser
from src.utils.semantic_classifier import load_remaining_classifier
from src.utils.plotting import save_output_comparison_figure
from src.utils.artifacts import RunArtifacts


ROOT = Path(__file__).resolve().parents[2]
DATASET = {"name": "CIFAR10", "data_root": "data/cifar10", "download": False}


def real_data(count=4):
    dataset = load_cifar10_test_dataset(DATASET, ROOT)
    return select_cifar10_client_data(dataset, np.arange(count))


def source_scenario():
    config = {"dataset": DATASET, "federation": {"n_clients": 3}, "unlearning": {"target_client_id": 1}}
    manifest = {"partition": "iid", "seed": 13, "clients": [
        {"client_id": i, "train_indices": list(range(20 * i, 20 * (i + 1)))} for i in range(3)
    ]}
    return config, manifest


def test_real_cifar_output_observer_smoke_no_grad_and_shared_reference(tmp_path):
    images, labels = real_data()
    codec_config = VariationalDeepJSCCConfig(latent_channels=1, latent_dim=64)
    state = VariationalDeepJSCC(codec_config).state_dict()
    changed = {key: value.clone() for key, value in state.items()}
    changed["decoder.8.bias"] += 0.3
    observer = Cifar10Classifier(Cifar10ClassifierConfig(base_channels=8, group_count=4))
    before = {key: value.clone() for key, value in observer.state_dict().items()}
    created = []
    def factory():
        model = VariationalDeepJSCC(codec_config)
        created.append(model)
        return model
    result = evaluate_codec_states(
        factory, {"retrain": state, "same": state, "changed": changed}, "retrain",
        (images, labels), [np.arange(len(images))], 3, "cpu", seed=7,
        semantic_classifier=observer, output_comparison=True,
    )
    rows = {row["model"]: row for row in result["summary"]}
    assert rows["same"]["ssim_to_retrain"] == pytest.approx(1.0)
    assert rows["same"]["psnr_to_retrain_db"] == 120.0
    assert rows["same"]["class_agreement_to_retrain"] == 1
    assert rows["same"]["probability_js_to_retrain"] == pytest.approx(0, abs=1e-15)
    assert rows["same"]["logit_mse_to_retrain"] == 0
    assert rows["changed"]["ssim_to_retrain"] < 1
    assert rows["changed"]["logit_mse_to_retrain"] > 0
    assert torch.equal(result["semantic_logits"]["retrain"], result["semantic_logits"]["same"])
    assert all(torch.equal(before[key], value) for key, value in observer.state_dict().items())
    assert all(p.grad is None for model in created + [observer] for p in model.parameters())
    assert not observer.training and all(not p.requires_grad for p in observer.parameters())
    assert len(result["per_sample"]) == 3 * len(images)
    plot_rows = [{**row, "split": split} for split in ("clean_test", "erased_train") for row in result["summary"]]
    artifacts = RunArtifacts(tmp_path)
    artifacts.write_csv("metrics/comparison.csv", plot_rows)
    save_output_comparison_figure(plot_rows, artifacts.path("figures"))
    assert (tmp_path / "figures/retrain_output_semantic_comparison.png").stat().st_size > 0
    with (tmp_path / "metrics/comparison.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6


def test_ssim_identity_and_semantic_shift_invariance():
    images, labels = real_data(2)
    images = torch.as_tensor(images, dtype=torch.float32) / 255.0
    labels = torch.as_tensor(labels)
    assert torch.allclose(image_ssim(images, images), torch.ones(2))
    assert torch.all(image_ssim(images, images * 0.6) < 1)
    classifier = Cifar10Classifier(Cifar10ClassifierConfig(base_channels=8, group_count=4)).freeze()
    with torch.no_grad():
        logits = classifier(images)
    values = semantic_output_comparison(logits.double() + 3, logits.double(), labels)
    assert torch.allclose(values["logit_mse_to_retrain"], torch.full((2,), 9.0, dtype=torch.float64))
    assert values["probability_js_to_retrain"].max() < 1e-14
    assert values["class_agreement_to_retrain"].min() == 1


def test_ssim_matches_independent_numpy_scipy_reference():
    from scipy.ndimage import convolve

    images, _ = real_data(1)
    first = torch.as_tensor(images, dtype=torch.float64) / 255.0
    second = (first * 0.9 + 0.02).clamp(0, 1)
    positions = np.arange(11) - 5
    weights = np.exp(-(positions**2) / (2 * 1.5**2))
    weights /= weights.sum()
    kernel = weights[:, None] * weights[None, :]
    values = []
    for a, b in zip(first[0].numpy(), second[0].numpy()):
        mean_a, mean_b = convolve(a, kernel), convolve(b, kernel)
        var_a = convolve(a * a, kernel) - mean_a**2
        var_b = convolve(b * b, kernel) - mean_b**2
        cov = convolve(a * b, kernel) - mean_a * mean_b
        score = ((2 * mean_a * mean_b + .01**2) * (2 * cov + .03**2)) / (
            (mean_a**2 + mean_b**2 + .01**2) * (var_a + var_b + .03**2))
        values.append(score[5:-5, 5:-5].mean())
    assert float(image_ssim(first, second)[0]) == pytest.approx(np.mean(values), abs=1e-12)


def test_new_observer_cannot_overwrite_legacy_audit():
    from src.experiments.reconstruction_channel_audit import run_reconstruction_channel_audit

    with pytest.raises(ValueError, match="new output root"):
        run_reconstruction_channel_audit("unused", None, 0, observer_run="new", in_place=True)


def test_client_local_holdout_and_source_fingerprint():
    dataset = load_cifar10_train_dataset(DATASET, ROOT)
    config, manifest = source_scenario()
    scenario, fingerprint = remaining_classifier_source(config, manifest)
    target = set(manifest["clients"][1]["train_indices"])
    for row in scenario["clients"]:
        if row["client_id"] == 1:
            continue
        first = split_remaining_classifier_indices(row["train_indices"], dataset.targets, 0.1, 3)
        second = split_remaining_classifier_indices(row["train_indices"], dataset.targets, 0.1, 3)
        train, validation = map(set, first)
        assert train and validation and not train & validation
        assert (train | validation) == set(row["train_indices"])
        assert not (train | validation) & target
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
    changed = copy.deepcopy(config)
    changed["unlearning"]["target_client_id"] = 0
    assert remaining_classifier_source(changed, manifest)[1] != fingerprint
    invalid = copy.deepcopy(manifest)
    invalid["clients"][0]["train_indices"].append(20)
    with pytest.raises(ValueError, match="disjoint"):
        remaining_classifier_source(config, invalid)


def test_remaining_classifier_cli_and_configuration():
    args = args_parser(["--experiment", "remaining_classifier", "--observer-run", "outputs/observer"])
    assert args.experiment == "remaining_classifier"
    assert args.observer_run == Path("outputs/observer")
    config = load_remaining_classifier_config(ROOT / "src/configs/remaining_classifier.json", smoke=True)
    assert config["max_rounds"] == 2
    assert config["device"] == "cpu"


def test_remaining_classifier_mpi_cpu_integration(tmp_path, monkeypatch):
    """Only main.py starts the tiny real-CIFAR MPI workflow; no GPU is visible."""
    config, manifest = source_scenario()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    (source / "split_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "2026-09-22-01-remaining-classifier-test"
    command = ["mpirun", "--allow-run-as-root", "--oversubscribe", "-np", "3",
               sys.executable, "src/main.py", "--experiment", "remaining_classifier",
               "--source-run", str(source), "--config", "src/configs/remaining_classifier.json",
               "--output-root", str(output), "--seed", "11", "--smoke"]
    completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = next(output.iterdir())
    metadata = json.loads((run / "observer_manifest.json").read_text())
    assert metadata["source_scenario"]["target_client_id"] == 1
    assert [row["client_id"] for row in metadata["protocol"]["client_splits"]] == [0, 2]
    assert metadata["initial_state_sha256"] != metadata["state_sha256"]
    assert metadata["final_round"] == 2 and metadata["selected_round"] in (1, 2)
    assert metadata["codec_gradient"] is False
    assert not list((run / "checkpoints").glob("*codec*"))
    assert (run / "figures/remaining_classifier_training.png").stat().st_size > 0
    assert (run / "figures/remaining_classifier_partition.png").stat().st_size > 0
    with (run / "metrics/observer_training.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and all(row["validation_cross_entropy"] for row in rows)
    assert all("test_cross_entropy" not in row for row in rows)
    observer, loaded = load_remaining_classifier(run, config, manifest, allow_smoke=True)
    assert loaded["state_sha256"] == metadata["state_sha256"]
    assert not observer.training and all(not p.requires_grad for p in observer.parameters())
    with pytest.raises(ValueError, match="smoke"):
        load_remaining_classifier(run, config, manifest)
    changed = copy.deepcopy(config)
    changed["unlearning"]["target_client_id"] = 0
    with pytest.raises(ValueError, match="fingerprint"):
        load_remaining_classifier(run, changed, manifest, allow_smoke=True)
    # Exercise the public main.py audit route with the new observer, not the old one.
    # Cheap FID test double avoids loading a large pretrained network on this CPU test.
    import src.experiments.reconstruction_channel_audit as audit_module
    from src.main import main
    class DiagnosticFeatures:
        def __init__(self, **kwargs):
            self.metadata = {"feature_dimensions": 3, "implementation": "test-double-only"}
        def __call__(self, images):
            assert not torch.is_grad_enabled()
            return images.mean((2, 3))
    monkeypatch.setattr(audit_module, "InceptionFIDFeatures", DiagnosticFeatures)
    codec_config = VariationalDeepJSCCConfig(latent_channels=1, latent_dim=64)
    config["codec"] = {"architecture": DEEPJSCC_ARCHITECTURE, **vars(codec_config)}
    config["evaluation"] = {"mia_min_examples_per_class": 2}
    for row in manifest["clients"]:
        row["test_indices"] = list(range(10 * row["client_id"], 10 * (row["client_id"] + 1)))
    (source / "config.json").write_text(json.dumps(config))
    (source / "split_manifest.json").write_text(json.dumps(manifest))
    (source / "checkpoints").mkdir()
    state = VariationalDeepJSCC(codec_config).state_dict()
    for name in ("original", "retrain_without_client_1", "unlearned"):
        torch.save(state, source / f"checkpoints/{name}.pt")
    audit_root = tmp_path / "2026-09-22-02-observer-audit-test"
    main(["--experiment", "reconstruction_channel_audit", "--source-run", str(source),
          "--observer-run", str(run), "--output-root", str(audit_root),
          "--audit-device", "cpu", "--snr_db", "20", "--batch_size", "20", "--smoke"])
    audit = next(audit_root.iterdir())
    with (audit / "metrics/retrain_output_semantic_comparison.csv").open() as handle:
        audit_rows = list(csv.DictReader(handle))
    assert len(audit_rows) == 6
    assert all(float(row["class_agreement_to_retrain"]) == 1 for row in audit_rows)
    assert (audit / "predictions/erased_train_semantic_logits.pt").stat().st_size > 0
    assert (audit / "figures/retrain_output_semantic_comparison.png").stat().st_size > 0
    protocol = json.loads((audit / "metrics/output_comparison_protocol.json").read_text())
    assert protocol["observer"]["state_sha256"] == metadata["state_sha256"]
    assert not (source / "metrics").exists()
    # Normal FSCU/V4 preparation also copies exactly this shared observer.
    import src.utils.trainer as trainer_module
    import src.utils.semantic_classifier as observer_module
    original_loader = observer_module.load_remaining_classifier
    def load_test_observer(*args, **kwargs):
        return original_loader(*args, **kwargs, allow_smoke=True)
    monkeypatch.setattr(observer_module, "load_remaining_classifier", load_test_observer)
    monkeypatch.setattr(trainer_module, "__file__", str(tmp_path / "src/utils/trainer.py"))
    class ServerComm:
        def Get_rank(self):
            return 0
        def bcast(self, value, root):
            return value
    config["semantic_classifier"] = {"source_run": "unused-legacy-classifier"}
    data_state = {"split_manifest": copy.deepcopy(manifest), "training_pool_metadata": {"selected_index_sha256": "source-pool"}}
    prepared_root = tmp_path / "prepared-codec-run"
    prepared, prepared_metadata, _ = trainer_module._prepare_frozen_semantic_classifier(
        ServerComm(), data_state, config, "cpu", 0, prepared_root,
        RunArtifacts(prepared_root), observer_run=run,
    )
    assert prepared_metadata["state_sha256"] == metadata["state_sha256"]
    assert prepared.config.base_channels == 8
    assert "remaining_observer_protocol_sha256" in data_state["split_manifest"]["semantic_classifier_training"]
    config["training"] = {"semantic_weight": 1.0}
    with pytest.raises(RuntimeError, match="gradients to codec"):
        trainer_module._prepare_frozen_semantic_classifier(
            ServerComm(), data_state, config, "cpu", 0, prepared_root,
            RunArtifacts(prepared_root), observer_run=run,
        )
    # Even a rehashed protocol may not silently reintroduce target indices.
    corrupt = copy.deepcopy(metadata)
    corrupt["protocol"]["client_splits"][0]["train_indices"][0] = 20
    corrupt["protocol_sha256"] = hashlib.sha256(json.dumps(corrupt["protocol"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (run / "observer_manifest.json").write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="indices"):
        load_remaining_classifier(run, config, manifest, allow_smoke=True)
    (run / "observer_manifest.json").write_text(json.dumps(metadata))
    with (run / "checkpoints/observer.pt").open("ab") as handle:
        handle.write(b"corruption-test")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        load_remaining_classifier(run, config, manifest, allow_smoke=True)
