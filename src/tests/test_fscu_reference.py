import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.utils.config import load_fscu_config
from src.utils.fscu_runtime import (
    build_fscu_reference_manifest,
    load_fscu_reference_models,
    make_fscu_codec,
    make_initial_fscu_state,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = (
    WORKSPACE_ROOT
    / "src/configs/federated_cifar10_vdeepjscc_scu_public_compensation_smoke.json"
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_reference_run(tmp_path, config, split_manifest, seed=0):
    run_dir = tmp_path / "20260907T000000Z_seed-0"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "run_metadata.json", {"global_seed": seed})
    _write_json(run_dir / "split_manifest.json", split_manifest)
    _write_json(
        run_dir / "metrics/efficiency.json",
        {
            "phase_wall_seconds": {
                "original_training": 12.5,
                "retrain_without_target": 10.0,
            }
        },
    )
    state = make_initial_fscu_state(config, seed)
    target = int(config["unlearning"]["target_client_id"])
    for name in ("initial", "original", f"retrain_without_client_{target}"):
        torch.save(state, checkpoint_dir / f"{name}.pt")
    _write_json(
        run_dir / "reference_manifest.json",
        build_fscu_reference_manifest(
            run_dir,
            config,
            split_manifest,
            state,
            seed,
            {
                "original_training": 12.5,
                "retrain_without_target": 10.0,
            },
        ),
    )
    return run_dir


def test_fscu_reference_loads_compatible_states_and_training_times(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    split_manifest = {"partition": "test", "clients": [0, 1]}
    run_dir = _make_reference_run(tmp_path, config, split_manifest)

    loaded = load_fscu_reference_models(
        tmp_path,
        config,
        split_manifest,
        make_initial_fscu_state(config, 0),
        seed=0,
    )

    assert loaded["reference_run"] == str(run_dir.resolve())
    assert set(loaded["original_state"]) == set(make_fscu_codec(config).state_dict())
    assert loaded["training_wall_seconds"]["original_training"] == 12.5
    assert loaded["training_wall_seconds"]["retrain_without_target"] == 10.0
    assert loaded["reference_manifest_verified"] is True


def test_fscu_reference_rejects_different_training_scenario(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    split_manifest = {"partition": "test", "clients": [0, 1]}
    run_dir = _make_reference_run(tmp_path, config, split_manifest)
    changed_config = copy.deepcopy(config)
    changed_config["training"]["snr_db"] = 7.0

    with pytest.raises(ValueError, match="training_scenario.training.snr_db"):
        load_fscu_reference_models(
            run_dir,
            changed_config,
            split_manifest,
            make_initial_fscu_state(changed_config, 0),
            seed=0,
        )


def test_fscu_reference_rejects_different_client_split(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    split_manifest = {"partition": "test", "clients": [0, 1]}
    run_dir = _make_reference_run(tmp_path, config, split_manifest)

    with pytest.raises(ValueError, match="split_manifest.clients"):
        load_fscu_reference_models(
            run_dir,
            config,
            {"partition": "test", "clients": [1, 0]},
            make_initial_fscu_state(config, 0),
            seed=0,
        )


def test_fscu_reference_accepts_corrected_remaining_observer_pool_hash(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    config["training"]["semantic_weight"] = 0.0
    identity = {
        "state_sha256": "state-hash",
        "source_checkpoint_sha256": "checkpoint-hash",
        "remaining_observer_protocol_sha256": "protocol-hash",
    }
    legacy_observer = {
        **identity,
        "source_run": "/workspace/outputs/legacy-observer",
        "source_run_workspace_relative": "outputs/legacy-observer",
        "source_checkpoint": "/workspace/outputs/legacy-observer/checkpoints/observer.pt",
        "source_split_training_pool_index_sha256": "remaining-only-hash",
        "current_split_training_pool_index_sha256": "full-pool-hash",
        "source_pool_matches_current_training_pool": False,
    }
    corrected_observer = {
        **legacy_observer,
        "source_split_training_pool_index_sha256": "full-pool-hash",
        "source_pool_matches_current_training_pool": True,
    }
    stored_split = {
        "partition": "test",
        "clients": [0, 1],
        "semantic_classifier_training": legacy_observer,
    }
    current_split = copy.deepcopy(stored_split)
    current_split["semantic_classifier_training"] = corrected_observer
    run_dir = _make_reference_run(tmp_path, config, stored_split)

    loaded = load_fscu_reference_models(
        run_dir,
        config,
        current_split,
        make_initial_fscu_state(config, 0),
        seed=0,
    )

    assert loaded["reference_manifest_verified"] is True


def test_fscu_reference_accepts_added_null_input_mask_metadata(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    stored_split = {
        "partition": "test",
        "clients": [0, 1],
        "target_client_reconstruction_marker": {
            "protocol": "unconditional_output_watermark_v1"
        },
    }
    current_split = copy.deepcopy(stored_split)
    current_split["target_client_reconstruction_marker"]["input_mask"] = None
    run_dir = _make_reference_run(tmp_path, config, stored_split)

    loaded = load_fscu_reference_models(
        run_dir,
        config,
        current_split,
        make_initial_fscu_state(config, 0),
        seed=0,
    )

    assert loaded["reference_manifest_verified"] is True


def test_fscu_reference_rejects_joint_head_schema_one_reference(tmp_path):
    config = load_fscu_config(SMOKE_CONFIG)
    split_manifest = {"partition": "test", "clients": [0, 1]}
    run_dir = _make_reference_run(tmp_path, config, split_manifest)
    manifest_path = run_dir / "reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = manifest["scenario"]
    scenario["schema_version"] = 1
    for name in (
        "global_seed",
        "initial_state_seed",
        "initial_state_sha256",
        "optimizer",
        "aggregation",
    ):
        scenario.pop(name)
    payload = json.dumps(
        scenario, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    manifest["schema_version"] = 1
    manifest["scenario_fingerprint_sha256"] = hashlib.sha256(payload).hexdigest()
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="joint-head references are incompatible"):
        load_fscu_reference_models(
            run_dir,
            config,
            split_manifest,
            make_initial_fscu_state(config, 0),
            seed=0,
        )
