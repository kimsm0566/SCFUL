import numpy as np
import pytest

from src.evaluation.support_response import (
    SupportGroupingConfig,
    build_class_support_records,
    merge_support_with_class_metrics,
    summarize_support_groups,
)
from src.experiments.support_response import (
    aggregate_support_response,
    collect_support_response_runs,
)


def test_build_class_support_records_assigns_groups_from_deletion_fraction():
    records = build_class_support_records(
        class_labels=np.array([0, 1, 2, 3]),
        train_allocation=np.array(
            [[900, 3500, 1500, 100], [4100, 1500, 3500, 4900]]
        ),
        target_client_id=0,
        config=SupportGroupingConfig(min_target_samples=500),
    )

    assert records[0]["support_group"] == "shared_high"
    assert records[1]["support_group"] == "low_support"
    assert records[2]["support_group"] == "medium_support"
    assert records[3]["support_group"] == "not_targeted"


def test_merge_support_with_class_metrics_adds_retrain_mse_delta_and_summary():
    support_records = build_class_support_records(
        class_labels=np.array([0, 1]),
        train_allocation=np.array([[900, 3500], [4100, 1500]]),
        target_client_id=0,
        config=SupportGroupingConfig(min_target_samples=500),
    )
    class_metrics = [
        {
            "model": "original",
            "group": "class",
            "group_id": 0,
            "samples": 10,
            "mse_input": 0.20,
            "psnr": 7.0,
            "fg_output": 0.02,
            "fg_mu": 0.03,
            "fg_latent": 0.04,
        },
        {
            "model": "retrain_without_client_0",
            "group": "class",
            "group_id": 0,
            "samples": 10,
            "mse_input": 0.18,
            "psnr": 7.4,
            "fg_output": 0.0,
            "fg_mu": 0.0,
            "fg_latent": 0.0,
        },
        {
            "model": "original",
            "group": "class",
            "group_id": 1,
            "samples": 10,
            "mse_input": 0.24,
            "psnr": 6.2,
            "fg_output": 0.06,
            "fg_mu": 0.07,
            "fg_latent": 0.08,
        },
        {
            "model": "retrain_without_client_0",
            "group": "class",
            "group_id": 1,
            "samples": 10,
            "mse_input": 0.23,
            "psnr": 6.4,
            "fg_output": 0.0,
            "fg_mu": 0.0,
            "fg_latent": 0.0,
        },
    ]

    records = merge_support_with_class_metrics(support_records, class_metrics)

    original_label_zero = next(
        record
        for record in records
        if record["model"] == "original" and record["label"] == 0
    )
    assert original_label_zero["support_group"] == "shared_high"
    assert original_label_zero["mse_delta_to_retrain"] == pytest.approx(0.02)

    summary = summarize_support_groups(records, candidate_models=["original"])
    shared_summary = next(
        record for record in summary if record["support_group"] == "shared_high"
    )
    assert shared_summary["model"] == "original"
    assert shared_summary["label_count"] == 1
    assert shared_summary["mse_input_mean"] == pytest.approx(0.20)


def test_collect_support_response_runs_uses_only_completed_support_artifacts(tmp_path):
    complete = tmp_path / "run_complete"
    incomplete = tmp_path / "run_incomplete"
    (complete / "metrics").mkdir(parents=True)
    (complete / "metrics" / "support_by_class.csv").write_text("model,label\n")
    incomplete.mkdir()

    assert collect_support_response_runs(tmp_path) == [complete]


def test_aggregate_support_response_writes_csv_and_figures(tmp_path):
    run = tmp_path / "runs" / "run_complete"
    (run / "metrics").mkdir(parents=True)
    (run / "run_metadata.json").write_text(
        '{"global_seed": 3, "split_seed": 17}'
    )
    (run / "config.json").write_text('{"experiment_name": "support_test"}')
    (run / "metrics" / "support_by_class.csv").write_text(
        "label,target_train_samples,retained_train_samples,total_train_samples,"
        "deletion_fraction,remaining_fraction,support_group,model,group,group_id,"
        "samples,mse_input,psnr,fg_output,fg_mu,fg_latent,test_samples,"
        "mse_delta_to_retrain\n"
        "1,3500,1500,5000,0.7,0.3,low_support,original,class,1,10,0.2,7.0,"
        "0.02,0.03,0.04,10,0.01\n"
        "1,3500,1500,5000,0.7,0.3,low_support,retrain_without_client_0,class,1,"
        "10,0.19,7.2,0.0,0.0,0.0,10,0.0\n"
    )

    output = aggregate_support_response([run], tmp_path / "analysis", seed=0)

    assert (output / "metrics" / "support_response.csv").exists()
    assert (output / "metrics" / "support_group_summary.csv").exists()
    assert (output / "figures" / "original_support_response.png").exists()
    assert (output / "figures" / "model_support_response.png").exists()
