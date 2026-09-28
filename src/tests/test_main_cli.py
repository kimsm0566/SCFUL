from pathlib import Path

from src.utils.option import args_parser


def test_option_parser_reads_current_scu_and_experiment_arguments():
    args = args_parser(
        [
            "--experiment",
            "controlled_behavior",
            "--algorithm",
            "SCU",
            "--seed",
            "7",
            "--config",
            "configs/example.json",
            "--output-root",
            "tmp/run",
            "--smoke",
        ]
    )

    assert args.experiment == "controlled_behavior"
    assert args.algorithm == "SCU"
    assert args.seed == 7
    assert args.config == Path("configs/example.json")
    assert args.output_root == Path("tmp/run")
    assert args.smoke is True
    assert args.scu_forget_ratio == 0.06


def test_option_parser_accepts_federated_deletion_experiment():
    args = args_parser(["--experiment", "federated_deletion"])

    assert args.experiment == "federated_deletion"


def test_option_parser_accepts_fscu_as_the_public_data_algorithm():
    args = args_parser(
        [
            "--algorithm",
            "FSCU",
            "--config",
            "configs/fscu.json",
            "--output-root",
            "outputs/2026-09-07-01-fscu",
            "--reference-run",
            "outputs/reference/20260907T000000Z_seed-0",
        ]
    )

    assert args.algorithm == "FSCU"
    assert args.config == Path("configs/fscu.json")
    assert args.output_root == Path("outputs/2026-09-07-01-fscu")
    assert args.reference_run == Path(
        "outputs/reference/20260907T000000Z_seed-0"
    )


def test_option_parser_accepts_fscuv4_retained_feature_kd():
    args = args_parser(
        [
            "--algorithm",
            "FSCUv4",
            "--config",
            "configs/fscuv4.json",
            "--output-root",
            "outputs/2026-09-08-19-fscuv4-smoke",
        ]
    )

    assert args.algorithm == "FSCUv4"


def test_option_parser_accepts_membership_inference_audit_sources():
    args = args_parser(
        [
            "--experiment",
            "membership_inference_audit",
            "--source-run",
            "outputs/run-a",
            "--source-run",
            "outputs/run-b",
            "--audit-device",
            "cpu",
        ]
    )

    assert args.experiment == "membership_inference_audit"
    assert args.source_run == [Path("outputs/run-a"), Path("outputs/run-b")]
    assert args.audit_device == "cpu"


def test_option_parser_accepts_reconstruction_channel_audit():
    args = args_parser(
        [
            "--experiment",
            "reconstruction_channel_audit",
            "--source-run",
            "outputs/run-a",
            "--audit-device",
            "cuda:0",
            "--in-place",
        ]
    )

    assert args.experiment == "reconstruction_channel_audit"
    assert args.source_run == [Path("outputs/run-a")]
    assert args.audit_device == "cuda:0"
    assert args.in_place is True


def test_option_parser_accepts_support_response_experiment():
    args = args_parser(
        ["--experiment", "support_response", "--runs-root", "outputs/support_runs"]
    )

    assert args.experiment == "support_response"
    assert args.runs_root == Path("outputs/support_runs")
