"""과거 baseline 재현과 저장 checkpoint 재평가를 위한 보조 실행 함수."""


def _run_controlled_behavior(args):
    from src.evaluation.oracle_behavior import run_oracle_behavior
    from src.utils.config import load_json_config

    config = load_json_config(args.config)
    run_dir = run_oracle_behavior(
        config,
        seed=args.seed,
        output_root=args.output_root,
        smoke_test=args.smoke,
    )
    print(f"Controlled behavior experiment finished: {run_dir}")


def _run_membership_inference_audit(args):
    from src.experiments.membership_inference_audit import (
        run_membership_inference_audit,
    )

    run_dir = run_membership_inference_audit(
        args.source_run,
        args.output_root,
        args.seed,
        device=args.audit_device,
    )
    print(f"Membership inference audit finished: {run_dir}")


def _run_reconstruction_channel_audit(args):
    from src.experiments.reconstruction_channel_audit import (
        run_reconstruction_channel_audit,
    )

    if len(args.source_run) != 1:
        raise ValueError(
            "reconstruction_channel_audit needs exactly one --source-run"
        )
    run_dir = run_reconstruction_channel_audit(
        source_run=args.source_run[0],
        output_root=args.output_root,
        seed=args.seed,
        device=args.audit_device,
        snr_db=args.snr_db,
        batch_size=args.batch_size,
        in_place=args.in_place,
        smoke_test=args.smoke,
        observer_run=getattr(args, "observer_run", None),
    )
    print(f"Reconstruction channel audit finished: {run_dir}")


def _run_support_response(args):
    from src.experiments.support_response import (
        aggregate_support_response,
        collect_support_response_runs,
    )

    if args.runs_root is None:
        raise ValueError("support_response needs --runs-root")
    run_directories = collect_support_response_runs(args.runs_root)
    run_dir = aggregate_support_response(
        run_directories,
        args.output_root,
        args.seed,
    )
    print(f"Support response analysis finished: {run_dir}")


def run_auxiliary_experiment(args):
    """명시적으로 요청된 과거 보조 작업을 실행했는지 반환한다."""
    runners = {
        "controlled_behavior": _run_controlled_behavior,
        "reconstruction_channel_audit": _run_reconstruction_channel_audit,
        "membership_inference_audit": _run_membership_inference_audit,
        "support_response": _run_support_response,
    }
    runner = runners.get(args.experiment)
    if runner is None:
        return False
    runner(args)
    return True
