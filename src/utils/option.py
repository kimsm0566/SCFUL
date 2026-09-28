import argparse
from pathlib import Path

from src.utils.config import DEFAULT_ORACLE_CONFIG_PATH


def args_parser(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=[
            "scu",
            "controlled_behavior",
            "federated_deletion",
            "reconstruction_channel_audit",
            "membership_inference_audit",
            "support_response",
            "training_curves",
            "remaining_classifier",
        ],
        default="scu",
        help=(
            "Optional legacy/baseline entry. The main FSCU workflow uses "
            "--algorithm FSCU and evaluates every registered metric automatically."
        ),
    )
    
    parser.add_argument("--algorithm", help="Training/unlearning algorithm.",
                        type=str, choices=['SCU', 'FSCU', 'FSCUv4'], default="SCU")

    #Data
    parser.add_argument(
        "--dataset",
        help="Dataset name (CIFAR-10 is the only supported dataset).",
        type=str,
        choices=["cifar10"],
        default="cifar10",
    )
    parser.add_argument("--partition_type", help="Name of the dataset: [iid, class, class-rep].",
                        type=str, choices=['iid', 'class', 'class-rep'], default="iid")    
    #path
    parser.add_argument("--result_path", help="Where to save results.", type=str, default="./results")
    parser.add_argument("--data_path", help="Where to find the data.", type=str, default="/home/imes-server4/data")
    
    #Hyper parameter
    parser.add_argument('--snr_db', type=int, default=10, help="Channel SNR in dB")
    parser.add_argument("--n_clients", help="Number of clients.", type=int, default=2)
    parser.add_argument("--n_client_data", help="Number of data points for each client.", type=int, default=1000)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.001)
    parser.add_argument("--major_percent", help="Percentage of majority class for client data partition.", type=float, default=0.8)
    parser.add_argument("--n_epochs", help="Number of DML epochs.", type=int, default=3)
    parser.add_argument("--batch_size", help="Batch size during training.", type=int, default=250)
    parser.add_argument(
        "--scu_forget_ratio",
        help="Fraction of one client's data selected for SCU deletion.",
        type=float,
        default=0.06,
    )
    # parser.add_argument("--beta", help="Batch size during training.", type=float, default=0.5)
    #utils
    parser.add_argument("--device", help="Which cuda device to use.", type=int, default=0)
    parser.add_argument("--seed", help="Random seed.", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_ORACLE_CONFIG_PATH)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/oracle_behavior")
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=None,
        help=(
            "Completed FSCU-versioned run whose compatible initial, original, and "
            "retrain checkpoints should be reused."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Support-response analysis에 합칠 completed run directory의 상위 경로.",
    )
    parser.add_argument(
        "--source-run",
        action="append",
        type=Path,
        default=[],
        help="Membership-inference audit로 평가할 completed federated run directory. 여러 번 지정 가능.",
    )
    parser.add_argument(
        "--observer-run",
        type=Path,
        default=None,
        help="Completed remaining-client classifier run used as one frozen evaluator for all codecs.",
    )
    parser.add_argument(
        "--audit-device",
        type=str,
        default="auto",
        help="Membership-inference audit device (auto, cpu, cuda:0).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write reconstruction channel audit files into the source run.",
    )
    parser.add_argument("--smoke", "--smoke-test", action="store_true")

    return parser.parse_args(argv)
