from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch
import torchvision


def compile_scu_tree(scu_root: Path) -> bool:
    for source_path in scu_root.rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        compile(source, str(source_path), "exec")
    return True


def import_fedosd(fedosd_root: Path) -> None:
    fedosd_path = str(fedosd_root)
    if fedosd_path not in sys.path:
        sys.path.insert(0, fedosd_path)
    importlib.import_module("tfedplat")


def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container")

    value = torch.tensor([1.0, 2.0], device="cuda").square().sum().item()
    if value != 5.0:
        raise RuntimeError(f"unexpected CUDA tensor result: {value}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    check_cuda()
    import_fedosd(repository_root / "third_party" / "FedOSD")
    if not compile_scu_tree(repository_root / "third_party" / "SCU"):
        raise RuntimeError("SCU source compilation failed")
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
