"""재현 가능한 실험 seed와 실행 환경 기록 도구."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision


def derive_seed(global_seed: int, namespace: str) -> int:
    """전역 seed와 이름에서 안정적인 31-bit 하위 seed를 만든다."""
    payload = f"{global_seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def set_global_seed(seed: int, deterministic: bool) -> None:
    """Python, NumPy, PyTorch의 난수를 같은 seed로 초기화한다."""
    if deterministic:
        # CUDA context/linear algebra를 사용하기 전에 설정해야 CuBLAS도 deterministic
        # algorithm policy를 따를 수 있다.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def _git_revision(path: Path) -> dict[str, str | None]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"repository": str(path), "revision": revision}
    except (OSError, subprocess.CalledProcessError):
        return {"repository": str(path), "revision": None}


def collect_environment(device: torch.device, workspace_root: Path) -> dict[str, Any]:
    """실행 재현에 필요한 주요 runtime과 revision 정보를 수집한다."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "selected_device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "top_level_git": _git_revision(workspace_root),
        "nested_source_revisions": [
            _git_revision(workspace_root / "src" / "SCU"),
            _git_revision(workspace_root / "src" / "FedOSD"),
        ],
    }
