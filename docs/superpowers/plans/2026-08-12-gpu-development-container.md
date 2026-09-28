# GPU Development Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build and run one persistent GPU Docker development container for FedOSD and SCU under Python 3.10, PyTorch 2.4.1, and CUDA 12.1.

**Architecture:** A CUDA 12.1 Ubuntu base image contains pinned PyTorch CUDA wheels and scientific Python dependencies. Docker Compose runs the image as a GPU-enabled keepalive service and bind-mounts the repository. A standalone verifier checks CUDA, FedOSD import, and SCU compilation without beginning a training run.

**Tech Stack:** Docker Compose, NVIDIA Container Toolkit, CUDA 12.1.1, Ubuntu 22.04, Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, NumPy, SciPy, Matplotlib, Pillow.

## Global Constraints

- Base image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`.
- Pin `torch==2.4.1+cu121` and `torchvision==0.19.1+cu121` from the official cu121 wheel index.
- Do not copy source code or datasets into image layers; bind-mount the repository at `/workspace`.
- Run as UID/GID 1000 by default, matching the current host account.
- Use all GPUs, 8 GB shared memory, `sleep infinity`, and `restart: unless-stopped`.
- Do not modify FedOSD or SCU algorithmic behavior.
- Do not claim success until CUDA, FedOSD import, and SCU source compilation pass.
- The workspace has no top-level Git repository, so plan-task commits are unavailable.

---

### Task 1: Create and build the reproducible CUDA image

**Files:**
- Create: `docker/requirements.txt`
- Create: `docker/Dockerfile`

**Interfaces:**
- Consumes: `HOST_UID` and `HOST_GID` build arguments, both defaulted to `1000`.
- Produces: `scful-dev:torch2.4.1-cu121`, with user `scful` and working directory `/workspace`.

- [ ] **Step 1: Declare non-PyTorch packages**

Create `docker/requirements.txt`:

```text
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.2
pillow==10.4.0
tqdm==4.66.5
```

- [ ] **Step 2: Define the image**

Create `docker/Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates git python-is-python3 python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    && python3 -m pip install --requirement /tmp/requirements.txt

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid "${HOST_GID}" scful \
    && useradd --create-home --uid "${HOST_UID}" --gid "${HOST_GID}" --shell /bin/bash scful

WORKDIR /workspace
USER scful
```

- [ ] **Step 3: Build and inspect**

Run:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build scful-dev
docker image inspect scful-dev:torch2.4.1-cu121 --format '{{.Config.WorkingDir}} {{.Config.User}}'
```

Expected: the build succeeds and inspection prints `/workspace scful`.

### Task 2: Create the persistent Compose service and documentation

**Files:**
- Create: `compose.yaml`
- Create: `docker/README.md`

**Interfaces:**
- Consumes: image from Task 1.
- Produces: a `scful-dev` service with GPU access and an editable `/workspace` mount.

- [ ] **Step 1: Define the service**

Create `compose.yaml`:

```yaml
services:
  scful-dev:
    build:
      context: .
      dockerfile: docker/Dockerfile
      args:
        HOST_UID: "${HOST_UID:-1000}"
        HOST_GID: "${HOST_GID:-1000}"
    image: scful-dev:torch2.4.1-cu121
    container_name: scful-dev
    working_dir: /workspace
    command: ["sleep", "infinity"]
    init: true
    stdin_open: true
    tty: true
    restart: unless-stopped
    shm_size: 8gb
    environment:
      MPLBACKEND: Agg
      PYTHONPATH: /workspace/src/FedOSD
      TORCH_HOME: /workspace/.cache/torch
    volumes:
      - .:/workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

- [ ] **Step 2: Check the Compose file**

Run:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose config --quiet
```

Expected: exit status 0.

- [ ] **Step 3: Document operation**

Create `docker/README.md`. State that NVIDIA Container Toolkit is required on the host and include:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build scful-dev
docker compose up -d scful-dev
docker compose exec scful-dev bash
docker compose exec scful-dev python docker/verify_environment.py
docker compose logs --tail=100 scful-dev
docker compose down
```

Explain that `data/`, `outputs/`, and `.cache/torch/` survive container recreation through the project bind mount.

- [ ] **Step 4: Start and inspect the keepalive service**

Run:

```bash
docker compose up -d scful-dev
docker compose ps scful-dev
```

Expected: `scful-dev` is running and does not restart.

### Task 3: Test-drive and implement environment verification

**Files:**
- Create: `docker/tests/test_verify_environment.py`
- Create: `docker/verify_environment.py`

**Interfaces:**
- `compile_scu_tree(scu_root: Path) -> bool`: compile every SCU Python file in memory without creating bytecode artefacts in the cloned source tree.
- `import_fedosd(fedosd_root: Path) -> None`: import the FedOSD `tfedplat` package.
- `check_cuda() -> None`: require CUDA availability and compute `[1, 2]^2.sum() == 5` on GPU.
- `main() -> None`: print versions then perform all checks.

- [ ] **Step 1: Write a test that initially fails**

Create `docker/tests/test_verify_environment.py`:

```python
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "docker"))

import verify_environment


class VerifyEnvironmentTests(unittest.TestCase):
    def test_scu_tree_compiles(self):
        self.assertTrue(
            verify_environment.compile_scu_tree(REPOSITORY_ROOT / "src" / "SCU")
        )

    def test_fedosd_package_imports(self):
        verify_environment.import_fedosd(REPOSITORY_ROOT / "src" / "FedOSD")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the test is red**

Run:

```bash
docker compose exec scful-dev python -m unittest docker.tests.test_verify_environment -v
```

Expected: failure because `verify_environment` is absent.

- [ ] **Step 3: Implement the verifier**

Create `docker/verify_environment.py`:

```python
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
    sys.path.insert(0, str(fedosd_root))
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
    import_fedosd(repository_root / "src" / "FedOSD")
    if not compile_scu_tree(repository_root / "src" / "SCU"):
        raise RuntimeError("SCU source compilation failed")
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run green and GPU verification**

Run:

```bash
docker compose exec scful-dev python -m unittest docker.tests.test_verify_environment -v
docker compose exec scful-dev nvidia-smi
docker compose exec scful-dev python docker/verify_environment.py
```

Expected: two passing unit tests, RTX 4070 in `nvidia-smi`, and a verifier output that ends in `Environment verification passed.`

### Task 4: Hand off the persistent environment

**Files:**
- Modify: `docker/README.md` only if a verified command differed from the planned command.

- [ ] **Step 1: Inspect final runtime**

Run:

```bash
docker compose ps scful-dev
docker compose exec scful-dev python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))'
```

Expected: running service, versions `2.4.1+cu121` and `0.19.1+cu121`, and the RTX 4070 name.

- [ ] **Step 2: Provide evidence-backed handoff**

Report image `scful-dev:torch2.4.1-cu121`, service `scful-dev`, shell entry `docker compose exec scful-dev bash`, and command output for all checks. Note that full FedOSD and SCU training is intentionally deferred; any runtime incompatibility will require a minimal behavior-preserving patch and reproduction command.
