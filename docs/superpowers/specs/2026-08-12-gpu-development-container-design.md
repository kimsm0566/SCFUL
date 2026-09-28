# GPU Development Container Design

## Goal

Provide one persistent NVIDIA GPU Docker development environment that can run and modernize the bundled FedOSD and SCU research code without copying source code or datasets into the image.

## Constraints

- Host GPU: NVIDIA GeForce RTX 4070 (12 GB).
- Host driver: 535.183.01, reporting CUDA 12.2 support.
- Runtime target: Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1 wheels.
- The server must not receive raw project data inside the built image. Source, datasets, and outputs are host bind mounts.
- The container must remain alive after startup and be restartable without rebuilding.
- GPU access must be explicit and verified after the container starts.
- SCU must be updated only as needed to run under the selected modern environment; algorithmic behavior is out of scope.

## Architecture

The Docker image is a reusable CUDA 12.1 Ubuntu 22.04 development base. It installs Python 3.10, the exact PyTorch/torchvision CUDA wheel pair, and the shared scientific Python dependencies used by FedOSD and SCU.

Docker Compose builds this image as `scful-dev:torch2.4.1-cu121` and runs one `scful-dev` service. The service receives all available NVIDIA GPUs, uses an 8 GB shared-memory segment, bind-mounts the repository at `/workspace`, and keeps its process alive with `sleep infinity`. Source changes on the host are immediately visible inside the container.

## Files and Responsibilities

- `docker/Dockerfile`: reproducible CUDA-enabled development image and non-root developer account.
- `docker/requirements.txt`: pinned Python dependencies beyond PyTorch and torchvision.
- `compose.yaml`: image build, GPU runtime, bind mounts, persistent process, shared-memory size, and restart policy.
- `docker/verify_environment.py`: non-training smoke test for the environment, FedOSD import, SCU source compilation, and CUDA tensor execution.
- `docker/README.md`: exact build, start, shell, stop, verification, and log commands; also records the host NVIDIA Container Toolkit prerequisite.

## Data Flow and Persistence

The repository root is mounted read-write at `/workspace`. Dataset downloads are written to `/workspace/data`; experiment artefacts are written to `/workspace/outputs`. Both paths therefore survive container recreation. The Docker image contains only its system and Python environment.

## Compatibility Approach

FedOSD is imported from `/workspace/src/FedOSD` to validate its `tfedplat` package. SCU is not imported as a package because its experiment files have training side effects at module import time; instead, every SCU Python source file is compiled in memory. This avoids training side effects and does not write bytecode artefacts into the cloned source tree.

The first environment pass must not silently change research algorithms. If the smoke test finds an incompatible runtime API, the follow-up fix will be limited to an equivalent compatibility adaptation and will be verified by the relevant minimal script or import check.

## Validation

After a successful build and service start, run `docker/verify_environment.py` inside the container. It must:

1. Print Python, PyTorch, torchvision, and CUDA versions.
2. Require `torch.cuda.is_available()` and successfully perform a CUDA tensor operation.
3. Import FedOSD's `tfedplat` package from its clone.
4. Compile all SCU Python files without syntax errors.
5. Exit nonzero if any check fails.

The project will also run `docker compose config` before startup to validate Compose interpolation and service syntax.

## Operational Commands

The documented normal lifecycle is:

```bash
docker compose build
docker compose up -d
docker compose exec scful-dev bash
docker compose exec scful-dev python docker/verify_environment.py
docker compose down
```

`docker compose down` stops the service without deleting mounted project data or Docker image layers. A separate `docker compose up -d` starts the keepalive service again.

## Non-Goals

- Reproducing SCU exactly on its original PyTorch 1.4 stack.
- Downloading datasets while building the image.
- Running full FedOSD or SCU training in the initial environment setup.
- Changing FedOSD or SCU unlearning algorithms.
- Creating a second CPU-only or legacy-PyTorch container.
