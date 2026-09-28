# GPU Development Environment

This Docker environment runs the bundled FedOSD and SCU code with Python 3.10,
PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1.

The host must have a working NVIDIA driver, Docker Engine, and NVIDIA Container
Toolkit. The container receives every host GPU. Run `nvidia-smi` on the host
before starting if GPU access is in doubt.

## Execution Policy

All project code execution must occur inside `scful-dev`: Python scripts,
tests, compilation/import checks, MPI jobs, and experiments. The host runs
only Docker lifecycle commands. Use `docker compose exec -T scful-dev ...`
for non-interactive commands; `/workspace` is the bind-mounted repository root.

## Lifecycle

Build the image with the current host user ID and group ID so files written from
the container remain editable by the host user:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build scful-dev
```

Start the persistent development container:

```bash
docker compose up -d scful-dev
```

Open a shell:

```bash
docker compose exec scful-dev bash
```

Run the environment checks:

```bash
docker compose exec -T scful-dev python docker/verify_environment.py
```

Run tests or an MPI experiment through the container:

```bash
docker compose exec -T scful-dev pytest -q src/tests
docker compose exec -T scful-dev mpirun --oversubscribe -np 11 \
  python src/main.py --experiment federated_deletion \
  --config src/configs/federated_cifar10_dirichlet_01_scu_public_compensation_smoke.json \
  --output-root outputs/2026-09-03-01-fedscu-pubcc-smoke
```

Every new `outputs/` root must follow `YYYY-MM-DD-Index-brief-title`, using
Asia/Seoul date and a two-digit daily index. Use a new index for repeated runs.

Inspect the service logs:

```bash
docker compose logs --tail=100 scful-dev
```

Stop the service without removing the image or project data:

```bash
docker compose down
```

## Persistence

The repository is bind-mounted at `/workspace`; the image contains no research
source or dataset files. Dataset downloads in `data/`, experiment outputs in
`outputs/`, and PyTorch downloads in `.cache/torch/` remain on the host when the
container is stopped or recreated.
