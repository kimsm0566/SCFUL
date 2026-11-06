FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates git libopenmpi-dev openmpi-bin \
        python-is-python3 python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0

COPY docker/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    && python3 -m pip install --requirement /tmp/requirements.txt

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid "${HOST_GID}" scful \
    && useradd --create-home --uid "${HOST_UID}" --gid "${HOST_GID}" --shell /bin/bash scful

WORKDIR /workspace
USER scful
