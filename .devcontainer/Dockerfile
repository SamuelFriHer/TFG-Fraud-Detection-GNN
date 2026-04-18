FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# System-level build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch_geometric

# Pre-install project dependencies
COPY pyproject.toml /tmp/pip-tmp/
COPY src/__init__.py /tmp/pip-tmp/src/
RUN pip install --disable-pip-version-check --no-cache-dir "/tmp/pip-tmp[dev]" \
    && rm -rf /tmp/pip-tmp
