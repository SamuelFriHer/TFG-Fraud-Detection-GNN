FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# System-level build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12 support
RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch_geometric

# Pre-install project dependencies
COPY pyproject.toml /tmp/pip-tmp/
COPY src/__init__.py /tmp/pip-tmp/src/
RUN pip install --disable-pip-version-check --no-cache-dir "/tmp/pip-tmp[dev]" \
    && rm -rf /tmp/pip-tmp
