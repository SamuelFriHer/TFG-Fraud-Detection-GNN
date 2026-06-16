FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# System-level build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to look for PyTorch packages with CUDA 12 support
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Install PyTorch with CUDA 12 support
RUN pip install --no-cache-dir torch==2.4.1

# Install PyTorch Geometric and optional dependencies
RUN pip install --no-cache-dir torch_geometric \
    && pip install --no-cache-dir pyg_lib torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# Pre-install project dependencies
COPY pyproject.toml /tmp/pip-tmp/
COPY src/__init__.py /tmp/pip-tmp/src/
RUN pip install --disable-pip-version-check --no-cache-dir "/tmp/pip-tmp[dev]" \
    && rm -rf /tmp/pip-tmp
