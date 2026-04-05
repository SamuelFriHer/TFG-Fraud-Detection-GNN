FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install PyTorch Geometric
RUN pip install --no-cache-dir torch_geometric

# Copy and install Python dependencies
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --disable-pip-version-check --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
