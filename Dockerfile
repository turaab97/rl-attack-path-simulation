# ============================================================================
# Dockerfile — RL Attack-Path Simulation
# Author: Syed Ali Turab
# Course: MMAI 845 – Reinforcement Learning
#
# Provides a reproducible, self-contained environment for training and
# evaluating PPO / DQN attack-path agents on the NASim AI-infrastructure
# network topology.
#
# Usage:
#   docker compose run train
#   docker compose run evaluate
#   docker compose run visualize
# ============================================================================

FROM python:3.12-slim AS base

LABEL maintainer="Syed Ali Turab <info@turab.sh>"
LABEL description="RL Attack-Path Simulation — MMAI 845 Final Project"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level dependencies needed by NASim and matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-tk \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy the full project
COPY . .

# Re-install in editable mode now that all source is present
RUN pip install --no-cache-dir -e .

# Default entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["train", "--compare", "--timesteps", "300000"]
