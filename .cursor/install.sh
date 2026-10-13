#!/usr/bin/env bash
# Idempotent dependency setup for the Generative Trajectory Planning MVP.
# Installs into the per-user site (~/.local) so `python3 -m gtp.*` works in any
# shell without activating a virtualenv. The base image's system Python is
# PEP 668 "externally managed", so --break-system-packages is required.
set -euo pipefail

# Cloud Agent VMs have no GPU; pin the CPU PyTorch wheel to avoid pulling the
# large CUDA build and to keep torch.cuda.is_available() == False.
python3 -m pip install --user --break-system-packages --upgrade pip
python3 -m pip install --user --break-system-packages \
  --index-url https://download.pytorch.org/whl/cpu "torch>=2.3"

# Project (with dev extras: pytest + ruff), resolved from PyPI. torch is
# already satisfied by the CPU wheel installed above.
python3 -m pip install --user --break-system-packages -e ".[dev]"
