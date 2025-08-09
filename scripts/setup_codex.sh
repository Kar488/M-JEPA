#!/usr/bin/env bash
set -euxo pipefail

# Codex runs Python 3.12 on CPU
python -m pip install --upgrade pip wheel setuptools

# Install Torch first (CPU wheels)
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Then install torch-scatter matching Torch (CPU)
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.1+cpu.html

# Install the rest (requirements.txt should NOT contain rdkit/rdkit-pypi)
pip install -r requirements.txt
