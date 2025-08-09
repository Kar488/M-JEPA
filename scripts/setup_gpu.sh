#!/usr/bin/env bash
set -euxo pipefail

python -m pip install --upgrade pip wheel setuptools
# CUDA 12.1 wheels (RTX A5000)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install -r requirements.txt
