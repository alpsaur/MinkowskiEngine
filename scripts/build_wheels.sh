#!/usr/bin/env bash
# Build a cu128 / sm_120 wheel for MinkowskiEngine locally.
#
# Run inside an environment that already has: the CUDA 12.8 toolkit (nvcc) at
# CUDA_HOME (default /usr/local/cuda), Python 3.10-3.12 with pip, and torch from
# the cu128 index already installed (pip install torch --index-url
# https://download.pytorch.org/whl/cu128).
#
# The resulting wheel is pinned to the installed torch's CUDA major (cu128) and
# targets compute capability 12.0 (Blackwell). It is a linux_x86_64 wheel (NOT
# manylinux-certified; no auditwheel repair, since this ext dynamically links to
# torch/CUDA libs provided by the user environment). Install + import-test to validate.
set -euo pipefail

export FORCE_CUDA=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export MAX_COMPILATION_THREADS="${MAX_COMPILATION_THREADS:-8}"

# --no-build-isolation: build in the current env so setup.py can import torch.
pip wheel . --no-build-isolation -w wheelhouse/

echo "Built wheels:"; ls -la wheelhouse/
echo "NOTE: linux_x86_64 wheel, cu128, sm_120. Not manylinux-certified; no auditwheel repair."
