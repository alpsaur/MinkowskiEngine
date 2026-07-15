# MinkowskiEngine — CUDA 12.8 / Blackwell (RTX 50-series, sm_120) build image.
# This fork's alpsaur/MinkowskiEngine replaces the archived NVIDIA/MinkowskiEngine,
# which does not compile under CUDA 12.8 / NumPy 2.0 / Blackwell.

###############################################################################
# Base image: the -devel flavour ships nvcc + the CUDA toolkit, which the
# MinkowskiEngine CUDAExtension build requires.
###############################################################################
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

###############################################################################
# System build dependencies + Python 3.12 + OpenBLAS + Ninja.
###############################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3.12-distutils \
        python3-pip \
        build-essential \
        git \
        libopenblas-dev \
        ninja-build \
        cmake \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point at python3.12 for the setup.py / pip invocations below.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

###############################################################################
# PyTorch (CUDA 12.8 wheels). Resolves to the latest 2.x cu128 build, which is
# the first line with Blackwell / sm_120 support.
###############################################################################
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

###############################################################################
# Build flags. sm_120 = RTX 5090 / 5080 / 5070 (Blackwell). +PTX keeps a
# forward-compatible SASS for future Blackwell SKUs.
###############################################################################
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# Speed up the kernel compilation; tune to your host core count.
ENV MAX_JOBS=8

###############################################################################
# Clone this fork (NOT the archived NVIDIA repo) and build the extension.
# --no-build-isolation so the build uses the torch + nvcc we just installed.
# The repo ships a MinkowskiEngineBackend/__init__.py namespace stub, so the
# build_ext copy step succeeds.
###############################################################################
RUN git clone https://github.com/alpsaur/MinkowskiEngine.git /opt/MinkowskiEngine

WORKDIR /opt/MinkowskiEngine
RUN python -m pip install . --no-build-isolation

###############################################################################
# Smoke check: confirm the compiled extension imports.
###############################################################################
CMD ["python", "-c", "import MinkowskiEngine as ME; print('MinkowskiEngine', ME.__version__)"]
