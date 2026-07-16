[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/
[pypi-download]: https://img.shields.io/pypi/dm/MinkowskiEngine
[slack-badge]: https://img.shields.io/badge/slack-join%20chats-brightgreen
[slack-url]: https://join.slack.com/t/minkowskiengine/shared_invite/zt-piq2x02a-31dOPocLt6bRqOGY3U_9Sw

# Minkowski Engine

[![CI](https://github.com/alpsaur/MinkowskiEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/alpsaur/MinkowskiEngine/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/alpsaur/MinkowskiEngine?label=release%20%2B%20wheels)](https://github.com/alpsaur/MinkowskiEngine/releases/latest) [![slack chat][slack-badge]][slack-url]

> Maintained fork of [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) (upstream dormant since 2022; its PyPI package does not build on NumPy 2 / CUDA 12.8).

---

## CUDA 12.8+ / Blackwell GPU Fork

This fork adds compatibility for **CUDA Toolkit 12.8+** and **NVIDIA Blackwell architecture GPUs** (RTX 5090, 5080, 5070, etc.).

### Quick Install (v0.5.7)

**Prebuilt wheels** (fastest — no compilation; CUDA 12.8 / sm_120, built against **torch 2.9.x cu128**, Python 3.10–3.12):

```bash
# pick the wheel matching your Python from the release page
pip install https://github.com/alpsaur/MinkowskiEngine/releases/download/v0.5.7/minkowskiengine-0.5.7-cp312-cp312-linux_x86_64.whl
```

All wheels: [Releases](https://github.com/alpsaur/MinkowskiEngine/releases/latest). The wheel dynamically links your torch install — it needs a **torch 2.9.x + cu128** runtime; for other torch lines, build from source instead.

**From source** (any torch >= 2.7 cu128; ~10–20 min compile):

```bash
# Requires CUDA 12.8+ toolkit installed
pip install git+https://github.com/alpsaur/MinkowskiEngine@master
```

Or build a **CUDA 12.8 / Blackwell** Docker image:

```bash
docker build -t minkowski_engine docker
```

The build no longer wipes `build/` on every run, `python setup.py build_ext --inplace` works out of the box (a `MinkowskiEngineBackend/` namespace stub is included), and a minimal `pyproject.toml` is provided for PEP 517 installs. `FORCE_CUDA=1` (env var) forces a CUDA build on machines without a visible GPU.

### New in v0.5.7

**Fused + vectorized gather/scatter.** The copy-GEMM convolution path used to launch tiny gather/scatter kernels once per kernel offset (27x per layer for a 3^3 kernel) — profiling a real sparse U-Net showed these copies eating **56% of total GPU time** across ~358k launches per 50 steps. They are now a single vectorized gather and a single vectorized scatter-accumulate per direction (16/8/4-byte chunks per thread), and the backward input-gradient GEMM is reoriented so its scatter is coalesced. Measured on the same workload: copy-kernel GPU time **5.9s → 1.7s**, kernel launches **358k → 14k**, total GPU time per step **−38%**. Enabled by default; `ME_FUSED_COPY=0` restores the legacy path, `ME_FUSED_COPY_MAX_MB` caps staging memory (default 2048).

### New in v0.5.6

**Half precision (fp16 / bf16).** All ops (convolution, pooling, broadcast, interpolation, spmm) now run in fp16 and bf16, with GEMMs on tensor cores accumulating in fp32. Works with standard `torch.autocast`:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(sparse_input)   # ME layers cast + run in bf16 automatically
```

Typical effect on real training: **~20% lower peak GPU memory**; throughput gains grow with channel width / batch size. fp32/fp64 behavior is unchanged.

**Opt-in sync elimination.** Set `ME_LAZY_SYNC=1` to skip redundant per-op stream synchronizations (the conv-backward ones fired once *per kernel offset* — 27x per layer for a 3^3 kernel). Measured **~7–11% faster training steps** on a sparse U-Net at small batch. Default **off** (behavior identical to previous releases); assumes all ME work runs on a single CUDA stream (the PyTorch default).

```bash
ME_LAZY_SYNC=1 python train.py
```

### What's Changed

The official MinkowskiEngine repo (last updated 2022) doesn't compile with CUDA 12.8 due to:
- Deprecated `numpy.distutils` API (removed in NumPy 2.0)
- NVTX3 header namespace conflicts with CUDA 12.8's built-in headers
- Thrust library API changes requiring additional includes
- `std::shared_ptr` ambiguity with `cuda::std::shared_ptr`

This fork applies community workarounds from issues [#543](https://github.com/NVIDIA/MinkowskiEngine/issues/543), [#594](https://github.com/NVIDIA/MinkowskiEngine/issues/594), and [#596](https://github.com/NVIDIA/MinkowskiEngine/issues/596).

**Tested on:** RTX 5090 (sm_120), CUDA 12.8, PyTorch 2.9 (cu128), Python 3.12, GCC 13. Half-precision paths validated against fp32 references on-GPU (`tests/python/half_precision.py`); the released wheels are import- and smoke-tested on real Blackwell hardware.

> All changes live on the default `master` branch. The previous `cuda12-compat` branch was merged in and removed — install from `master`. BLAS is auto-detected at build time (OpenBLAS by default), so `--install-option` is not required.

---

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

## News

- 2026-07 **v0.5.7 — fused gather/scatter: 38% less GPU time per training step.** Profiling showed the per-kernel-offset copy kernels were the engine's biggest cost (56% of GPU time); they are now fused and vectorized. On by default with an opt-out (`ME_FUSED_COPY=0`).
- 2026-07 **v0.5.6 released — [prebuilt wheels](https://github.com/alpsaur/MinkowskiEngine/releases/tag/v0.5.6), half precision, faster training.** One-line install for Blackwell GPUs (no more compiling from source). fp16/bf16 now work across all ops with tensor-core GEMMs and `torch.autocast` — in our training runs this cut peak GPU memory by ~20%. An opt-in `ME_LAZY_SYNC=1` flag removes redundant GPU synchronizations for ~7–11% faster training steps.
- 2026-07 **v0.5.5 — the fork is now a maintained project**: CI (build + tests on every change), working `pip install` / Docker / [docs](https://alpsaur.github.io/MinkowskiEngine/), a repaired test suite, and packaging/build-system fixes throughout.
- 2026-07 All CUDA 12.8 / Blackwell and NumPy 2.0 fixes unified on the default `master` branch — just install from `master`.
- 2025-12 The engine builds again on modern toolchains: NumPy 2.0 support (upstream relied on the removed `numpy.distutils`) and CUDA 12.8 / Blackwell (RTX 50-series) compatibility.
- 2021-08-11 Docker installation instruction added
- 2021-08-06 All installation errors with pytorch 1.8 and 1.9 have been resolved.
- 2021-04-08 Due to recent errors in [pytorch 1.8 + CUDA 11](https://github.com/NVIDIA/MinkowskiEngine/issues/330), it is recommended to use [anaconda for installation](#anaconda).
- 2020-12-24 v0.5 is now available! The new version provides CUDA accelerations for all coordinate management functions.

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://nvidia.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.classification_modelnet40`                                                          |
| Reconstruction        | <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://nvidia.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                       |
| Detection             | <img src="https://nvidia.github.io/MinkowskiEngine/_images/detection_3d_net.png">                                                                                               |


## Sparse Tensor Networks: Neural Networks for Spatially Sparse Tensors

Compressing a neural network to speedup inference and minimize memory footprint has been studied widely. One of the popular techniques for model compression is pruning the weights in convnets, is also known as [*sparse convolutional networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf). Such parameter-space sparsity used for model compression compresses networks that operate on dense tensors and all intermediate activations of these networks are also dense tensors.

However, in this work, we focus on [*spatially* sparse data](https://arxiv.org/abs/1409.6070), in particular, spatially sparse high-dimensional inputs and 3D data and convolution on the surface of 3D objects, first proposed in [Siggraph'17](https://wang-ps.github.io/O-CNN.html). We can also represent these data as sparse tensors, and these sparse tensors are commonplace in high-dimensional problems such as 3D perception, registration, and statistical data. We define neural networks specialized for these inputs as *sparse tensor networks*  and these sparse tensor networks process and generate sparse tensors as outputs. To construct a sparse tensor network, we build all standard neural network layers such as MLPs, non-linearities, convolution, normalizations, pooling operations as the same way we define them on a dense tensor and implemented in the Minkowski Engine.

We visualized a sparse tensor network operation on a sparse tensor, convolution, below. The convolution layer on a sparse tensor works similarly to that on a dense tensor. However, on a sparse tensor, we compute convolution outputs on a few specified points which we can control in the [generalized convolution](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html). For more information, please visit [the documentation page on sparse tensor networks](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html) and [the terminology page](https://nvidia.github.io/MinkowskiEngine/terminology.html).

| Dense Tensor                                                                | Sparse Tensor                                                                |
|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_dense.gif"> | <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_sparse.gif"> |

--------------------------------------------------------------------------------

## Features

- Unlimited high-dimensional sparse tensor support
- All standard neural network layers (Convolution, Pooling, Broadcast, etc.)
- Mixed precision: fp16 / bf16 with `torch.autocast`, tensor-core GEMMs (v0.5.6+)
- Dynamic computation graph
- Custom kernel shapes
- Multi-GPU training (upstream feature; not yet validated on this fork)
- Multi-threaded kernel map
- Multi-threaded compilation
- Highly-optimized GPU kernels


## Requirements

**For this fork (CUDA 12.8+ / Blackwell):**
- Linux (tested on Ubuntu 24.04 under WSL2); CUDA Toolkit **12.8 or newer** — must match the CUDA PyTorch was built with
- PyTorch **>= 2.7** with cu128 (CI tests 2.7 and 2.9; prebuilt wheels require **2.9.x**)
- Python **3.10–3.13** (wheels: 3.10–3.12)
- GCC **11–13** (required by CUDA 12.8; source builds only)
- `ninja` (for compilation; source builds only)
- OpenBLAS — auto-detected at build time

**Original engine support (reference only, not validated on this fork):** CUDA >= 10.1, PyTorch >= 1.7, Python >= 3.6, GCC >= 7.4.0, Ubuntu >= 14.04. You must always match the CUDA version PyTorch uses with the one used to compile MinkowskiEngine.


## Installation

> **Blackwell / RTX 50-series users:** use the [Quick Install](#cuda-128--blackwell-gpu-fork) at the top of this README (`pip install git+...@master`). The detailed sections below are retained from upstream for older CUDA configurations; BLAS is auto-detected, so `--install-option` is not required (and is unsupported on modern pip).

You can install the Minkowski Engine with `pip`, with anaconda, or on the system directly. If you experience issues installing the package, please check the [installation wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation).
If you cannot find a relevant problem, please report the issue on [this fork's issue page](https://github.com/alpsaur/MinkowskiEngine/issues).

- [PIP](#pip) installation
- [Conda](#anaconda) installation
- [Python](#system-python) installation
- [Docker](#docker) installation


### Pip

> **Note:** the `MinkowskiEngine` package on PyPI is the unmaintained upstream 0.5.4 — it does **not** build on NumPy 2 / CUDA 12.8. Use this fork's [prebuilt wheels](https://github.com/alpsaur/MinkowskiEngine/releases/tag/v0.5.6) or install from source below. (`--install-option` is also no longer supported by modern pip.)

First, install pytorch following the [instruction](https://pytorch.org). Next, install `openblas`.

```
sudo apt install build-essential python3-dev libopenblas-dev
pip install torch ninja

# From this fork's latest source (BLAS auto-detected):
pip install -U git+https://github.com/alpsaur/MinkowskiEngine@master
```

Build-time knobs are environment variables now that pip no longer forwards setup flags:

```
# export CXX=c++                       # use a different C++ compiler
# export CUDA_HOME=/usr/local/cuda-12.8  # select the CUDA toolkit explicitly
# export FORCE_CUDA=1                  # force a CUDA build when no GPU is visible (e.g. containers/CI)
# export TORCH_CUDA_ARCH_LIST="12.0+PTX"  # target specific compute capabilities
# export MAX_COMPILATION_THREADS=24    # parallel compile jobs
pip install -U git+https://github.com/alpsaur/MinkowskiEngine@master
```

### Anaconda

> **Legacy (upstream, pre-CUDA-12):** kept for reference for older CUDA 10.2 / 11.x setups. Not validated on this fork, and the `--install-option` flags below no longer work on modern pip — on current toolchains use the [Quick Install](#cuda-128--blackwell-gpu-fork) instead.

MinkowskiEngine supports both CUDA 10.2 and cuda 11.1, which work for most of latest pytorch versions.
#### CUDA 10.2

We recommend `python>=3.6` for installation.
First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
sudo apt install g++-7  # For CUDA 10.2, must use GCC < 8
# Make sure `g++-7 --version` is at least 7.4.0
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=10.2 -c pytorch -c nvidia

# Install MinkowskiEngine
export CXX=g++-7
# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 10.2
# export CUDA_HOME=/usr/local/cuda-10.2
pip install -U git+https://github.com/alpsaur/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want local MinkowskiEngine
git clone https://github.com/alpsaur/MinkowskiEngine.git
cd MinkowskiEngine
export CXX=g++-7
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### CUDA 11.X

We recommend `python>=3.6` for installation.
First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# Install MinkowskiEngine

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1
pip install -U git+https://github.com/alpsaur/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want local MinkowskiEngine
git clone https://github.com/alpsaur/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### System Python

Like the anaconda installation, make sure that you install pytorch with the same CUDA version that `nvcc` uses.

```
# install system requirements
sudo apt install build-essential python3-dev libopenblas-dev

# Skip if you already have pip installed on your python3
curl https://bootstrap.pypa.io/get-pip.py | python3

# Get pip and install python requirements
python3 -m pip install torch numpy ninja

git clone https://github.com/alpsaur/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install
# To specify blas, CXX, CUDA_HOME and force CUDA installation, use the following command
# export CXX=c++; export CUDA_HOME=/usr/local/cuda-11.1; python setup.py install --blas=openblas --force_cuda
```

### Docker

```
git clone https://github.com/alpsaur/MinkowskiEngine
cd MinkowskiEngine
docker build -t minkowski_engine docker
```

Once the docker is built, check it loads MinkowskiEngine correctly.

```
docker run MinkowskiEngine python3 -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
```

## CPU only build and BLAS configuration (MKL)

The Minkowski Engine supports CPU only build on other platforms that do not have NVidia GPUs. Please refer to [quick start](https://nvidia.github.io/MinkowskiEngine/quick_start.html) for more details.


## Quick Start

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`MinkowskiEngine.utils.sparse_quantize`).


### Creating a Network

```python
import torch.nn as nn
import MinkowskiEngine as ME

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)
```

### Forward and backward using the custom network

```python
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    input = ME.SparseTensor(feat, coordinates=coords)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)
```

## Discussion and Documentation

API documentation for this fork is published at
[alpsaur.github.io/MinkowskiEngine](https://alpsaur.github.io/MinkowskiEngine/)
(rebuilt from `master` by CI). The [upstream documentation
page](http://nvidia.github.io/MinkowskiEngine/) remains a valid reference for
general concepts and the original API.

For issues not listed on the API and feature requests, feel free to submit
an issue on [this fork's issue
page](https://github.com/alpsaur/MinkowskiEngine/issues).

To profile where time goes inside ME ops and coordinate-map construction, see
[Profiling MinkowskiEngine](docs/profiling.md).


## Performance

For the full set of runtime and build knobs — TF32, mixed precision (fp16/bf16 via `torch.autocast`), the `ME_LAZY_SYNC=1` sync-elimination flag, `OMP_NUM_THREADS` tuning, and `torch.cuda.empty_cache()` for varying point counts — see the [Performance Guide](https://alpsaur.github.io/MinkowskiEngine/performance.html) (`docs/performance.md`). The headline items: enable TF32 on Ampere+ for ~15% faster steps, run bf16 autocast for ~20% lower peak memory, and set `ME_LAZY_SYNC=1` for another ~7–11% speedup at small batch.


## Known Issues

### Specifying CUDA architecture list

In some cases, you need to explicitly specify which compute capability your GPU uses. The default list might not contain your architecture.

```bash
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"; python setup.py install --force_cuda

# For Blackwell (RTX 5090 / RTX 50-series), include compute capability 12.0:
export TORCH_CUDA_ARCH_LIST="8.9 9.0 12.0+PTX"; python setup.py install --force_cuda
```

### Unhandled Out-Of-Memory thrust::system exception

There is [a known issue](https://github.com/NVIDIA/thrust/issues/1448) in thrust with CUDA 10 that leads to an unhandled thrust exception. Please refer to the [issue](https://github.com/NVIDIA/MinkowskiEngine/issues/357) for detail.

### Too much GPU memory usage or Frequent Out of Memory

There are a few causes for this error.

1. Out of memory during a long running training

MinkowskiEngine is a specialized library that can handle different number of points or different number of non-zero elements at every iteration during training, which is common in point cloud data.
However, pytorch is implemented assuming that the number of point, or size of the activations do not change at every iteration. Thus, the GPU memory caching used by pytorch can result in unnecessarily large memory consumption.

Specifically, pytorch caches chunks of memory spaces to speed up allocation used in every tensor creation. If it fails to find the memory space, it splits an existing cached memory or allocate new space if there's no cached memory large enough for the requested size. Thus, every time we use different number of point (number of non-zero elements) with pytorch, it either split existing cache or reserve new memory. If the cache is too fragmented and allocated all GPU space, it will raise out of memory error.

**To prevent this, you must clear the cache at regular interval with `torch.cuda.empty_cache()`.**

### CUDA 11.1 Installation

```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override

# Install MinkowskiEngine with CUDA 11.1
export CUDA_HOME=/usr/local/cuda-11.1; pip install MinkowskiEngine -v --no-deps
```

### Running the MinkowskiEngine on nodes with a large number of CPUs

The MinkowskiEngine uses OpenMP to parallelize the kernel map generation. However, when the number of threads used for parallelization is too large (e.g. OMP_NUM_THREADS=80), the efficiency drops rapidly as all threads simply wait for multithread locks to be released.
In such cases, set the number of threads used for OpenMP. Usually, any number below 24 would be fine, but search for the optimal setup on your system.

```
export OMP_NUM_THREADS=<number of threads to use>; python <your_program.py>
```

## Citing Minkowski Engine

If you use the Minkowski Engine, please cite:

- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755), [[pdf]](https://arxiv.org/pdf/1904.08755.pdf)

```
@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}
```

For multi-threaded kernel map generation, please cite:

```
@inproceedings{choy2019fully,
  title={Fully Convolutional Geometric Features},
  author={Choy, Christopher and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8958--8966},
  year={2019}
}
```

For strided pooling layers for high-dimensional convolutions, please cite:

```
@inproceedings{choy2020high,
  title={High-dimensional Convolutional Networks for Geometric Pattern Recognition},
  author={Choy, Christopher and Lee, Junha and Ranftl, Rene and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

For generative transposed convolution, please cite:

```
@inproceedings{gwak2020gsdn,
  title={Generative Sparse Detection Networks for 3D Single-shot Object Detection},
  author={Gwak, JunYoung and Choy, Christopher B and Savarese, Silvio},
  booktitle={European conference on computer vision},
  year={2020}
}
```


## Unittest

The suite runs under pytest (`pytest` from the repo root; `pytest.ini` targets `tests/python/`). GPU-dependent tests skip automatically on CPU-only builds, and `tests/python/half_precision.py` covers the fp16/bf16 paths against fp32 references (CUDA required). CI runs the CPU matrix on every push/PR.

## Projects using Minkowski Engine

Please feel free to update [the wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage) to add your projects!

- [Projects using MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage)

- Segmentation: [3D and 4D Spatio-Temporal Semantic Segmentation, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- Representation Learning: [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- 3D Registration: [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- 3D Registration: [Deep Global Registration, CVPR'20](https://arxiv.org/abs/2004.11540)
- Pattern Recognition: [High-Dimensional Convolutional Networks for Geometric Pattern Recognition, CVPR'20](https://arxiv.org/abs/2005.08144)
- Detection: [Generative Sparse Detection Networks for 3D Single-shot Object Detection, ECCV'20](https://arxiv.org/abs/2006.12356)
- Image matching: [Sparse Neighbourhood Consensus Networks, ECCV'20](https://www.di.ens.fr/willow/research/sparse-ncnet/)
