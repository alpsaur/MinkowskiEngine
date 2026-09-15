"""Pure-Python build/runtime checks; safe to use before loading the extension."""

import importlib.util
import json
from pathlib import Path
import re


_REBUILD = (
    "Install the matching torch/CUDA runtime, or rebuild from a MinkowskiEngine "
    "checkout with this interpreter: python -m pip install --no-build-isolation "
    "--no-deps --force-reinstall ."
)


def read_build_info():
    """Read the manifest adjacent to the backend selected by Python imports."""
    spec = importlib.util.find_spec("MinkowskiEngineBackend")
    if spec is not None:
        for location in spec.submodule_search_locations or ():
            manifest = Path(location) / "_build_info.json"
            if manifest.is_file():
                return json.loads(manifest.read_text())
    return None


def _minor(version):
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match is None:
        raise ImportError(f"Cannot interpret torch version {version!r}. {_REBUILD}")
    return tuple(map(int, match.groups()))


def check_build_compatibility(info, torch):
    """Reject libtorch minor/ABI and GPU CUDA-runtime mismatches without CUDA calls.

    Patch-level torch differences are supported. CPU-only extensions may run
    with either CPU or CUDA torch, provided the libtorch minor and ABI match.
    """
    problems = []
    if _minor(info["torch_version"]) != _minor(torch.__version__):
        problems.append(
            f"built against torch {info['torch_version']}, running torch {torch.__version__}"
        )
    runtime_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    if info["cxx11_abi"] != runtime_abi:
        problems.append(f"C++ ABI {info['cxx11_abi']} != runtime ABI {runtime_abi}")
    if not info["cpu_only"] and info["torch_cuda"] != torch.version.cuda:
        problems.append(
            f"built for torch CUDA {info['torch_cuda']}, running torch CUDA {torch.version.cuda}"
        )
    if problems:
        raise ImportError(
            "Incompatible MinkowskiEngine extension: "
            + "; ".join(problems)
            + ". "
            + _REBUILD
        )
