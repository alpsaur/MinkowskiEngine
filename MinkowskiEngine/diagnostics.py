"""Environment diagnostics, also runnable as ``python MinkowskiEngine/diagnostics.py``."""

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys


def _command(args):
    if not args or shutil.which(args[0]) is None:
        return "not installed"
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def print_diagnostics():
    """Print runtime and compiled build facts without requiring nvcc or a GPU."""
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "nvcc": _command(["nvcc", "--version"]),
        "gpu": _command(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "compiler": _command(
            shlex.split(os.environ.get("CXX", os.environ.get("CC", "c++")))
            + ["--version"]
        ),
    }
    try:
        import torch

        info["torch_runtime"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        }
    except ImportError as exc:
        info["torch_error"] = str(exc)
    try:
        import MinkowskiEngine as ME

        info["minkowski_version"] = ME.__version__
        # Run-by-path can import an older installed ME, not this checkout.
        # Missing provenance is unknown, not this checkout's sidecar metadata.
        get_build_info = getattr(ME, "get_build_info", None)
        info["extension_build"] = get_build_info() if get_build_info else None
        info["compiled_nvcc"] = ME.cuda_version()
        info["compiled_cudart"] = ME.cudart_version()
    except ImportError as exc:
        info["minkowski_error"] = str(exc)
        # Still report build facts when an incompatible native module cannot
        # be imported. This script's directory is on sys.path when run by path.
        try:
            from _compat import read_build_info

            info["extension_build"] = read_build_info()
        except (ImportError, ValueError, OSError) as metadata_error:
            info["metadata_error"] = str(metadata_error)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    print_diagnostics()
