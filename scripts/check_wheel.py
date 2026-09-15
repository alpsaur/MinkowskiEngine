#!/usr/bin/env python3
"""Install each wheel into an isolated target and smoke-test outside the checkout.

Uses this interpreter's existing torch/numpy runtime, but never installs into or
modifies that environment. Pass --cuda only on trusted GPU hosts.
"""

import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile


SMOKE = r"""
import importlib.metadata
from pathlib import Path
import sys
import torch
import MinkowskiEngine as ME
import MinkowskiEngineBackend._C as backend

root = Path(sys.argv[1]).resolve()
for module in (ME, backend):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
info = ME.get_build_info()
requirements = importlib.metadata.requires("MinkowskiEngine")
minor = ".".join(info["torch_version"].split("+")[0].split(".")[:2])
assert any(req.replace(" ", "").startswith("torch~=" + minor + ".0")
           for req in requirements), requirements
coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.int32)
for device in (["cpu", "cuda"] if sys.argv[2] == "cuda" else ["cpu"]):
    features = torch.ones(3, 4, device=device, requires_grad=True)
    x = ME.SparseTensor(features, coords.to(device))
    conv = ME.MinkowskiConvolution(4, 4, 3, dimension=3).to(device)
    ME.set_deterministic(True)
    y = conv(x)
    assert y.coordinate_map_key == x.coordinate_map_key
    (y + x).F.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    print("wheel smoke OK:", device, ME.__version__, info)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path, nargs="+")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--tests",
        action="store_true",
        help="also run the copied regression suite against the installed wheel (requires pytest)",
    )
    args = parser.parse_args()
    for wheel in args.wheels:
        wheel = wheel.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="me-wheelcheck-") as directory:
            target = Path(directory) / "site"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-compile",
                    "--target",
                    str(target),
                    str(wheel),
                ],
                check=True,
                cwd=directory,
            )
            env = dict(os.environ, PYTHONPATH=str(target), PYTHONDONTWRITEBYTECODE="1")
            if not args.cuda:
                env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    SMOKE,
                    str(target),
                    "cuda" if args.cuda else "cpu",
                ],
                check=True,
                cwd=directory,
                env=env,
                timeout=120,
            )
            if args.tests:
                repo = Path(__file__).resolve().parents[1]
                # conftest adds its parent to sys.path: copy the tests, not
                # the package, so this cannot mask a broken installed wheel.
                shutil.copytree(
                    repo / "tests",
                    Path(directory) / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"),
                )
                shutil.copy2(repo / "pytest.ini", directory)
                scripts = Path(directory) / "scripts"
                scripts.mkdir()
                shutil.copy2(repo / "scripts" / "benchmark.py", scripts)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-p",
                        "no:cacheprovider",
                        "-o",
                        "addopts=--assert=plain",
                        "-q",
                        "--disable-warnings",
                    ],
                    check=True,
                    cwd=directory,
                    env=env,
                    timeout=900,
                )


if __name__ == "__main__":
    main()
