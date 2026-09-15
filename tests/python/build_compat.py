"""Build provenance and compatibility checks require no CUDA execution."""

import contextlib
import copy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import MinkowskiEngine as ME
import MinkowskiEngineBackend._C as backend
from MinkowskiEngine._compat import check_build_compatibility, read_build_info


def runtime(version="2.9.1+cu128", cuda="12.8", abi=True):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=abi),
    )


class TestBuildCompatibility(unittest.TestCase):
    def test_manifest_matches_native_binary(self):
        info = ME.get_build_info()
        self.assertEqual(info, json.loads(backend._build_info))
        self.assertEqual(info, read_build_info())
        check_build_compatibility(info, torch)
        info["torch_version"] = "0.0.0"
        self.assertNotEqual(info, ME.get_build_info())

    def test_minor_cuda_and_abi_contract(self):
        info = dict(
            torch_version="2.9.0+cu128",
            torch_cuda="12.8",
            cxx11_abi=True,
            cpu_only=False,
            cuda_arch_flags=[],
        )
        check_build_compatibility(info, runtime())  # patch upgrades are allowed
        for incompatible in (
            runtime(version="2.10.0"),
            runtime(cuda="13.0"),
            runtime(cuda=None),
            runtime(abi=False),
        ):
            with self.assertRaisesRegex(ImportError, "--no-build-isolation"):
                check_build_compatibility(info, incompatible)
        cpu = copy.copy(info)
        cpu["cpu_only"] = True
        check_build_compatibility(cpu, runtime(cuda=None))
        check_build_compatibility(cpu, runtime(cuda="13.0"))

    def test_diagnostics_without_external_tools(self):
        with (
            patch("shutil.which", return_value=None),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            ME.print_diagnostics()
        info = json.loads(out.getvalue())
        self.assertEqual(info["extension_build"], ME.get_build_info())
        self.assertEqual(info["nvcc"], "not installed")
        self.assertEqual(info["torch_runtime"]["version"], torch.__version__)


if __name__ == "__main__":
    unittest.main()
