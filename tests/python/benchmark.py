"""Small subprocess smoke tests for the benchmark's real training path."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"


class TestBenchmark(unittest.TestCase):
    def run_benchmark(self, *args):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--points",
                "32",
                "--channels",
                "4",
                "--warmup",
                "1",
                "--iterations",
                "2",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["schema_version"], 1)
        self.assertGreater(data["timings"]["step_wall_ms"]["median"], 0)
        self.assertEqual(data["active_points"], 32)
        return data

    def test_cpu_cold_and_warm(self):
        cold = self.run_benchmark("--device", "cpu", "--map-mode", "cold")
        warm = self.run_benchmark(
            "--device", "cpu", "--map-mode", "warm", "--deterministic"
        )
        self.assertAlmostEqual(cold["last_loss"], warm["last_loss"], places=6)
        self.assertIsNone(cold["peak_allocated_bytes"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_mixed_precision(self):
        data = self.run_benchmark(
            "--device", "cuda", "--precision", "bf16", "--lazy-sync"
        )
        self.assertGreater(data["peak_allocated_bytes"], 0)
        self.assertEqual(data["configuration"]["precision"], "bf16")

    def test_cpu_half_rejected(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--device", "cpu", "--precision", "fp16"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --device cuda", result.stderr)


if __name__ == "__main__":
    unittest.main()
