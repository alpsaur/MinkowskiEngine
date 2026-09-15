"""The public helper must work without pytest monkeypatching PyTorch."""

import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from MinkowskiEngine.utils.gradcheck import gradcheck


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return 2 * x * grad


class TestGradcheckCompatibility(unittest.TestCase):
    def test_real_gradcheck(self):
        x = torch.randn(3, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(Square, (x,), check_grad_dtypes=True))

    def test_supported_options_are_forwarded(self):
        module = importlib.import_module("MinkowskiEngine.utils.gradcheck")
        with patch.object(module, "_gradcheck", return_value=True) as check:
            gradcheck(Square, (), nondet_tol=0.125, check_grad_dtypes=True)
        self.assertEqual(check.call_args.kwargs["nondet_tol"], 0.125)
        self.assertTrue(check.call_args.kwargs["check_grad_dtypes"])
        self.assertNotIn("check_sparse_nnz", check.call_args.kwargs)
        with self.assertRaisesRegex(ValueError, "as_sparse_gradcheck"):
            gradcheck(Square, (), check_sparse_nnz=True)

    def test_standalone_process(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    [
                        "import torch",
                        "from MinkowskiEngine.utils.gradcheck import gradcheck",
                        "from types import SimpleNamespace",
                        "x = torch.randn(3, dtype=torch.double, requires_grad=True)",
                        "assert gradcheck(SimpleNamespace(apply=lambda x: x.square()), (x,))",
                    ]
                ),
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
