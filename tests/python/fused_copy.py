# Copyright (c) 2026 alpsaur/MinkowskiEngine contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""A/B tests for the fused + vectorized copy-GEMM gather/scatter path.

Runs the same convolution forward+backward with ME_FUSED_COPY=1 (fused, the
default) and ME_FUSED_COPY=0 (legacy per-offset loop) and compares outputs and
gradients. The env var is read per call on the C++ side, so both paths can be
exercised in one process.

Exact bitwise equality between the two paths is not attainable, even at fp64:
the fused scatter accumulates contributions from all kernel offsets through
atomics in arbitrary order, while the legacy loop adds offsets sequentially
(floating-point addition is not associative). The fused backward additionally
reorients the input-gradient GEMM (same math, different cublas reduction
order). Tolerances below are therefore tight but nonzero for float, and match
tests/python/half_precision.py for the 16-bit dtypes, whose storage rounding
dominates.
"""
import os
import unittest

import torch

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from MinkowskiEngineBackend._C import ConvolutionMode

# `atol` is scaled by each tensor's max magnitude before use: every compared
# entry is a same-length floating-point reduction, so the reordering noise
# floor is proportional to the reduction magnitude (~eps * sqrt(terms) * max),
# not to the (possibly cancelled-to-zero) entry value itself. bf16 is loosest:
# the weight gradient is accumulated across offsets through bf16 storage
# (8 mantissa bits) in both paths, and the upstream gradient entering the
# comparison already differs at bf16 storage precision.
TOLERANCES = {
    torch.float64: dict(rtol=1e-10, atol=1e-12),
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=2e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=1e-1, atol=5e-2),
}

# widths cover the odd/scalar path (3) and the vector paths (32..256)
CHANNELS = [3, 32, 64, 128, 256]
KERNEL_SIZES = [2, 3, 5]
DIMENSIONS = [2, 3]


def _random_coordinates(nrows, batch, D, max_coord, seed):
    g = torch.Generator().manual_seed(seed)
    per_batch = []
    for b in range(batch):
        c = torch.randint(0, max_coord, (nrows, D), generator=g, dtype=torch.int32)
        c = torch.unique(c, dim=0)
        per_batch.append(
            torch.cat([torch.full((c.size(0), 1), b, dtype=torch.int32), c], 1)
        )
    return torch.cat(per_batch, 0)


def _run_conv(conv, coords, base_feats, fused, tensor_stride=1):
    """One forward+backward with the fused path on or off.

    Returns (out_feats, input_grad, kernel_grad), detached copies.
    """
    os.environ["ME_FUSED_COPY"] = "1" if fused else "0"
    try:
        feats = base_feats.clone().requires_grad_()
        x = SparseTensor(feats, coordinates=coords, tensor_stride=tensor_stride)
        if conv.kernel.grad is not None:
            conv.kernel.grad = None
        out = conv(x)
        # deterministic unit-variance upstream gradient (same seed -> same
        # values on both runs)
        g = torch.empty_like(out.F, dtype=torch.float32)
        g.normal_(generator=torch.Generator(device=g.device).manual_seed(1234))
        out.F.backward(g.to(out.F.dtype))
        torch.cuda.synchronize()
        return (
            out.F.detach().clone(),
            feats.grad.detach().clone(),
            conv.kernel.grad.detach().clone(),
        )
    finally:
        os.environ.pop("ME_FUSED_COPY", None)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedCopyAB(unittest.TestCase):
    """Fused vs legacy copy-GEMM must agree on outputs and gradients."""

    def _compare(self, dtype, nchannel, kernel_size, D, transpose=False,
                 max_mb=None, seed=0):
        device = torch.device("cuda")
        torch.manual_seed(seed)
        coords = _random_coordinates(
            nrows=2000, batch=2, D=D, max_coord=24, seed=seed + 17
        ).to(device)
        # probe once to get the deduplicated row count
        probe = SparseTensor(
            torch.zeros(coords.size(0), nchannel, device=device), coordinates=coords
        )
        n = probe.F.size(0)
        base = torch.randn(n, nchannel, device=device).to(dtype)

        cls = ME.MinkowskiConvolutionTranspose if transpose else ME.MinkowskiConvolution
        conv = (
            cls(
                nchannel,
                nchannel,
                kernel_size=kernel_size,
                stride=2 if transpose else 1,
                bias=False,
                convolution_mode=ConvolutionMode.COPY_GEMM,
                dimension=D,
            )
            .to(device)
            .to(dtype)
        )
        tensor_stride = 1
        if transpose:
            # transpose conv consumes a strided tensor; pool down first
            pool = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D).to(
                device
            )
            probe = pool(probe)
            coords = probe.C
            tensor_stride = 2
            n = probe.F.size(0)
            base = torch.randn(n, nchannel, device=device).to(dtype)

        if max_mb is not None:
            os.environ["ME_FUSED_COPY_MAX_MB"] = str(max_mb)
        try:
            out_f, gin_f, gk_f = _run_conv(conv, coords, base, True, tensor_stride)
            out_l, gin_l, gk_l = _run_conv(conv, coords, base, False, tensor_stride)
        finally:
            os.environ.pop("ME_FUSED_COPY_MAX_MB", None)

        tol = TOLERANCES[dtype]
        ctx = f"dtype={dtype}, C={nchannel}, k={kernel_size}, D={D}, tr={transpose}"
        for name, a, b in [
            ("output", out_f, out_l),
            ("input_grad", gin_f, gin_l),
            ("kernel_grad", gk_f, gk_l),
        ]:
            a, b = a.float(), b.float()
            atol = tol["atol"] * max(b.abs().max().item(), 1.0)
            self.assertTrue(
                torch.allclose(a, b, rtol=tol["rtol"], atol=atol),
                msg=f"{name} mismatch ({ctx}): "
                f"max abs diff {(a - b).abs().max().item():.3e}, "
                f"max |ref| {b.abs().max().item():.3e}, atol {atol:.3e}",
            )

    def test_fp32_matrix(self):
        for D in DIMENSIONS:
            for k in KERNEL_SIZES:
                for c in CHANNELS:
                    with self.subTest(dtype="fp32", D=D, kernel_size=k, channels=c):
                        self._compare(torch.float32, c, k, D)

    def test_fp16_matrix(self):
        for D in DIMENSIONS:
            for k in KERNEL_SIZES:
                for c in CHANNELS:
                    with self.subTest(dtype="fp16", D=D, kernel_size=k, channels=c):
                        self._compare(torch.float16, c, k, D)

    def test_fp64(self):
        for c in [3, 64]:
            with self.subTest(channels=c):
                self._compare(torch.float64, c, 3, 3)

    def test_bf16(self):
        for c in [3, 64]:
            with self.subTest(channels=c):
                self._compare(torch.bfloat16, c, 3, 3)

    def test_transpose_conv(self):
        for dtype in [torch.float32, torch.float16]:
            with self.subTest(dtype=dtype):
                self._compare(dtype, 64, 2, 3, transpose=True)

    def test_group_chunking(self):
        # A 1 MB staging cap forces the fused path to split the kernel
        # offsets into multiple groups; results must be unchanged.
        self._compare(torch.float32, 128, 3, 3, max_mb=1)
        self._compare(torch.float16, 128, 3, 3, max_mb=1)


if __name__ == "__main__":
    unittest.main()
