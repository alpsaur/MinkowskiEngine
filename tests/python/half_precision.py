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
"""fp16 / bf16 feature support.

Each op runs forward + backward at reduced precision and is compared against
an fp32 reference computed on the *exact upcast* of the same 16-bit inputs and
weights (so the only differences are compute/storage rounding, not input
rounding — this also keeps max-pool argmax selections identical). Reference
and reduced-precision tensors share one coordinate manager so row orderings
match.
"""
import unittest

import torch

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

# Storage rounding of outputs/gradients dominates (GEMMs accumulate in fp32
# via cublasGemmEx / cusparseSpMM with a CUDA_R_32F compute type).
TOLERANCES = {
    torch.float16: dict(rtol=2e-2, atol=2e-3),
    torch.bfloat16: dict(rtol=1e-1, atol=1e-2),
}
HALF_DTYPES = (torch.float16, torch.bfloat16)


def _random_coordinates(nrows=1000, batch=2, D=3, max_coord=30, seed=42):
    g = torch.Generator().manual_seed(seed)
    per_batch = []
    for b in range(batch):
        c = torch.randint(0, max_coord, (nrows, D), generator=g, dtype=torch.int32)
        c = torch.unique(c, dim=0)
        per_batch.append(
            torch.cat([torch.full((c.size(0), 1), b, dtype=torch.int32), c], 1)
        )
    return torch.cat(per_batch, 0)


def _paired_inputs(coords, nchannel, dtype, device, seed=7):
    """A 16-bit leaf and an fp32 leaf holding the exact same values, both as
    SparseTensors sharing one coordinate manager."""
    torch.manual_seed(seed)
    probe = SparseTensor(
        torch.zeros(coords.size(0), nchannel, device=device),
        coordinates=coords.to(device),
    )
    n = probe.F.size(0)
    base = torch.randn(n, nchannel, device=device).to(dtype)
    half_leaf = base.clone().requires_grad_()
    ref_leaf = base.float().clone().requires_grad_()
    common = dict(
        coordinate_map_key=probe.coordinate_map_key,
        coordinate_manager=probe.coordinate_manager,
    )
    return SparseTensor(half_leaf, **common), SparseTensor(ref_leaf, **common), \
        half_leaf, ref_leaf


def _paired_modules(module_factory, dtype, device):
    """16-bit module and an fp32 module whose weights are the exact upcast.

    ME modules hold pybind-backed attributes that do not survive deepcopy, so
    build two instances and copy weights by name.
    """
    half_mod = module_factory().to(device=device, dtype=dtype)
    ref_mod = module_factory().to(device=device)
    half_params = dict(half_mod.named_parameters())
    with torch.no_grad():
        for name, p in ref_mod.named_parameters():
            p.copy_(half_params[name].float())
    return half_mod, ref_mod


class HalfPrecisionOpTestBase(unittest.TestCase):
    device = "cuda"

    def assert_close(self, half_tensor, ref_tensor, dtype, what):
        self.assertTrue(
            torch.isfinite(half_tensor.float()).all(), f"{what} not finite ({dtype})"
        )
        self.assertTrue(
            torch.allclose(half_tensor.float(), ref_tensor, **TOLERANCES[dtype]),
            f"{what} mismatch ({dtype}): max abs err "
            f"{(half_tensor.float() - ref_tensor).abs().max().item():.3e}",
        )

    def _run_module(self, module_factory, dtype, nchannel=8, needs_weights=False):
        coords = _random_coordinates()
        x_half, x_ref, half_leaf, ref_leaf = _paired_inputs(
            coords, nchannel, dtype, self.device
        )
        half_mod, ref_mod = _paired_modules(module_factory, dtype, self.device)

        y_half = half_mod(x_half)
        y_ref = ref_mod(x_ref)
        self.assertEqual(y_half.F.dtype, dtype)

        g = torch.randn(y_ref.F.shape, device=self.device).to(dtype)
        (y_half.F * g).sum().backward()
        (y_ref.F * g.float()).sum().backward()

        self.assert_close(y_half.F, y_ref.F, dtype, "output")
        self.assert_close(half_leaf.grad, ref_leaf.grad, dtype, "input grad")
        if needs_weights:
            self.assert_close(
                half_mod.kernel.grad, ref_mod.kernel.grad, dtype, "kernel grad"
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHalfPrecisionOps(HalfPrecisionOpTestBase):
    def test_convolution(self):
        for dtype in HALF_DTYPES:
            self._run_module(
                lambda: ME.MinkowskiConvolution(
                    8, 16, kernel_size=3, stride=2, dimension=3
                ),
                dtype,
                needs_weights=True,
            )

    def test_convolution_transpose(self):
        for dtype in HALF_DTYPES:
            self._run_module(
                lambda: ME.MinkowskiConvolutionTranspose(
                    8, 16, kernel_size=3, stride=1, dimension=3
                ),
                dtype,
                needs_weights=True,
            )

    def test_avg_pooling(self):
        for dtype in HALF_DTYPES:
            self._run_module(
                lambda: ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3),
                dtype,
            )

    def test_sum_pooling(self):
        for dtype in HALF_DTYPES:
            self._run_module(
                lambda: ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=3),
                dtype,
            )

    def test_max_pooling(self):
        for dtype in HALF_DTYPES:
            self._run_module(
                lambda: ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
                dtype,
            )

    def test_global_pooling(self):
        for dtype in HALF_DTYPES:
            self._run_module(lambda: ME.MinkowskiGlobalPooling(), dtype)

    def test_broadcast_addition(self):
        for dtype in HALF_DTYPES:
            coords = _random_coordinates()
            x_half, x_ref, half_leaf, ref_leaf = _paired_inputs(
                coords, 8, dtype, self.device
            )
            pool = ME.MinkowskiGlobalPooling()
            bcast = ME.MinkowskiBroadcastAddition()

            y_half = bcast(x_half, pool(x_half))
            y_ref = bcast(x_ref, pool(x_ref))
            self.assertEqual(y_half.F.dtype, dtype)

            g = torch.randn(y_ref.F.shape, device=self.device).to(dtype)
            (y_half.F * g).sum().backward()
            (y_ref.F * g.float()).sum().backward()

            self.assert_close(y_half.F, y_ref.F, dtype, "broadcast output")
            self.assert_close(half_leaf.grad, ref_leaf.grad, dtype, "broadcast grad")

    def test_spmm(self):
        for dtype in HALF_DTYPES:
            torch.manual_seed(3)
            nnz, nrows, ncols, nch = 500, 64, 128, 16
            rows = torch.randint(0, nrows, (nnz,), device=self.device).int()
            cols = torch.randint(0, ncols, (nnz,), device=self.device).int()
            vals = torch.randn(nnz, device=self.device).to(dtype)
            mat = torch.randn(ncols, nch, device=self.device).to(dtype)

            mat_half = mat.clone().requires_grad_()
            mat_ref = mat.float().clone().requires_grad_()
            size = torch.Size([nrows, ncols])
            fn = ME.MinkowskiSPMMFunction()
            out_half = fn.apply(rows, cols, vals, size, mat_half)
            out_ref = fn.apply(rows, cols, vals.float(), size, mat_ref)
            self.assertEqual(out_half.dtype, dtype)

            g = torch.randn(out_ref.shape, device=self.device).to(dtype)
            (out_half * g).sum().backward()
            (out_ref * g.float()).sum().backward()

            self.assert_close(out_half, out_ref, dtype, "spmm output")
            self.assert_close(mat_half.grad, mat_ref.grad, dtype, "spmm grad")

    def test_interpolation_promotes_under_autocast(self):
        # Interpolation promotes 16-bit features to the fp32 coordinate-field
        # dtype under autocast (grid_sampler-style fp32 policy).
        coords = _random_coordinates()
        x_half, _, _, _ = _paired_inputs(coords, 8, torch.bfloat16, self.device)
        tfield = coords[:50, 1:].float().to(self.device) + 0.5
        tfield = torch.cat(
            [torch.zeros(50, 1, device=self.device), tfield], 1
        ).contiguous()
        interp = ME.MinkowskiInterpolation()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = interp(x_half, tfield)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out).all())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAutocast(unittest.TestCase):
    def test_conv_autocast(self):
        for dtype in HALF_DTYPES:
            coords = _random_coordinates()
            feats = torch.randn(coords.size(0), 8, device="cuda", requires_grad=True)
            x = SparseTensor(feats, coordinates=coords.cuda())
            conv = ME.MinkowskiConvolution(
                8, 16, kernel_size=3, stride=2, dimension=3
            ).cuda()
            with torch.autocast("cuda", dtype=dtype):
                y = conv(x)
            self.assertEqual(y.F.dtype, dtype)
            y.F.float().sum().backward()
            # the autograd engine converts reduced-precision grads back to the
            # fp32 leaf/parameter dtypes
            self.assertEqual(feats.grad.dtype, torch.float32)
            self.assertEqual(conv.kernel.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(feats.grad).all())
            self.assertTrue(torch.isfinite(conv.kernel.grad).all())

    def test_network_autocast(self):
        # conv -> pooling -> global pooling -> broadcast under one autocast
        coords = _random_coordinates()
        feats = torch.randn(coords.size(0), 8, device="cuda", requires_grad=True)
        x = SparseTensor(feats, coordinates=coords.cuda())
        conv = ME.MinkowskiConvolution(
            8, 16, kernel_size=3, stride=2, dimension=3
        ).cuda()
        pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        gpool = ME.MinkowskiGlobalPooling()
        bcast = ME.MinkowskiBroadcastAddition()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = conv(x)
            y = pool(y)
            y = bcast(y, gpool(y))
        self.assertEqual(y.F.dtype, torch.bfloat16)
        y.F.float().sum().backward()
        self.assertEqual(feats.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feats.grad).all())
        self.assertTrue(torch.isfinite(conv.kernel.grad).all())


class TestHalfPrecisionCPU(unittest.TestCase):
    """CPU paths (naive fp32-accumulating loops) — runs on CPU-only CI."""

    def test_conv_cpu(self):
        for dtype in HALF_DTYPES:
            coords = _random_coordinates(nrows=200, batch=1, max_coord=10)
            probe = SparseTensor(torch.zeros(coords.size(0), 4), coordinates=coords)
            n = probe.F.size(0)
            base = torch.randn(n, 4).to(dtype)
            common = dict(
                coordinate_map_key=probe.coordinate_map_key,
                coordinate_manager=probe.coordinate_manager,
            )
            x_half = SparseTensor(base.clone(), **common)
            x_ref = SparseTensor(base.float(), **common)
            conv_half, conv_ref = _paired_modules(
                lambda: ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3),
                dtype,
                "cpu",
            )
            y_half = conv_half(x_half)
            y_ref = conv_ref(x_ref)
            self.assertEqual(y_half.F.dtype, dtype)
            self.assertTrue(torch.isfinite(y_half.F.float()).all())
            self.assertTrue(
                torch.allclose(y_half.F.float(), y_ref.F, **TOLERANCES[dtype])
            )


if __name__ == "__main__":
    unittest.main()
