# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
"""Regression tests for interpolation-weight renormalization at sparse
boundaries (NVIDIA/MinkowskiEngine issues #477, #363, #383).

Query points whose 2^D lattice neighborhood is only partially occupied used
to be systematically under-weighted (down to zero) because the weights of
missing corners were dropped without renormalizing over the surviving ones.
"""
import itertools
import math
import unittest

import torch

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiInterpolation,
    MinkowskiInterpolationFunction,
)
from MinkowskiEngine.MinkowskiCommon import get_minkowski_function


def _cube_plus_isolated_voxel(dtype, device):
    """A fully occupied 2x2x2 cube at (0..1)^3 plus an isolated voxel at
    (5,5,5), all in batch 0. Features are 1..9."""
    coords = [[0, i, j, k] for i, j, k in itertools.product(range(2), repeat=3)]
    coords.append([0, 5, 5, 5])
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    feats = torch.arange(1, len(coords) + 1, dtype=dtype, device=device).unsqueeze(1)
    return coords, feats


def _reference_interpolation(coords, feats, tfield, renormalize=True):
    """Manual multilinear interpolation over occupied corners only,
    renormalized by the sum of surviving weights (in fp64)."""
    fmap = {tuple(c.tolist()): f for c, f in zip(coords.cpu(), feats.cpu().double())}
    D = tfield.size(1) - 1
    out = torch.zeros(tfield.size(0), feats.size(1), dtype=torch.float64)
    for i, q in enumerate(tfield.cpu().double()):
        b = int(round(q[0].item()))
        acc = torch.zeros(feats.size(1), dtype=torch.float64)
        wsum = 0.0
        lb = [math.floor(q[1 + d].item()) for d in range(D)]
        for corner in itertools.product(*[(lb[d], lb[d] + 1) for d in range(D)]):
            w = 1.0
            for d in range(D):
                w *= 1 - abs(q[1 + d].item() - corner[d])
            key = (b,) + corner
            if key in fmap:
                acc += w * fmap[key]
                wsum += w
        if renormalize and wsum > 0:
            acc /= wsum
        out[i] = acc
    return out


def _raw_backend_forward(x, tfield):
    """Un-renormalized backend output (the pre-fix behavior)."""
    fw_fn = get_minkowski_function("InterpolationForward", x.F)
    out_feat, in_map, out_map, weights = fw_fn(
        x.F, tfield, x.coordinate_map_key, x.coordinate_manager._manager
    )
    return out_feat


class InterpolationRenormTestCase(unittest.TestCase):
    def _run_forward_case(self, device, dtype=torch.float32):
        coords, feats = _cube_plus_isolated_voxel(dtype, device)
        x = SparseTensor(feats, coordinates=coords)
        interp = MinkowskiInterpolation()

        # (a) exactly on occupied voxels, (b) interior with a full
        # neighborhood, (c) partial neighborhoods (1/8 and 4/8 corners).
        tfield = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 5, 5, 5],
                [0, 0.25, 0.5, 0.75],
                [0, 5.25, 5.25, 5.25],
                [0, 0.5, 0.5, 1.75],
                # no occupied corner at all -> must stay exactly zero
                [0, 20.5, 20.5, 20.5],
            ],
            dtype=dtype,
            device=device,
        )
        out = interp(x, tfield)
        raw = _raw_backend_forward(x, tfield)

        # exact-on-voxel and full-neighborhood rows are bit-identical to the
        # un-renormalized backend result
        for row in range(4):
            self.assertTrue(
                torch.equal(out[row], raw[row]),
                f"row {row} must be bit-identical to the backend output",
            )
        # exact-on-voxel rows equal the stored features exactly
        self.assertTrue(torch.equal(out[0], feats[0]))
        self.assertTrue(torch.equal(out[1], feats[7]))
        self.assertTrue(torch.equal(out[2], feats[8]))

        # all rows match the renormalized manual reference
        ref = _reference_interpolation(coords, feats, tfield).to(out)
        self.assertTrue(
            torch.allclose(out, ref, rtol=1e-5, atol=1e-6),
            f"expected {ref.flatten().tolist()}, got {out.flatten().tolist()}",
        )

        # the bug: boundary rows used to be badly under-weighted
        # (0.421875 * 9 = 3.796875 and 0.25 * 5 = 1.25); they must now be the
        # renormalized average of the existing neighbors, NOT near-zero
        self.assertAlmostEqual(out[4].item(), 9.0, places=4)
        self.assertAlmostEqual(out[5].item(), 5.0, places=4)
        # empty neighborhood stays exactly zero
        self.assertTrue(torch.equal(out[6], torch.zeros_like(out[6])))

    def test_forward_cpu(self):
        self._run_forward_case("cpu")

    def test_forward_gpu(self):
        self._run_forward_case("cuda")

    def _run_return_weights_case(self, device):
        coords, feats = _cube_plus_isolated_voxel(torch.float32, device)
        x = SparseTensor(feats, coordinates=coords)
        interp = MinkowskiInterpolation(return_kernel_map=True, return_weights=True)
        tfield = torch.tensor(
            [[0, 5.25, 5.25, 5.25], [0, 0.25, 0.5, 0.75]],
            dtype=torch.float32,
            device=device,
        )
        out, (in_map, out_map), weights = interp(x, tfield)
        # returned weights are the ones actually applied: they sum to 1 for
        # every query that found at least one occupied corner
        wsum = torch.zeros(tfield.size(0), device=device)
        wsum.scatter_add_(0, out_map.long(), weights)
        self.assertTrue(torch.allclose(wsum, torch.ones_like(wsum), atol=1e-6))

    def test_return_weights_cpu(self):
        self._run_return_weights_case("cpu")

    def test_return_weights_gpu(self):
        self._run_return_weights_case("cuda")

    def _run_gradcheck_case(self, device):
        # fp64 finite-difference gradient check on a tiny case with a
        # partial neighborhood (exercises the renormalized backward)
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 5, 5, 5]],
            dtype=torch.int32,
            device=device,
        )
        feats = torch.randn(3, 2, dtype=torch.float64, device=device)
        feats.requires_grad_()
        x = SparseTensor(feats, coordinates=coords)
        tfield = torch.tensor(
            [
                [0, 0.5, 0, 0],  # full 1D-edge neighborhood along x
                [0, 0.25, 0.75, 0.5],  # partial: 2/8 corners occupied
                [0, 5.5, 5.25, 5.75],  # partial: 1/8 corners occupied
            ],
            dtype=torch.float64,
            device=device,
        )

        def interp_only_features(f):
            return MinkowskiInterpolationFunction.apply(
                f, tfield, x.coordinate_map_key, x.coordinate_manager
            )[0]

        self.assertTrue(
            torch.autograd.gradcheck(interp_only_features, (feats,), eps=1e-6, atol=1e-4)
        )

    def test_gradcheck_cpu(self):
        self._run_gradcheck_case("cpu")

    def test_gradcheck_gpu(self):
        self._run_gradcheck_case("cuda")

    def test_half_precision_autocast_gpu(self):
        # 16-bit features are promoted to the fp32 coordinate-field dtype
        # under autocast; boundary queries must match the fp32 renormalized
        # reference within half-precision tolerance
        for dtype, tol in ((torch.float16, 1e-3), (torch.bfloat16, 1e-2)):
            coords, feats = _cube_plus_isolated_voxel(torch.float32, "cuda")
            x16 = SparseTensor(feats.to(dtype), coordinates=coords)
            x32 = SparseTensor(feats, coordinates=coords)
            tfield = torch.tensor(
                [[0, 5.25, 5.25, 5.25], [0, 0.5, 0.5, 1.75], [0, 0.25, 0.5, 0.75]],
                dtype=torch.float32,
                device="cuda",
            )
            interp = MinkowskiInterpolation()
            with torch.autocast("cuda", dtype=dtype):
                out16 = interp(x16, tfield)
            out32 = interp(x32, tfield)
            self.assertEqual(out16.dtype, torch.float32)
            self.assertTrue(torch.isfinite(out16).all())
            self.assertTrue(
                torch.allclose(out16, out32, rtol=tol, atol=tol),
                f"{dtype}: expected {out32.flatten().tolist()}, "
                f"got {out16.flatten().tolist()}",
            )

    def _run_features_at_coordinates_case(self, device):
        # SparseTensor.features_at_coordinates goes through the same
        # function and must return renormalized boundary values
        coords, feats = _cube_plus_isolated_voxel(torch.float32, device)
        x = SparseTensor(feats, coordinates=coords)
        q = torch.tensor([[0, 5.25, 5.25, 5.25]], dtype=torch.float32, device=device)
        out = x.features_at_coordinates(q)
        self.assertAlmostEqual(out.item(), 9.0, places=4)

    def test_features_at_coordinates_cpu(self):
        self._run_features_at_coordinates_case("cpu")

    def test_features_at_coordinates_gpu(self):
        self._run_features_at_coordinates_case("cuda")


if __name__ == "__main__":
    unittest.main()
