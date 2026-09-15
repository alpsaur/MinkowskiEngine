# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#
# Regression tests for the opt-in deterministic convolution path
# (ME.set_deterministic / ME.utils.sorted_coordinates). See upstream issues
# https://github.com/NVIDIA/MinkowskiEngine/issues/554 and
# https://github.com/NVIDIA/MinkowskiEngine/issues/504.
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

import MinkowskiEngine as ME
from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
)


def _canonical_feats(sparse_tensor):
    r"""Return the features of ``sparse_tensor`` reordered into a canonical
    lexicographic coordinate order so that outputs built from differently
    permuted inputs can be compared row-for-row."""
    C = sparse_tensor.C.cpu().numpy()
    F = sparse_tensor.F.detach().cpu()
    # np.lexsort uses the last key as primary; we want column 0 (batch) primary.
    keys = tuple(C[:, i] for i in reversed(range(C.shape[1])))
    order = np.lexsort(keys)
    return F[order].contiguous(), C[order]


def _make_point_set(N, D, in_channels, seed):
    r"""Build a unique batched coordinate/feature set (2 batches)."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randint(0, 24, (N, D), generator=g)
    batch = torch.randint(0, 2, (N, 1), generator=g)
    full = torch.cat([batch, coords], dim=1)
    full = torch.unique(full, dim=0).int()
    feats = torch.randn(full.shape[0], in_channels, generator=g)
    return full, feats


class TestDeterminism(unittest.TestCase):
    def tearDown(self):
        # Never leak the global flag into other tests.
        ME.set_deterministic(False)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sorted_coordinates_helper(self):
        # Handles arbitrary D and negative coordinates, keeps feats aligned.
        coords = torch.IntTensor(
            [[1, -2, 3, 0], [0, 5, -1, 2], [0, -3, 4, 4], [1, -2, 3, -1]]
        )
        feats = torch.arange(4 * 2).view(4, 2).float()
        st = SparseTensor(feats, coords, device="cuda")
        srt = ME.sorted_coordinates(st)
        self.assertEqual(len(srt), len(st))
        # Rows are lexicographically non-decreasing (batch col most significant).
        C = srt.C.cpu().numpy()
        keys = tuple(C[:, i] for i in reversed(range(C.shape[1])))
        order = np.lexsort(keys)
        self.assertTrue(np.array_equal(order, np.arange(len(C))))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_deterministic_conv_stride2(self):
        in_channels, out_channels, D = 3, 5, 3
        coords, feats = _make_point_set(4000, D, in_channels, seed=0)

        torch.manual_seed(1234)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D
        ).cuda()

        ME.set_deterministic(True)
        reference = None
        for trial in range(5):
            g = torch.Generator().manual_seed(100 + trial)
            perm = torch.randperm(coords.shape[0], generator=g)
            inp = SparseTensor(feats[perm].clone(), coords[perm].clone(), device="cuda")
            out = conv(inp)
            F_canon, C_canon = _canonical_feats(out)
            if reference is None:
                reference = (F_canon, C_canon)
            else:
                self.assertTrue(
                    np.array_equal(C_canon, reference[1]),
                    "coordinate sets differ across permutations",
                )
                self.assertTrue(
                    torch.equal(F_canon, reference[0]),
                    f"deterministic stride-2 conv output differs on trial {trial}",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_deterministic_conv_transpose(self):
        in_channels, out_channels, D = 3, 5, 3
        coords, feats = _make_point_set(3000, D, in_channels, seed=7)
        # Strided (tensor_stride=2) input so the transpose upsamples to stride 1.
        coords[:, 1:] = coords[:, 1:] * 2

        torch.manual_seed(4321)
        conv_tr = MinkowskiConvolutionTranspose(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D
        ).cuda()

        ME.set_deterministic(True)
        reference = None
        for trial in range(5):
            g = torch.Generator().manual_seed(200 + trial)
            perm = torch.randperm(coords.shape[0], generator=g)
            inp = SparseTensor(
                feats[perm].clone(),
                coords[perm].clone(),
                tensor_stride=2,
                device="cuda",
            )
            out = conv_tr(inp)
            F_canon, C_canon = _canonical_feats(out)
            if reference is None:
                reference = (F_canon, C_canon)
            else:
                self.assertTrue(
                    np.array_equal(C_canon, reference[1]),
                    "coordinate sets differ across permutations",
                )
                self.assertTrue(
                    torch.equal(F_canon, reference[0]),
                    f"deterministic transpose conv output differs on trial {trial}",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_flag_off_path_runs(self):
        # With the flag off we do NOT assert cross-permutation equality (that
        # would be flaky by design); we only assert the default path still runs
        # and backprops.
        in_channels, out_channels, D = 3, 4, 3
        coords, feats = _make_point_set(2000, D, in_channels, seed=3)

        torch.manual_seed(11)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D
        ).cuda()

        ME.set_deterministic(False)
        self.assertFalse(ME.is_deterministic())
        inp = SparseTensor(feats.clone(), coords.clone(), device="cuda")
        inp.F.requires_grad_()
        out = conv(inp)
        out.F.sum().backward()
        self.assertIsNotNone(inp.F.grad)
        self.assertTrue(torch.isfinite(inp.F.grad).all())


class TestDeterministicIntegration(unittest.TestCase):
    def tearDown(self):
        ME.set_deterministic(False)

    def devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_residual_cat_cache_and_backward(self):
        for device in self.devices():
            coords, feats = _make_point_set(80, 3, 4, seed=17)
            x = SparseTensor(feats, coords, device=device)
            x.F.requires_grad_()
            conv = MinkowskiConvolution(4, 4, 3, dimension=3).to(device)
            reference = conv(x)
            ME.set_deterministic(True)
            out = conv(x)
            self.assertIs(out.coordinate_manager, x.coordinate_manager)
            self.assertEqual(out.coordinate_map_key, x.coordinate_map_key)
            torch.testing.assert_close(out.F, reference.F)
            self.assertEqual(ME.cat(out, x).F.shape[1], 8)
            (out + x).F.sum().backward()
            self.assertTrue(torch.isfinite(x.F.grad).all())
            inplace = conv(x)
            inplace += x  # the normal ResNet block uses +=, not a union
            self.assertEqual(inplace.coordinate_map_key, x.coordinate_map_key)
            cache_size = len(x.coordinate_manager._deterministic_maps)
            other = SparseTensor(
                (x.F.detach() + 1).requires_grad_(),
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            s1, s2 = ME.sorted_coordinates(x), ME.sorted_coordinates(other)
            self.assertEqual(s1.coordinate_map_key, s2.coordinate_map_key)
            torch.testing.assert_close(s2.F, s1.F + 1)
            conv(other).F.sum().backward()
            self.assertIsNotNone(other.F.grad)
            self.assertEqual(len(x.coordinate_manager._deterministic_maps), cache_size)
            ME.set_deterministic(False)

    def test_unet_targets_and_field_slice(self):
        for device in self.devices():
            coords, feats = _make_point_set(80, 3, 4, seed=18)
            field = ME.TensorField(feats.to(device).requires_grad_(), coords.float().to(device))
            skip = field.sparse()
            down = MinkowskiConvolution(4, 4, 3, stride=2, dimension=3).to(device)
            up = MinkowskiConvolutionTranspose(4, 4, 3, stride=2, dimension=3).to(device)
            ME.set_deterministic(True)
            encoded = down(skip)
            for target in (None, skip, skip.coordinate_map_key, skip.C):
                decoded = up(encoded, coordinates=target)
                self.assertIs(decoded.coordinate_manager, skip.coordinate_manager)
                a, ca = _canonical_feats(decoded)
                b, cb = _canonical_feats(up(encoded, coordinates=skip))
                self.assertTrue(np.array_equal(ca, cb))
                torch.testing.assert_close(a, b)
                if not isinstance(target, torch.Tensor):
                    self.assertEqual(decoded.coordinate_map_key, skip.coordinate_map_key)
                    self.assertEqual(len(decoded.slice(field)), len(field))
                    ME.cat(decoded, skip).F.sum().backward(retain_graph=True)
            self.assertIsNotNone(down.kernel.grad)
            self.assertIsNotNone(up.kernel.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_setting_and_permutations(self):
        coords = torch.cartesian_prod(*[torch.arange(6)] * 3).int()
        coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords], 1)
        torch.manual_seed(29)
        feats = torch.randn(len(coords), 32)
        with patch.dict(os.environ, {"ME_FUSED_COPY": "1"}):
            for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                for transpose in (False, True):
                    cls = MinkowskiConvolutionTranspose if transpose else MinkowskiConvolution
                    conv = cls(32, 32, 3, stride=2, dimension=3).cuda().to(dtype)
                    c = coords.clone()
                    if transpose:
                        c[:, 1:] *= 2
                    ME.set_deterministic(True)
                    reference = None
                    for trial in range(5):
                        p = torch.randperm(len(c))
                        x = SparseTensor(feats[p].to(dtype), c[p],
                                         tensor_stride=2 if transpose else 1, device="cuda")
                        with torch.no_grad():
                            out = conv(x)
                        current = _canonical_feats(out)
                        if reference is None:
                            reference = current
                        else:
                            self.assertTrue(np.array_equal(current[1], reference[1]))
                            self.assertTrue(torch.equal(current[0], reference[0]),
                                            f"{dtype}, transpose={transpose}, trial={trial}")
                    self.assertEqual(os.environ["ME_FUSED_COPY"], "1")

    def test_expansion_and_explicit_mode(self):
        for device in self.devices():
            coords, feats = _make_point_set(20, 3, 4, seed=33)
            for transpose in (False, True):
                c = coords.clone()
                if transpose:
                    c[:, 1:] *= 2
                x = SparseTensor(feats, c, tensor_stride=2 if transpose else 1, device=device)
                cls = ME.MinkowskiGenerativeConvolutionTranspose if transpose else MinkowskiConvolution
                kwargs = {} if transpose else {"expand_coordinates": True}
                conv = cls(4, 4, 3, stride=2, dimension=3, **kwargs).to(device)
                ME.set_deterministic(False)
                reference = _canonical_feats(conv(x))
                conv.convolution_mode = ME.ConvolutionMode.DETERMINISTIC
                current = _canonical_feats(conv(x))
                self.assertTrue(np.array_equal(current[1], reference[1]))
                torch.testing.assert_close(current[0], reference[0])


if __name__ == "__main__":
    unittest.main()
