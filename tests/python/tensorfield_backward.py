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
# Regression test for TensorField -> sparse -> conv -> slice backprop.
# Upstream issue: https://github.com/NVIDIA/MinkowskiEngine/issues/395
#
# The reported crash was a merge_sort cudaErrorIllegalAddress in the
# inverse_mapping backward pass, fixed upstream by commit 02fc608 (present in
# this fork's history). This test proves the fix holds: it builds a TensorField
# from tens of thousands of random points, runs field -> sparse -> conv ->
# slice back to a field, computes a loss on the FIELD-level output, and
# backpropagates. It runs several iterations with varying N to catch the
# intermittent illegal-address failure the original bug exhibited.
import unittest

import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import (
    TensorField,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiLinear,
    MinkowskiReLU,
    MinkowskiToSparseTensor,
)


def _make_field(N, D, device, seed):
    torch.manual_seed(seed)
    n0 = N // 2
    n1 = N - n0
    c0 = torch.rand(n0, D) * 100.0
    c1 = torch.rand(n1, D) * 100.0
    coords = torch.cat(
        [
            torch.cat([torch.zeros(n0, 1), c0], dim=1),
            torch.cat([torch.ones(n1, 1), c1], dim=1),
        ],
        dim=0,
    ).float()
    feats = torch.rand(N, 3)
    feats.requires_grad_()
    return coords.to(device), feats


class TestTensorFieldBackward(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_field_sparse_conv_slice_backward(self):
        device = "cuda"
        D = 3

        # Vary N across iterations to catch the intermittent illegal-address
        # crash the original #395 bug exhibited.
        for it, N in enumerate([50000, 40000, 60000, 30000, 55000]):
            coords, feats = _make_field(N, D, device, seed=it)
            tfield = TensorField(coordinates=coords, features=feats.to(device))

            net = nn.Sequential(
                MinkowskiLinear(3, 16),
                MinkowskiReLU(),
                MinkowskiToSparseTensor(),
                MinkowskiConvolution(16, 32, kernel_size=3, stride=2, dimension=D),
                MinkowskiConvolutionTranspose(
                    32, 16, kernel_size=3, stride=2, dimension=D
                ),
            ).to(device)

            soutput = net(tfield)
            # Slice back to the original (field-level) resolution and take the
            # loss there, so gradients flow through the field inverse mapping.
            ofield = soutput.slice(tfield)
            self.assertEqual(len(ofield), len(tfield))

            loss = ofield.F.pow(2).sum()
            loss.backward()
            torch.cuda.synchronize()

            self.assertIsNotNone(
                feats.grad, f"iter {it} (N={N}): field feature grad is None"
            )
            self.assertEqual(tuple(feats.grad.shape), (N, 3))
            self.assertTrue(
                torch.isfinite(feats.grad).all(),
                f"iter {it} (N={N}): non-finite gradient",
            )


if __name__ == "__main__":
    unittest.main()
