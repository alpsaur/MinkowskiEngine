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
# Regression test for pruning with an all-False mask on CUDA.
# Upstream issue: https://github.com/NVIDIA/MinkowskiEngine/issues/579
#
# On this fork the CUDA pruning kernel already emits a valid empty SparseTensor
# for an empty mask (src/pruning_gpu.cu handles the size-0 case), so this test
# documents/locks in that behavior: an all-False mask followed by a downstream
# convolution and backward must not crash, must produce zero-size outputs, and
# must yield finite (empty) gradients.
import unittest

import torch

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiPruning,
)


class TestPruningEmptyMask(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_all_false_mask_downstream_conv_backward(self):
        in_channels, out_channels, D = 4, 8, 3
        coords = torch.IntTensor(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 1, 1],
            ]
        )
        feats = torch.rand(len(coords), in_channels)
        feats.requires_grad_()
        inp = SparseTensor(feats, coords, device="cuda")

        mask = torch.zeros(len(inp), dtype=torch.bool, device="cuda")  # all False
        pruning = MinkowskiPruning()
        pruned = pruning(inp, mask)

        # Empty tensor: zero rows, feature width preserved.
        self.assertEqual(len(pruned), 0)
        self.assertEqual(pruned.F.shape[0], 0)
        self.assertEqual(pruned.F.shape[1], in_channels)
        # Coordinate access on the empty tensor must not crash.
        self.assertEqual(pruned.C.shape[0], 0)
        torch.cuda.synchronize()

        # Downstream convolution on the empty tensor.
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=1, dimension=D
        ).cuda()
        out = conv(pruned)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.F.shape[0], 0)
        self.assertEqual(out.F.shape[1], out_channels)
        torch.cuda.synchronize()

        # Backward through the empty pipeline: grad must exist, be shape-correct
        # and finite (vacuously, since it is empty).
        out.F.sum().backward()
        self.assertIsNotNone(feats.grad)
        self.assertEqual(tuple(feats.grad.shape), (len(coords), in_channels))
        self.assertTrue(torch.isfinite(feats.grad).all())
        # No feature survived the all-False prune, so nothing propagated back.
        self.assertEqual(float(feats.grad.abs().sum()), 0.0)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
