"""Regression for publishing pointers to not-yet-visible GPU coordinates.

The old single-kernel region generator could insert the same coordinate twice:
one hash entry won later lookups, leaving an extra, zero-feature output row.
"""

import itertools
import unittest

import numpy as np
import torch
import MinkowskiEngine as ME
from tests.python.determinism import _make_point_set


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGeneratedCoordinates(unittest.TestCase):
    def tearDown(self):
        ME.set_deterministic(False)

    def assert_coordinates(self, output, expected):
        coordinates = output.C.cpu().numpy()
        unique = np.unique(coordinates, axis=0)
        self.assertEqual(len(coordinates), len(unique), "phantom duplicate output rows")
        self.assertTrue(
            np.array_equal(unique, expected), "wrong generated coordinate set"
        )

    def make_case(self, transpose):
        coords, feats = _make_point_set(3000, 3, 3, seed=7)
        if transpose:
            coords[:, 1:] *= 2
        offsets = np.array([[0, *p] for p in itertools.product((-1, 0, 1), repeat=3)])
        candidates = (coords.numpy()[:, None, :] + offsets[None, :, :]).reshape(-1, 4)
        if not transpose:
            candidates = candidates[np.all(candidates[:, 1:] % 2 == 0, axis=1)]
        return coords, feats, np.unique(candidates, axis=0)

    def test_transpose_has_no_phantom_rows(self):
        # Fails frequently on the old kernel even with determinism DISABLED.
        coords, feats, expected = self.make_case(transpose=True)
        conv = ME.MinkowskiConvolutionTranspose(3, 5, 3, stride=2, dimension=3).cuda()
        ME.set_deterministic(False)
        for trial in range(40):
            perm = torch.randperm(
                len(coords), generator=torch.Generator().manual_seed(200 + trial % 6)
            )
            x = ME.SparseTensor(
                feats[perm], coords[perm], tensor_stride=2, device="cuda"
            )
            with torch.no_grad():
                self.assert_coordinates(conv(x), expected)

    def test_other_region_map_users(self):
        for transpose in (False, True):
            coords, feats, expected = self.make_case(transpose)
            op = (
                ME.MinkowskiPoolingTranspose(3, stride=2, dimension=3)
                if transpose
                else ME.MinkowskiConvolution(
                    3, 5, 3, stride=2, expand_coordinates=True, dimension=3
                )
            ).cuda()
            for trial in range(4):
                x = ME.SparseTensor(
                    feats, coords, tensor_stride=2 if transpose else 1, device="cuda"
                )
                with torch.no_grad():
                    self.assert_coordinates(op(x), expected)

    def test_empty_generated_map(self):
        x = ME.SparseTensor(
            torch.empty(0, 3, device="cuda", requires_grad=True),
            torch.empty(0, 4, dtype=torch.int32, device="cuda"),
            tensor_stride=2,
        )
        conv = ME.MinkowskiGenerativeConvolutionTranspose(
            3, 5, 3, stride=2, dimension=3
        ).cuda()
        out = conv(x)
        self.assertEqual(tuple(out.C.shape), (0, 4))
        self.assertEqual(tuple(out.F.shape), (0, 5))
        out.F.sum().backward()
        self.assertIsNotNone(x.F.grad)
        self.assertEqual(int(torch.count_nonzero(conv.kernel.grad)), 0)


class TestEmptyKernelMapCPU(unittest.TestCase):
    def test_empty_transpose_backward(self):
        x = ME.SparseTensor(
            torch.empty(0, 3, requires_grad=True),
            torch.empty(0, 4, dtype=torch.int32),
            tensor_stride=2,
        )
        conv = ME.MinkowskiGenerativeConvolutionTranspose(
            3, 5, 3, stride=2, dimension=3
        )
        out = conv(x)
        self.assertEqual(tuple(out.F.shape), (0, 5))
        out.F.sum().backward()
        self.assertIsNotNone(x.F.grad)
        self.assertEqual(int(torch.count_nonzero(conv.kernel.grad)), 0)


if __name__ == "__main__":
    unittest.main()
