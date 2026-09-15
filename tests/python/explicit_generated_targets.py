"""Explicit targets override coordinate expansion, including generative layers."""

import unittest

import torch
import MinkowskiEngine as ME


class TestExplicitGeneratedTargets(unittest.TestCase):
    def tearDown(self):
        ME.set_deterministic(False)

    def _check_targets(self, device):
        for kind in ("convolution", "transpose", "generative"):
            transpose = kind != "convolution"
            for deterministic in (False, True):
                ME.set_deterministic(deterministic)
                for target_kind in ("tensor", "key", "sparse"):
                    with self.subTest(
                        kind=kind, deterministic=deterministic, target=target_kind
                    ):
                        in_stride, out_stride = (2, 1) if transpose else (1, 2)
                        coords = torch.tensor(
                            [[0, 0], [0, in_stride], [0, 2 * in_stride]],
                            dtype=torch.int32,
                            device=device,
                        )
                        features = torch.tensor(
                            [[1.0], [2.0], [3.0]],
                            dtype=torch.float64,
                            device=device,
                            requires_grad=True,
                        )
                        x = ME.SparseTensor(features, coords, tensor_stride=in_stride)
                        target_coords = torch.tensor(
                            [[0, 1], [0, 9]] if transpose else [[0, 0], [0, 8]],
                            dtype=torch.int32,
                            device=device,
                        )
                        target = ME.SparseTensor(
                            torch.zeros(2, 1, device=device),
                            target_coords,
                            tensor_stride=out_stride,
                            coordinate_manager=x.coordinate_manager,
                        )
                        cls = {
                            "convolution": ME.MinkowskiConvolution,
                            "transpose": ME.MinkowskiConvolutionTranspose,
                            "generative": ME.MinkowskiGenerativeConvolutionTranspose,
                        }[kind]
                        options = (
                            {} if kind == "generative" else {"expand_coordinates": True}
                        )
                        conv = (
                            cls(1, 1, 3, stride=2, dimension=1, **options)
                            .to(device)
                            .double()
                        )
                        with torch.no_grad():
                            conv.kernel.copy_(
                                torch.tensor([0.25, 0.5, 0.75], device=device).reshape(
                                    3, 1, 1
                                )
                            )
                        requested = {
                            "tensor": target.C,
                            "key": target.coordinate_map_key,
                            "sparse": target,
                        }[target_kind]
                        output = conv(x, coordinates=requested)
                        self.assertIs(output.coordinate_manager, x.coordinate_manager)
                        self.assertEqual(output.tensor_stride, [out_stride])
                        self.assertEqual(
                            {tuple(c) for c in output.C.tolist()},
                            {tuple(c) for c in target_coords.tolist()},
                        )
                        self.assertEqual(len(output), 2)
                        if target_kind != "tensor":
                            self.assertEqual(
                                output.coordinate_map_key, target.coordinate_map_key
                            )
                        # Independent scalar convolution oracle, in returned row order.
                        reference = []
                        for query in output.C.tolist():
                            value = features.sum() * 0
                            for i, source in enumerate(coords.tolist()):
                                for k, offset in enumerate((-1, 0, 1)):
                                    matches = (
                                        query[1] == source[1] + offset * out_stride
                                        if transpose
                                        else source[1] == query[1] + offset * in_stride
                                    )
                                    if matches:
                                        value = (
                                            value
                                            + features[i, 0] * conv.kernel[k, 0, 0]
                                        )
                            reference.append(value)
                        expected = torch.stack(reference).reshape(-1, 1)
                        torch.testing.assert_close(
                            output.F, expected, atol=1e-12, rtol=1e-12
                        )
                        actual_grad = torch.autograd.grad(
                            output.F.sum(), (features, conv.kernel)
                        )
                        expected_grad = torch.autograd.grad(
                            expected.sum(), (features, conv.kernel)
                        )
                        for actual, expected in zip(actual_grad, expected_grad):
                            torch.testing.assert_close(
                                actual, expected, atol=1e-12, rtol=1e-12
                            )

    def test_cpu(self):
        self._check_targets("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda(self):
        self._check_targets("cuda")
