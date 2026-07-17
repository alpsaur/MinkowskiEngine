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
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
from typing import Union

import torch
from torch.autograd import Function

from MinkowskiEngineBackend._C import CoordinateMapKey
from MinkowskiSparseTensor import SparseTensor
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)


def _renormalize_partial_neighborhoods(
    out_feat: torch.Tensor,
    out_map: torch.Tensor,
    weights: torch.Tensor,
    dimension: int,
):
    r"""Renormalize interpolation results over the lattice corners that exist.

    The backend computes, for every query point, multilinear weights over the
    :math:`2^D` surrounding lattice corners and silently drops corners that
    are not occupied in the sparse tensor. Without renormalization the
    remaining weights no longer sum to one, so any query with a partially
    missing neighborhood is systematically under-weighted -- down to exactly
    zero when no corner exists (NVIDIA/MinkowskiEngine issues #477, #363,
    #383). This divides both the output features and the weights of such
    rows by the sum of the surviving weights.

    Rows with a complete :math:`2^D` neighborhood (whose weights sum to one
    by construction) and rows with no surviving corner (which stay zero) are
    returned bit-identical to the backend output.
    """
    if out_feat.shape[0] == 0 or weights.numel() == 0:
        return out_feat, weights

    num_out = out_feat.shape[0]
    full_neighborhood = 2 ** dimension
    out_map_long = out_map.long()
    # accumulate in fp32 (or fp64) even for 16-bit weights
    w_acc = (
        weights.float()
        if weights.dtype in (torch.float16, torch.bfloat16)
        else weights
    )
    weight_sum = torch.zeros(num_out, dtype=w_acc.dtype, device=w_acc.device)
    weight_sum.scatter_add_(0, out_map_long, w_acc)
    num_found = torch.zeros(num_out, dtype=torch.long, device=w_acc.device)
    num_found.scatter_add_(0, out_map_long, torch.ones_like(out_map_long))

    partial = (num_found < full_neighborhood) & (weight_sum > 0)
    if not bool(partial.any()):
        return out_feat, weights

    scale = torch.where(
        partial, weight_sum.reciprocal(), torch.ones_like(weight_sum)
    )
    # multiplication by exactly 1.0 leaves full-neighborhood /
    # exact-on-voxel rows bit-identical
    out_feat = out_feat * scale.to(out_feat.dtype).unsqueeze(1)
    # keep the weights consistent with the produced features; the backward
    # pass and the `return_weights=True` output both use the renormalized
    # weights
    weights = weights * scale.to(weights.dtype)[out_map_long]
    return out_feat, weights


class MinkowskiInterpolationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        tfield: torch.Tensor,
        in_coordinate_map_key: CoordinateMapKey,
        coordinate_manager: CoordinateManager = None,
    ):
        # AMP: the backend requires features and the coordinate field to share
        # a dtype, and interpolation weights are computed from the raw
        # coordinates — promote 16-bit features to the fp32 coordinate field
        # (mirrors PyTorch's fp32 autocast policy for grid_sampler-like ops)
        # rather than degrading coordinates to 16 bits.
        if (
            torch.is_autocast_enabled("cuda")
            and input_features.is_cuda
            and input_features.dtype in (torch.float16, torch.bfloat16)
            and tfield.dtype == torch.float32
        ):
            input_features = input_features.to(torch.float32)
        input_features = input_features.contiguous()
        # in_map, out_map, weights = coordinate_manager.interpolation_map_weight(
        #     in_coordinate_map_key, tfield)
        fw_fn = get_minkowski_function("InterpolationForward", input_features)
        with torch.profiler.record_function("ME::Interpolation.forward"):
            out_feat, in_map, out_map, weights = fw_fn(
                input_features,
                tfield,
                in_coordinate_map_key,
                coordinate_manager._manager,
            )
            out_feat, weights = _renormalize_partial_neighborhoods(
                out_feat, out_map, weights, tfield.size(1) - 1
            )
        ctx.save_for_backward(in_map, out_map, weights)
        ctx.inputs = (
            in_coordinate_map_key,
            coordinate_manager,
        )
        return out_feat, in_map, out_map, weights

    @staticmethod
    def backward(
        ctx, grad_out_feat=None, grad_in_map=None, grad_out_map=None, grad_weights=None
    ):
        grad_out_feat = grad_out_feat.contiguous()
        bw_fn = get_minkowski_function("InterpolationBackward", grad_out_feat)
        (
            in_coordinate_map_key,
            coordinate_manager,
        ) = ctx.inputs
        in_map, out_map, weights = ctx.saved_tensors

        with torch.profiler.record_function("ME::Interpolation.backward"):
            grad_in_feat = bw_fn(
                grad_out_feat,
                in_map,
                out_map,
                weights,
                in_coordinate_map_key,
                coordinate_manager._manager,
            )
        return grad_in_feat, None, None, None


class MinkowskiInterpolation(MinkowskiModuleBase):
    r"""Sample linearly interpolated features at the provided points.

    Interpolation weights are renormalized over the lattice corners that are
    actually occupied, so queries at the boundary of the sparse support
    return a weighted average of the existing neighbors instead of decaying
    to zero. Queries whose :math:`2^D` neighborhood contains no occupied
    corner return zeros.
    """

    def __init__(self, return_kernel_map=False, return_weights=False):
        r"""Sample linearly interpolated features at the specified coordinates.

        Args:
            :attr:`return_kernel_map` (bool): In addition to the sampled
            features, the layer returns the kernel map as a pair of input row
            indices and output row indices. False by default.

            :attr:`return_weights` (bool): When True, return the linear
            interpolation weights. False by default.
        """
        MinkowskiModuleBase.__init__(self)
        self.return_kernel_map = return_kernel_map
        self.return_weights = return_weights
        self.interp = MinkowskiInterpolationFunction()

    def forward(
        self,
        input: SparseTensor,
        tfield: torch.Tensor,
    ):
        # Get a new coordinate map key or extract one from the coordinates
        out_feat, in_map, out_map, weights = self.interp.apply(
            input.F,
            tfield,
            input.coordinate_map_key,
            input._manager,
        )

        return_args = [out_feat]
        if self.return_kernel_map:
            return_args.append((in_map, out_map))
        if self.return_weights:
            return_args.append(weights)
        if len(return_args) > 1:
            return tuple(return_args)
        else:
            return out_feat

    def __repr__(self):
        return self.__class__.__name__ + "()"
