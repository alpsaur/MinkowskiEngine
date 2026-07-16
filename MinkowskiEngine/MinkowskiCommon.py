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
from collections.abc import Sequence
import functools

import numpy as np
from typing import Union

import torch

from torch.nn import Module

import MinkowskiEngineBackend._C as MEB


StrideType = Union[int, Sequence, np.ndarray, torch.IntTensor]


def convert_to_int_list(
    arg: Union[int, Sequence, np.ndarray, torch.Tensor], dimension: int
):
    if isinstance(arg, list):
        assert len(arg) == dimension
        return arg

    if isinstance(arg, (Sequence, np.ndarray, torch.Tensor)):
        tmp = [i for i in arg]
        assert len(tmp) == dimension
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = [int(arg) for i in range(dimension)]
    else:
        raise ValueError("Input must be a scalar or a sequence")

    return tmp


def convert_to_int_tensor(
    arg: Union[int, Sequence, np.ndarray, torch.IntTensor], dimension: int
):
    if isinstance(arg, torch.IntTensor):
        assert arg.numel() == dimension
        return arg

    if isinstance(arg, (Sequence, np.ndarray)):
        tmp = torch.IntTensor([i for i in arg])
        assert tmp.numel() == dimension
    elif np.isscalar(arg):  # Assume that it is a scalar
        tmp = torch.IntTensor([int(arg) for i in range(dimension)])
    else:
        raise ValueError("Input must be a scalar or a sequence")

    return tmp


def prep_args(
    tensor_stride: Union[int, Sequence, np.ndarray, torch.IntTensor],
    stride: Union[int, Sequence, np.ndarray, torch.IntTensor],
    kernel_size: Union[int, Sequence, np.ndarray, torch.IntTensor],
    dilation: Union[int, Sequence, np.ndarray, torch.IntTensor],
    region_type: Union[int, MEB.RegionType],
    D=-1,
):
    assert torch.prod(
        kernel_size > 0
    ), f"kernel_size must be a positive integer, provided {kernel_size}"
    assert D > 0, f"dimension must be a positive integer, {D}"
    tensor_stride = convert_to_int_tensor(tensor_stride, D)
    stride = convert_to_int_tensor(stride, D)
    kernel_size = convert_to_int_tensor(kernel_size, D)
    dilation = convert_to_int_tensor(dilation, D)
    region_type = int(region_type)
    return (
        tensor_stride,
        stride,
        kernel_size,
        dilation,
        region_type,
    )


def get_postfix(tensor: torch.Tensor):
    postfix = "GPU" if tensor.is_cuda else "CPU"
    return postfix


class MinkowskiModuleBase(Module):
    pass


@functools.lru_cache(maxsize=None)
def _resolve_minkowski_function(fn_name, is_cuda):
    if hasattr(MEB, fn_name):
        return getattr(MEB, fn_name)
    if is_cuda:
        raise ValueError(
            f"Function {fn_name} not available. Please compile MinkowskiEngine with `torch.cuda.is_available()` is `True`."
        )
    raise ValueError(f"Function {fn_name} not available.")


def get_minkowski_function(name, variable):
    fn_name = name + get_postfix(variable)
    # getattr-by-string on MEB happens on every call; cache the resolved
    # function by the resolved name so repeat dispatch is a dict lookup.
    # lru_cache does not memoize exceptions, so the error path is unchanged.
    return _resolve_minkowski_function(fn_name, variable.is_cuda)


def maybe_autocast(*tensors):
    """Cast fp32 CUDA tensors to the active autocast dtype.

    The Minkowski backend ops are opaque pybind functions without autocast
    dispatch keys, so ``torch.autocast`` cannot cast their inputs. Calling this
    at the top of each autograd.Function forward makes the ops participate in
    AMP: under an active CUDA autocast context, fp32 CUDA feature/weight
    tensors are cast to ``torch.get_autocast_dtype("cuda")`` (the autograd
    engine converts the resulting reduced-precision gradients back to the
    input dtypes). Outside autocast, or for non-fp32/non-CUDA tensors, inputs
    are returned unchanged.
    """
    if not torch.is_autocast_enabled("cuda"):
        return tensors if len(tensors) > 1 else tensors[0]
    dtype = torch.get_autocast_dtype("cuda")
    cast = tuple(
        t.to(dtype)
        if isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32
        else t
        for t in tensors
    )
    return cast if len(cast) > 1 else cast[0]
