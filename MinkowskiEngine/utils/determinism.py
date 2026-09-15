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
r"""Opt-in deterministic convolution helpers.

MinkowskiEngine convolutions accumulate contributions with ``atomicAdd`` and
dispatch the kernel map in an order that depends on how the input coordinates
were hashed / inserted.  As a consequence, running the *same* convolution twice
on the *same* set of points but in a different row order can produce outputs
that differ in the low-order bits.  The effect is largest for strided
(``stride > 1``) and transposed convolutions.  See upstream issues
https://github.com/NVIDIA/MinkowskiEngine/issues/554 and
https://github.com/NVIDIA/MinkowskiEngine/issues/504.

Canonical coordinate ordering is combined with the non-fused copy-GEMM
backend: sorting alone cannot order the atomic adds in fused scatter kernels.
Sorted maps are cached within the ORIGINAL coordinate manager, preserving
residual connections, explicit output coordinates, and TensorField mappings.
This module exposes that as:

* :func:`sorted_coordinates` -- return a new :class:`SparseTensor` whose rows
  are in a canonical lexicographic coordinate order, and
* :func:`set_deterministic` / :func:`is_deterministic` -- a process-wide flag
  that, when enabled, makes :class:`MinkowskiConvolution` and
  :class:`MinkowskiConvolutionTranspose` route their input through
  :func:`sorted_coordinates` before calling the backend.

This is *opt-in*: the default behavior is unchanged.
"""
import torch

from MinkowskiSparseTensor import SparseTensor

__all__ = [
    "set_deterministic",
    "is_deterministic",
    "sorted_coordinates",
]

# Process-wide opt-in flag. Off by default; toggled via set_deterministic().
_DETERMINISTIC = False


def set_deterministic(mode: bool = True):
    r"""Enable or disable opt-in deterministic convolution behavior.

    When enabled, :class:`MinkowskiEngine.MinkowskiConvolution` and
    :class:`MinkowskiEngine.MinkowskiConvolutionTranspose` sort their input into
    a canonical coordinate order and select non-fused copy-GEMM, overriding
    the convolution algorithm and ``ME_FUSED_COPY`` performance preferences.
    The contract covers convolution forward values on the same device/build,
    not bitwise deterministic training, other operations, or different GPUs.
    Backward remains supported, but weight-gradient reductions and external
    PyTorch/cuBLAS settings can introduce nondeterminism.

    .. warning::

        This costs a sort and an additional map per distinct input map (cached
        for the coordinate manager's lifetime), feature reindexing per call,
        and slower per-offset gather/scatter. It is intended for debugging,
        not throughput-critical training. The default is ``False``.

    Args:
        :attr:`mode` (bool): ``True`` to enable deterministic convolutions,
        ``False`` to restore the default behavior. Defaults to ``True``.
    """
    global _DETERMINISTIC
    _DETERMINISTIC = bool(mode)


def is_deterministic() -> bool:
    r"""Return whether opt-in deterministic convolution behavior is enabled."""
    return _DETERMINISTIC


def _lexicographic_permutation(coordinates: torch.Tensor) -> torch.Tensor:
    r"""Return a permutation that lexicographically sorts ``coordinates`` rows.

    Column 0 (the batch index) is the most significant key and the last spatial
    column is the least significant. The sort is implemented as a stable radix
    sort -- a sequence of per-column stable ``argsort`` passes from the least to
    the most significant column. This avoids packing the coordinates into a
    single integer rank (which would overflow for large or high-dimensional
    coordinates) and is correct for arbitrary spatial dimension ``D`` and for
    negative coordinates (signed integer comparison).
    """
    n, ncol = coordinates.shape
    perm = torch.arange(n, device=coordinates.device)
    # Least significant column first, most significant (batch) column last, so
    # that the final stable pass leaves the batch column dominant.
    for col in reversed(range(ncol)):
        keys = coordinates[perm, col]
        order = torch.argsort(keys, stable=True)
        perm = perm[order]
    return perm


def sorted_coordinates(sparse_tensor: SparseTensor) -> SparseTensor:
    r"""Return a copy of ``sparse_tensor`` with rows in canonical coordinate order.

    The returned :class:`SparseTensor` contains the same points and features as
    the input, but its rows are sorted by a lexicographic ranking of the
    coordinates (batch index most significant). The row (insertion) order
    becomes a deterministic function of the point set, independent of the
    input row permutation.

    The reconstruction preserves the input ``tensor_stride``,
    ``quantization_mode``, and coordinate manager. Only the map and permutation
    are cached, never features or an autograd graph. Gradients flow through
    reordered features. Sorting alone does NOT make fused convolution
    deterministic; use :func:`set_deterministic` for that.

    Args:
        :attr:`sparse_tensor` (:class:`MinkowskiEngine.SparseTensor`): the tensor
        to reorder.

    Returns:
        A new :class:`MinkowskiEngine.SparseTensor` with canonically ordered rows.
    """
    assert isinstance(
        sparse_tensor, SparseTensor
    ), "sorted_coordinates expects a SparseTensor"

    manager = sparse_tensor.coordinate_manager
    stride, string_id = sparse_tensor.coordinate_map_key.get_key()
    cache_key = (tuple(stride), string_id)
    cache = getattr(manager, "_deterministic_maps", None)
    if cache is None:
        cache = manager._deterministic_maps = {}
    if cache_key not in cache:
        coords = sparse_tensor.C
        if len(coords) <= 1:
            return sparse_tensor
        with torch.no_grad():
            perm = _lexicographic_permutation(coords)
            unique_key = manager.get_unique_coordinate_map_key(stride)
            key, (unique, _) = manager.insert_and_map(
                coords[perm], stride, unique_key[1]
            )
            # Honor the backend's insertion mapping; never assume row order.
            perm = perm[unique.long()]
        cache[cache_key] = (key, perm)
        # Sorting an already canonical map need not allocate another map.
        cache[(tuple(stride), key.get_key()[1])] = (key, None)
    key, perm = cache[cache_key]
    if perm is None:
        return sparse_tensor
    return SparseTensor(
        sparse_tensor.F[perm],
        coordinate_map_key=key,
        coordinate_manager=manager,
        quantization_mode=sparse_tensor.quantization_mode,
    )
