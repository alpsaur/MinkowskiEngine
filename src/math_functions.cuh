/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#ifndef MATH_FUNCTIONS_CUH
#define MATH_FUNCTIONS_CUH

#include "mkl_alternate.hpp"

#include "gpu.cuh"

#include <ATen/cuda/Atomic.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace minkowski {

namespace detail {

// Reduced-precision (16-bit) feature types. These accumulate in fp32 and are
// routed to cublasGemmEx / cusparseSpMM with a CUDA_R_32F compute type.
template <typename Dtype> struct is_reduced_fp : std::false_type {};
template <> struct is_reduced_fp<c10::Half> : std::true_type {};
template <> struct is_reduced_fp<c10::BFloat16> : std::true_type {};

// Scalar type used for accumulation / library compute: fp32 for 16-bit
// feature types, the feature type itself otherwise (keeps fp32/fp64
// behavior unchanged).
template <typename Dtype>
using accum_type_t =
    typename std::conditional<is_reduced_fp<Dtype>::value, float, Dtype>::type;

template <typename Dtype> constexpr cudaDataType cuda_data_type_of() {
  return std::is_same<Dtype, double>::value
             ? CUDA_R_64F
             : (std::is_same<Dtype, c10::Half>::value
                    ? CUDA_R_16F
                    : (std::is_same<Dtype, c10::BFloat16>::value ? CUDA_R_16BF
                                                                 : CUDA_R_32F));
}

// cusparseSpMM compute type: 16-bit data computes in fp32.
template <typename Dtype> constexpr cudaDataType cusparse_compute_type_of() {
  return std::is_same<Dtype, double>::value ? CUDA_R_64F : CUDA_R_32F;
}

} // end namespace detail

template <typename Dtype>
void gpu_gemm(cublasHandle_t handle, const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const Dtype alpha, const Dtype *A, const Dtype *B,
              const Dtype beta, Dtype *C);

template <typename Dtype>
void gpu_addition(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                  cudaStream_t stream);

template <typename Dtype>
void gpu_multiplication(const int N, const Dtype *a, const Dtype *b, Dtype *y,
                        cudaStream_t stream);

template <typename Dtype>
void col2row_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream);

template <typename Dtype>
void row2col_major(const int nrows, const int ncols, const Dtype *colA,
                   Dtype *rowA, cudaStream_t stream);

template <typename allocator_type>
void sort_coo_gpu(cusparseHandle_t handle, const int m, const int n,
                  const int nnz, int *d_coo_row, int *d_coo_col,
                  allocator_type &allocator);

namespace detail {

// copy_kernel_map for block thread > length
template <typename Dtype, typename Itype>
__global__ void __shared_copy_kernel_map(Dtype *__restrict__ dst,
                                         const Dtype *__restrict__ const src,
                                         const Itype *__restrict__ const map,
                                         const Itype nthreads,
                                         const Itype length) {
  // cchoy: cache map and benchmark.
  extern __shared__ unsigned int smap[];
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const Itype src_index = i / length;
  const Itype length_index = i % length;
  const Itype block_rem = (blockIdx.x * blockDim.x) % length;
  const Itype smap_index = (threadIdx.x + block_rem) / length;
  if ((threadIdx.x == 0 || (threadIdx.x + block_rem) % length == 0) &&
      i < nthreads)
    smap[smap_index] = map[src_index];
  __syncthreads();
  if (i < nthreads) {
    dst[i] = src[smap[smap_index] * length + length_index];
  }
}

template <typename Dtype, typename Itype>
__global__ void
__shared_accumulate_kernel_map(Dtype *__restrict__ dst,
                               const Dtype *__restrict__ const src,
                               const Itype *__restrict__ const map,
                               const Itype nthreads, const Itype length) {
  // cchoy: cache map and benchmark.
  extern __shared__ unsigned int smap[];
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const Itype src_index = i / length;
  const Itype length_index = i % length;
  const Itype block_rem = (blockIdx.x * blockDim.x) % length;
  const Itype smap_index = (threadIdx.x + block_rem) / length;
  if ((threadIdx.x == 0 || (threadIdx.x + block_rem) % length == 0) &&
      i < nthreads)
    smap[smap_index] = map[src_index];
  __syncthreads();
  if (i < nthreads)
    gpuAtomicAdd(&dst[smap[smap_index] * length + length_index], src[i]);
}

// Launches on `stream` so the copies are stream-ordered with the cublas
// calls that consume them (torch's blas handle runs on the current stream);
// under default-stream usage stream == 0 and the launch is unchanged.
template <typename Dtype, typename Itype>
void shared_copy_kernel_map(Dtype *dst, const Dtype *const src,
                            const Itype *const map, const Itype nthreads,
                            const Itype length, cudaStream_t stream = 0) {
  constexpr Itype MAX_THREADS = 512;
  if (MAX_THREADS >= length) {
    LOG_DEBUG("Blocks:", GET_BLOCKS(nthreads, MAX_THREADS),
              "Threads:", MAX_THREADS,
              "Shared:", GET_BLOCKS(MAX_THREADS, length));
    __shared_copy_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(MAX_THREADS, length) * sizeof(unsigned int), stream>>>(
            dst, src, map, nthreads, length);
  } else {
    LOG_DEBUG("Blocks:", GET_BLOCKS(nthreads, MAX_THREADS),
              "Threads:", MAX_THREADS,
              "Shared:", GET_BLOCKS(length, MAX_THREADS));
    __shared_copy_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(length, MAX_THREADS) * sizeof(unsigned int), stream>>>(
            dst, src, map, nthreads, length);
  }
}

template <typename Dtype, typename Itype>
void shared_accumulate_kernel_map(Dtype *dst, const Dtype *const src,
                                  const Itype *const map, const Itype nthreads,
                                  const Itype length, cudaStream_t stream = 0) {
  constexpr Itype MAX_THREADS = 512;
  if (MAX_THREADS >= length)
    __shared_accumulate_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(MAX_THREADS, length) * sizeof(unsigned int), stream>>>(
            dst, src, map, nthreads, length);
  else
    __shared_accumulate_kernel_map<Dtype, Itype>
        <<<GET_BLOCKS(nthreads, MAX_THREADS), MAX_THREADS,
           GET_BLOCKS(length, MAX_THREADS) * sizeof(unsigned int), stream>>>(
            dst, src, map, nthreads, length);
}

/*******************************************************************************
 * Fused (batched over kernel offsets) vectorized gather / scatter-accumulate
 * for the copy-GEMM convolution path.
 *
 * The kernel map stores the per-offset in/out index arrays contiguously,
 * sorted by offset (see gpu_kernel_map::decompose), so one launch over the
 * concatenated map replaces one launch per offset. Rows are contiguous in the
 * feature dimension; when the row byte-width and the base pointers allow it,
 * elements move in 16/8/4-byte chunks (8 halfs = one uint4) instead of one
 * scalar per thread.
 ******************************************************************************/

constexpr uint32_t FUSED_COPY_THREADS = 256;

template <int BYTES> struct aligned_chunk;
template <> struct aligned_chunk<16> { using type = uint4; };
template <> struct aligned_chunk<8> { using type = uint2; };
template <> struct aligned_chunk<4> { using type = unsigned int; };

// Largest chunk size (bytes) that divides the row width and keeps every row
// of both operands chunk-aligned; 0 requests the scalar path.
inline int fused_copy_chunk_bytes(size_t const row_bytes, void const *a,
                                  void const *b) {
  auto fits = [&](size_t n) {
    return row_bytes % n == 0 && reinterpret_cast<uintptr_t>(a) % n == 0 &&
           reinterpret_cast<uintptr_t>(b) % n == 0;
  };
  if (fits(16))
    return 16;
  if (fits(8))
    return 8;
  if (fits(4))
    return 4;
  return 0;
}

inline unsigned int fused_copy_grid(int64_t const total_chunks) {
  int64_t const blocks =
      (total_chunks + FUSED_COPY_THREADS - 1) / FUSED_COPY_THREADS;
  return static_cast<unsigned int>(
      std::min<int64_t>(blocks, std::numeric_limits<int>::max()));
}

// dst[r * cpr + c] = src[map[r] * cpr + c]; one ChunkT per thread. ChunkT is
// either an opaque 4/8/16-byte chunk or Dtype itself (scalar fallback).
template <typename ChunkT, typename Itype>
__global__ void fused_gather_kernel(ChunkT *__restrict__ dst,
                                    ChunkT const *__restrict__ src,
                                    Itype const *__restrict__ map,
                                    int64_t const total_chunks,
                                    uint32_t const chunks_per_row) {
  for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       idx < total_chunks; idx += int64_t(gridDim.x) * blockDim.x) {
    int64_t const row = idx / chunks_per_row;
    uint32_t const c = idx - row * chunks_per_row;
    dst[idx] = src[int64_t(map[row]) * chunks_per_row + c];
  }
}

// Atomic add of VEC consecutive elements. Rows from different kernel offsets
// can collide on the same destination row, so plain stores are not enough.
template <typename Dtype, int VEC> struct vec_atomic_add {
  static __device__ __forceinline__ void apply(Dtype *dst,
                                               Dtype const (&v)[VEC]) {
#pragma unroll
    for (int j = 0; j < VEC; ++j)
      gpuAtomicAdd(dst + j, v[j]);
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
template <int VEC> struct vec_atomic_add<c10::Half, VEC> {
  static __device__ __forceinline__ void apply(c10::Half *dst,
                                               c10::Half const (&v)[VEC]) {
    if constexpr (VEC % 2 == 0) {
      // dst is 4-byte aligned here: the vector path is only taken when the
      // row byte-width divides the chunk size, so every VEC-element slice
      // starts on a chunk boundary.
      __half2 *dst2 = reinterpret_cast<__half2 *>(dst);
      __half const *vh = reinterpret_cast<__half const *>(v);
#pragma unroll
      for (int j = 0; j < VEC / 2; ++j)
        atomicAdd(dst2 + j, __halves2half2(vh[2 * j], vh[2 * j + 1]));
    } else {
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        gpuAtomicAdd(dst + j, v[j]);
    }
  }
};
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <int VEC> struct vec_atomic_add<c10::BFloat16, VEC> {
  static __device__ __forceinline__ void apply(c10::BFloat16 *dst,
                                               c10::BFloat16 const (&v)[VEC]) {
    if constexpr (VEC % 2 == 0) {
      __nv_bfloat162 *dst2 = reinterpret_cast<__nv_bfloat162 *>(dst);
      __nv_bfloat16 const *vb = reinterpret_cast<__nv_bfloat16 const *>(v);
#pragma unroll
      for (int j = 0; j < VEC / 2; ++j)
        atomicAdd(dst2 + j, __halves2bfloat162(vb[2 * j], vb[2 * j + 1]));
    } else {
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        gpuAtomicAdd(dst + j, v[j]);
    }
  }
};
#endif

// dst[map[r], :] += src[r, :]; one VEC-element slice per thread, vector load
// from src, atomic adds into dst.
template <typename Dtype, int VEC, typename Itype>
__global__ void fused_scatter_add_kernel(Dtype *__restrict__ dst,
                                         Dtype const *__restrict__ src,
                                         Itype const *__restrict__ map,
                                         int64_t const total_chunks,
                                         uint32_t const chunks_per_row) {
  for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       idx < total_chunks; idx += int64_t(gridDim.x) * blockDim.x) {
    int64_t const row = idx / chunks_per_row;
    uint32_t const c = idx - row * chunks_per_row;
    Dtype vals[VEC];
    if constexpr (VEC == 1) {
      vals[0] = src[idx];
    } else {
      using ChunkT = typename aligned_chunk<VEC * sizeof(Dtype)>::type;
      *reinterpret_cast<ChunkT *>(vals) =
          reinterpret_cast<ChunkT const *>(src)[idx];
    }
    vec_atomic_add<Dtype, VEC>::apply(
        dst + (int64_t(map[row]) * chunks_per_row + c) * VEC, vals);
  }
}

// Gather nrows rows of nchannel elements: dst[r, :] = src[map[r], :].
template <typename Dtype, typename Itype>
void fused_gather(Dtype *dst, Dtype const *src, Itype const *map,
                  size_t const nrows, size_t const nchannel,
                  cudaStream_t stream) {
  if (nrows == 0)
    return;
  size_t const row_bytes = nchannel * sizeof(Dtype);
  int const chunk_bytes = fused_copy_chunk_bytes(row_bytes, dst, src);

  auto launch = [&](auto chunk_tag) {
    using ChunkT = decltype(chunk_tag);
    uint32_t const cpr = row_bytes / sizeof(ChunkT);
    int64_t const total = int64_t(nrows) * cpr;
    fused_gather_kernel<ChunkT, Itype>
        <<<fused_copy_grid(total), FUSED_COPY_THREADS, 0, stream>>>(
            reinterpret_cast<ChunkT *>(dst),
            reinterpret_cast<ChunkT const *>(src), map, total, cpr);
  };

  switch (chunk_bytes) {
  case 16:
    launch(uint4{});
    break;
  case 8:
    launch(uint2{});
    break;
  case 4:
    launch(0u);
    break;
  default:
    launch(Dtype{});
    break;
  }
  CUDA_CHECK(cudaGetLastError());
}

// Scatter-accumulate nrows rows of nchannel elements:
// dst[map[r], :] += src[r, :] (atomic; rows may collide across offsets).
template <typename Dtype, typename Itype>
void fused_scatter_add(Dtype *dst, Dtype const *src, Itype const *map,
                       size_t const nrows, size_t const nchannel,
                       cudaStream_t stream) {
  if (nrows == 0)
    return;
  size_t const row_bytes = nchannel * sizeof(Dtype);
  int const chunk_bytes = fused_copy_chunk_bytes(row_bytes, dst, src);
  int const vec = chunk_bytes / int(sizeof(Dtype));

  auto launch = [&](auto vec_tag) {
    constexpr int VEC = decltype(vec_tag)::value;
    // Chunks larger than 16 bytes are never selected; the guard only keeps
    // invalid aligned_chunk instantiations out of the other dtypes.
    if constexpr (VEC * sizeof(Dtype) <= 16) {
      uint32_t const cpr = nchannel / VEC;
      int64_t const total = int64_t(nrows) * cpr;
      fused_scatter_add_kernel<Dtype, VEC, Itype>
          <<<fused_copy_grid(total), FUSED_COPY_THREADS, 0, stream>>>(
              dst, src, map, total, cpr);
    }
  };

  switch (vec) {
  case 8:
    launch(std::integral_constant<int, 8>{});
    break;
  case 4:
    launch(std::integral_constant<int, 4>{});
    break;
  case 2:
    launch(std::integral_constant<int, 2>{});
    break;
  default:
    launch(std::integral_constant<int, 1>{});
    break;
  }
  CUDA_CHECK(cudaGetLastError());
}

} // end namespace detail

} // end namespace minkowski

#endif // MATH_FUNCTIONS
