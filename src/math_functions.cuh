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

} // end namespace detail

} // end namespace minkowski

#endif // MATH_FUNCTIONS
