/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "math_functions.hpp"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace minkowski {

// 16-bit types have no CBLAS kernels; use plain loops with fp32 accumulation.
// The CPU path at reduced precision exists for API completeness, not speed.
namespace {

template <typename Dtype>
void cpu_gemm_16bit(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const int M, const int N,
                    const int K, const float alpha, const Dtype *A,
                    const Dtype *B, const float beta, Dtype *C) {
  // Index helpers honoring layout and transposition.
  auto index_A = [&](int m, int k) {
    if (Layout == CblasRowMajor)
      return (TransA == CblasNoTrans) ? m * K + k : k * M + m;
    return (TransA == CblasNoTrans) ? m + k * M : k + m * K;
  };
  auto index_B = [&](int k, int n) {
    if (Layout == CblasRowMajor)
      return (TransB == CblasNoTrans) ? k * N + n : n * K + k;
    return (TransB == CblasNoTrans) ? k + n * K : n + k * N;
  };
  auto index_C = [&](int m, int n) {
    return (Layout == CblasRowMajor) ? m * N + n : m + n * M;
  };
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<float>(A[index_A(m, k)]) *
               static_cast<float>(B[index_B(k, n)]);
      float const c_prev =
          (beta == 0.f) ? 0.f : beta * static_cast<float>(C[index_C(m, n)]);
      C[index_C(m, n)] = static_cast<Dtype>(alpha * acc + c_prev);
    }
  }
}

} // namespace

template <>
void cpu_gemm<float>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const float *B, const float beta, float *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_gemm<double>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N,
                      const int K, const double alpha, const double *A,
                      const double *B, const double beta, double *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_gemm<c10::Half>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                         const CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, const c10::Half alpha, const c10::Half *A,
                         const c10::Half *B, const c10::Half beta,
                         c10::Half *C) {
  cpu_gemm_16bit<c10::Half>(Layout, TransA, TransB, M, N, K,
                            static_cast<float>(alpha), A, B,
                            static_cast<float>(beta), C);
}

template <>
void cpu_gemm<c10::BFloat16>(const CBLAS_ORDER Layout,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, const int M,
                             const int N, const int K, const c10::BFloat16 alpha,
                             const c10::BFloat16 *A, const c10::BFloat16 *B,
                             const c10::BFloat16 beta, c10::BFloat16 *C) {
  cpu_gemm_16bit<c10::BFloat16>(Layout, TransA, TransB, M, N, K,
                                static_cast<float>(alpha), A, B,
                                static_cast<float>(beta), C);
}

template <>
void cpu_add<float>(const int n, const float *a, const float *b, float *y) {
  vsAdd(n, a, b, y);
}

template <>
void cpu_add<double>(const int n, const double *a, const double *b, double *y) {
  vdAdd(n, a, b, y);
}

template <>
void cpu_mul<float>(const int n, const float *a, const float *b, float *y) {
  vsMul(n, a, b, y);
}

template <>
void cpu_mul<double>(const int n, const double *a, const double *b, double *y) {
  vdMul(n, a, b, y);
}

template <>
void cpu_div<float>(const int n, const float *a, const float *b, float *y) {
  vsDiv(n, a, b, y);
}

template <>
void cpu_div<double>(const int n, const double *a, const double *b, double *y) {
  vdMul(n, a, b, y);
}

template <>
void cpu_axpy<float>(const int N, const float alpha, const float *X, float *Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void cpu_axpy<double>(const int N, const double alpha, const double *X,
                      double *Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

// 16-bit elementwise ops: scalar loops in fp32.
#define MINK_CPU_ELEMENTWISE_16BIT(Dtype)                                      \
  template <>                                                                  \
  void cpu_add<Dtype>(const int n, const Dtype *a, const Dtype *b, Dtype *y) { \
    for (int i = 0; i < n; ++i)                                                \
      y[i] = static_cast<Dtype>(static_cast<float>(a[i]) +                     \
                                static_cast<float>(b[i]));                     \
  }                                                                            \
  template <>                                                                  \
  void cpu_mul<Dtype>(const int n, const Dtype *a, const Dtype *b, Dtype *y) { \
    for (int i = 0; i < n; ++i)                                                \
      y[i] = static_cast<Dtype>(static_cast<float>(a[i]) *                     \
                                static_cast<float>(b[i]));                     \
  }                                                                            \
  template <>                                                                  \
  void cpu_div<Dtype>(const int n, const Dtype *a, const Dtype *b, Dtype *y) { \
    for (int i = 0; i < n; ++i)                                                \
      y[i] = static_cast<Dtype>(static_cast<float>(a[i]) /                     \
                                static_cast<float>(b[i]));                     \
  }                                                                            \
  template <>                                                                  \
  void cpu_axpy<Dtype>(const int N, const Dtype alpha, const Dtype *X,         \
                       Dtype *Y) {                                             \
    float const alpha_f = static_cast<float>(alpha);                           \
    for (int i = 0; i < N; ++i)                                                \
      Y[i] = static_cast<Dtype>(alpha_f * static_cast<float>(X[i]) +           \
                                static_cast<float>(Y[i]));                     \
  }

MINK_CPU_ELEMENTWISE_16BIT(c10::Half)
MINK_CPU_ELEMENTWISE_16BIT(c10::BFloat16)

#undef MINK_CPU_ELEMENTWISE_16BIT

} // end namespace minkowski
