#pragma once
#include "common.cuh"

// All matrices are row-major: A is MxK, B is KxN, C is MxN.
// cuBLAS is column-major, so compute C^T = B^T * A^T: a row-major KxN B is exactly a
// column-major NxK B^T with ld = N, and likewise for A and C. No explicit transposes needed.

inline void cublas_sgemm(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

// FP16 inputs, FP32 accumulate, FP32 output: the same contract as our tensor-core kernels.
inline void cublas_hgemm(cublasHandle_t h, const half* A, const half* B, float* C, int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                            &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Host reference, used only on tiny shapes to validate the cuBLAS wrappers themselves.
template <typename T>
inline std::vector<float> cpu_gemm(const std::vector<T>& A, const std::vector<T>& B, int M, int N, int K) {
  auto f = [](T v) { return (float)v; };
  std::vector<float> C((size_t)M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)f(A[(size_t)i * K + k]) * f(B[(size_t)k * N + j]);
      C[(size_t)i * N + j] = (float)acc;
    }
  return C;
}
