#pragma once
#include "common.cuh"

// k0: one thread per C element, everything from global memory.
// threadIdx.x walks along N so B and C accesses coalesce; A[row, k] is a warp-wide broadcast.
__global__ void k0_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                         int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = acc;
}

inline void k0_launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s = 0) {
  dim3 block(32, 8), grid(cdiv(N, 32), cdiv(M, 8));
  k0_naive<<<grid, block, 0, s>>>(A, B, C, M, N, K);
}
