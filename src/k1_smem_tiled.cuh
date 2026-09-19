#pragma once
#include "common.cuh"

// k1: classic 32x32 shared-memory tiling. Each input element is fetched from global memory
// K/32 times fewer than in k0, but every FMA still needs two shared-memory loads, so this is
// shared-memory-bandwidth bound.
template <int T>
__global__ void k1_smem_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                              int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];  // As: broadcast, Bs: conflict-free
    __syncthreads();
  }
  if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

inline void k1_launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s = 0) {
  constexpr int T = 32;
  dim3 block(T, T), grid(cdiv(N, T), cdiv(M, T));
  k1_smem_tiled<T><<<grid, block, 0, s>>>(A, B, C, M, N, K);
}
