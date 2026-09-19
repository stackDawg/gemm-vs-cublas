#pragma once
#include <mma.h>
#include "common.cuh"

// k3: first tensor-core kernel, using the portable WMMA API. 128x128x32 block tile, 8 warps in a
// 2x4 grid, each warp computes a 64x32 tile as 4x2 16x16x16 fragments. Shared memory is padded
// (+8 halves per row) rather than swizzled, which WMMA's opaque fragment loads accept.
// No pipelining: load, sync, compute, sync. That serialization is what k4/k5 attack.
template <bool VEC>
__global__ void __launch_bounds__(256)
k3_wmma(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int BM = 128, BN = 128, BK = 32, PAD = 8;
  constexpr int LDA = BK + PAD, LDB = BN + PAD;
  __shared__ __align__(32) half As[BM * LDA];
  __shared__ __align__(32) half Bs[BK * LDB];
  __shared__ __align__(32) float scratch[8][16 * 16];  // per-warp staging for edge tiles

  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wm = warp / 4, wn = warp % 4;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int c = tid; c < BM * BK / 8; c += 256) {  // A: 128 rows x 4 chunks of 8 halves
      const int r = c / 4, cc = (c % 4) * 8;
      const int gr = bm + r, gk = k0 + cc;
      const half* src = A + (size_t)gr * K + gk;
      half* dst = As + r * LDA + cc;
      if (VEC && gr < M && gk + 7 < K) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = (gr < M && gk + i < K) ? src[i] : __float2half(0.f);
      }
    }
    for (int c = tid; c < BK * BN / 8; c += 256) {  // B: 32 rows x 16 chunks
      const int r = c / 16, cc = (c % 16) * 8;
      const int gk = k0 + r, gn = bn + cc;
      const half* src = B + (size_t)gk * N + gn;
      half* dst = Bs + r * LDB + cc;
      if (VEC && gk < K && gn + 7 < N) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = (gk < K && gn + i < N) ? src[i] : __float2half(0.f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(fa[i], As + (wm * 64 + i * 16) * LDA + kk, LDA);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], Bs + kk * LDB + wn * 32 + j * 16, LDB);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r0 = bm + wm * 64 + i * 16, c0 = bn + wn * 32 + j * 16;
      if (r0 + 16 <= M && c0 + 16 <= N && N % 4 == 0) {
        wmma::store_matrix_sync(C + (size_t)r0 * N + c0, acc[i][j], N, wmma::mem_row_major);
      } else {
        wmma::store_matrix_sync(scratch[warp], acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
          const int r = r0 + e / 16, c = c0 + e % 16;
          if (r < M && c < N) C[(size_t)r * N + c] = scratch[warp][e];
        }
        __syncwarp();
      }
    }
}

inline void k3_launch(const half* A, const half* B, float* C, int M, int N, int K, bool force_scalar = false,
                      cudaStream_t s = 0) {
  dim3 grid(cdiv(N, 128), cdiv(M, 128));
  if (!force_scalar && K % 8 == 0 && N % 8 == 0)
    k3_wmma<true><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
  else
    k3_wmma<false><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
}
