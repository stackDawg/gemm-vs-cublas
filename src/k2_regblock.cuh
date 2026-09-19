#pragma once
#include "common.cuh"

// k2: 128x128 block tile, BK = 8, 256 threads, each thread owns an 8x8 micro-tile of C in
// registers. Per k-step a thread loads 8 A values + 8 B values from shared memory and does 64
// FMAs, so the FMA:smem-load ratio goes from 1:2 (k1) to 4:1. A is stored transposed in smem so
// both operands are read as float4.
//
// The thread's 8 rows are split as {ty*4..+3} and {64+ty*4..+3} (same for columns). Adjacent
// threads therefore read adjacent 16B chunks, avoiding the bank conflicts an 8-contiguous
// layout would cause.
//
// VEC: 16B global loads/stores. Requires K % 4 == 0 and N % 4 == 0 (row starts 16B-aligned);
// otherwise the scalar path handles any shape.
template <bool VEC>
__global__ void __launch_bounds__(256)
k2_regblock(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 8;
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  const int ty = tid / 16, tx = tid % 16;
  const int a_r = tid / 2, a_c = (tid % 2) * 4;   // A tile 128x8: one float4 per thread
  const int b_r = tid / 32, b_c = (tid % 32) * 4;  // B tile 8x128: one float4 per thread

  float acc[8][8] = {};
  float ra[8], rb[8];

  for (int k0 = 0; k0 < K; k0 += BK) {
    {
      const int gr = bm + a_r, gc = k0 + a_c;
      const float* src = A + (size_t)gr * K + gc;
      float v[4];
      if (VEC && gr < M && gc + 3 < K) {
        float4 t = *reinterpret_cast<const float4*>(src);
        v[0] = t.x, v[1] = t.y, v[2] = t.z, v[3] = t.w;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = (gr < M && gc + i < K) ? src[i] : 0.f;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) As[a_c + i][a_r] = v[i];
    }
    {
      const int gr = k0 + b_r, gc = bn + b_c;
      const float* src = B + (size_t)gr * N + gc;
      float4 t;
      if (VEC && gr < K && gc + 3 < N) {
        t = *reinterpret_cast<const float4*>(src);
      } else {
        bool r = gr < K;
        t.x = (r && gc + 0 < N) ? src[0] : 0.f;
        t.y = (r && gc + 1 < N) ? src[1] : 0.f;
        t.z = (r && gc + 2 < N) ? src[2] : 0.f;
        t.w = (r && gc + 3 < N) ? src[3] : 0.f;
      }
      *reinterpret_cast<float4*>(&Bs[b_r][b_c]) = t;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float4 a0 = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
      float4 a1 = *reinterpret_cast<const float4*>(&As[k][64 + ty * 4]);
      float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
      float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][64 + tx * 4]);
      ra[0] = a0.x, ra[1] = a0.y, ra[2] = a0.z, ra[3] = a0.w;
      ra[4] = a1.x, ra[5] = a1.y, ra[6] = a1.z, ra[7] = a1.w;
      rb[0] = b0.x, rb[1] = b0.y, rb[2] = b0.z, rb[3] = b0.w;
      rb[4] = b1.x, rb[5] = b1.y, rb[6] = b1.z, rb[7] = b1.w;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = bm + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
    if (r >= M) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = bn + h * 64 + tx * 4;
      float* dst = C + (size_t)r * N + c;
      if (VEC && c + 3 < N) {
        *reinterpret_cast<float4*>(dst) = make_float4(acc[i][h * 4 + 0], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (c + j < N) dst[j] = acc[i][h * 4 + j];
      }
    }
  }
}

inline void k2_launch(const float* A, const float* B, float* C, int M, int N, int K, bool force_scalar = false,
                      cudaStream_t s = 0) {
  dim3 grid(cdiv(N, 128), cdiv(M, 128));
  if (!force_scalar && K % 4 == 0 && N % 4 == 0)
    k2_regblock<true><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
  else
    k2_regblock<false><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
}
