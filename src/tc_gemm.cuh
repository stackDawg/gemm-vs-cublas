#pragma once
#include "common.cuh"

// Tensor-core GEMM built directly on PTX: mma.sync.m16n8k16 + ldmatrix + cp.async.
// One templated kernel covers three stages of the ladder:
//   k4: STAGES == 1  -> synchronous 16B loads into a single smem buffer (load, sync, compute, sync)
//   k5: STAGES >= 2  -> cp.async multistage pipeline (STAGES == 2 is classic double buffering)
//   k6: gridDim.z > 1 -> split-K; each z-slice covers a K range and writes FP32 partials to a
//       workspace (reduced by splitk_reduce) or atomically adds into C.
//
// Layout: A row-major MxK, B row-major KxN, C row-major MxN (FP32).
// smem: A tile [BM][BK], B tile [BK][BN], both split into 16-byte chunks (8 halves), with the
// chunk index XOR-swizzled by row so the 8 row addresses of every ldmatrix phase hit distinct
// banks (see swz()).

enum OutMode : int { OUT_DIRECT = 0, OUT_WORKSPACE = 1, OUT_ATOMIC = 2 };

__host__ __device__ constexpr size_t tc_smem_bytes(int BM, int BN, int BK, int STAGES) {
  return (size_t)(STAGES > 1 ? STAGES : 1) * (size_t)(BM * BK + BK * BN) * sizeof(half);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D = A * B + D, 16x8x16, FP16 in, FP32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// src_bytes == 0 zero-fills the 16B destination without reading global memory.
__device__ __forceinline__ void cp_async16(uint32_t saddr, const void* gptr, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(gptr), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Physical 16B-chunk index for (row, chunk) in a tile whose rows are CH chunks wide.
// ldmatrix reads 8 consecutive rows at the same logical chunk; a phase is conflict-free iff those
// 8 addresses land in 8 distinct 16B slots of a 128B bank window.
//  - CH >= 8: row stride is a multiple of 128B, so XOR the low 3 chunk bits with row % 8.
//  - CH == 4: two rows share a 128B window; XOR with (row / 2) % 4 so the pairs spread out.
template <int CH>
__device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (CH >= 8)
    return chunk ^ (row & 7);
  else
    return chunk ^ ((row / (8 / CH)) & (CH - 1));
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool VEC, int MINB>
__global__ void __launch_bounds__(WM * WN * 32, MINB)
tc_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, float* __restrict__ ws,
        int M, int N, int K, int k_per_split, int out_mode) {
  constexpr int THREADS = WM * WN * 32;
  constexpr int WTM = BM / WM, WTN = BN / WN;  // warp tile
  constexpr int MI = WTM / 16, NI = WTN / 8;   // mma tiles per warp
  constexpr int ACH = BK / 8, BCH = BN / 8;    // 16B chunks per smem row
  constexpr int A_STAGE = BM * BK, B_STAGE = BK * BN;
  constexpr int NBUF = STAGES > 1 ? STAGES : 1;
  static_assert(WTM % 16 == 0 && WTN % 16 == 0, "warp tile must be a multiple of 16x16");
  static_assert(BK % 16 == 0 && ACH >= 2, "BK must be a multiple of 16 (and >= 16)");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  half* As = reinterpret_cast<half*>(smem_raw);
  half* Bs = As + NBUF * A_STAGE;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  const int kbeg = blockIdx.z * k_per_split;
  const int kend = min(K, kbeg + k_per_split);
  const int ktiles = kend > kbeg ? cdiv(kend - kbeg, BK) : 0;

  float acc[MI][NI][4] = {};

  // Global -> smem for one BK slice. VEC requires K % 8 == 0 and N % 8 == 0 so every 16B chunk is
  // aligned and either fully in or fully out of bounds (kend is a multiple of BK or equals K).
  auto load_tile = [&](int stage, int k0) {
    half* as = As + stage * A_STAGE;
    half* bs = Bs + stage * B_STAGE;
#pragma unroll
    for (int it = 0; it < cdiv(BM * ACH, THREADS); ++it) {
      const int c = tid + it * THREADS;
      if (BM * ACH % THREADS != 0 && c >= BM * ACH) break;
      const int r = c / ACH, ch = c % ACH;
      const int gr = bm + r, gk = k0 + ch * 8;
      const half* src = A + (size_t)gr * K + gk;
      half* dst = as + r * BK + swz<ACH>(r, ch) * 8;
      if constexpr (VEC) {
        const bool in = gr < M && gk < kend;
        if constexpr (STAGES > 1)
          cp_async16(smem_u32(dst), in ? src : A, in ? 16 : 0);
        else
          *reinterpret_cast<uint4*>(dst) = in ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = (gr < M && gk + i < kend) ? src[i] : __float2half(0.f);
      }
    }
#pragma unroll
    for (int it = 0; it < cdiv(BK * BCH, THREADS); ++it) {
      const int c = tid + it * THREADS;
      if (BK * BCH % THREADS != 0 && c >= BK * BCH) break;
      const int r = c / BCH, ch = c % BCH;
      const int gk = k0 + r, gn = bn + ch * 8;
      const half* src = B + (size_t)gk * N + gn;
      half* dst = bs + r * BN + swz<BCH>(r, ch) * 8;
      if constexpr (VEC) {
        const bool in = gk < kend && gn < N;
        if constexpr (STAGES > 1)
          cp_async16(smem_u32(dst), in ? src : B, in ? 16 : 0);
        else
          *reinterpret_cast<uint4*>(dst) = in ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = (gk < kend && gn + i < N) ? src[i] : __float2half(0.f);
      }
    }
  };

  // Tensor-core math on one smem stage.
  // A fragment (16x16, row-major): lane l supplies row l%16, k-chunk l/16 -> regs a0..a3 in mma order.
  // B fragments via ldmatrix.trans on the [k][n] tile: lane l supplies k-row l%16, n-chunk l/16,
  // yielding (b0,b1) for two adjacent n8 tiles.
  auto compute_tile = [&](int stage) {
    const half* as = As + stage * A_STAGE;
    const half* bs = Bs + stage * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t af[MI][4];
      uint32_t bf[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = wm * WTM + mi * 16 + (lane & 15);
        const int ch = kk * 2 + (lane >> 4);
        ldmatrix_x4(af[mi], smem_u32(as + r * BK + swz<ACH>(r, ch) * 8));
      }
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        const int r = kk * 16 + (lane & 15);
        const int ch = (wn * WTN + nj * 16) / 8 + (lane >> 4);
        ldmatrix_x4_trans(bf[2 * nj][0], bf[2 * nj][1], bf[2 * nj + 1][0], bf[2 * nj + 1][1],
                          smem_u32(bs + r * BN + swz<BCH>(r, ch) * 8));
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
    }
  };

  if constexpr (STAGES == 1) {
    for (int kt = 0; kt < ktiles; ++kt) {
      load_tile(0, kbeg + kt * BK);
      __syncthreads();
      compute_tile(0);
      __syncthreads();
    }
  } else {
    // Prologue: put STAGES-1 tiles in flight. Always commit (possibly empty groups) so the
    // wait_group arithmetic below stays uniform.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ktiles) load_tile(s, kbeg + s * BK);
      cp_async_commit();
    }
    for (int kt = 0; kt < ktiles; ++kt) {
      cp_async_wait<STAGES - 2>();  // tile kt has landed (for this thread)...
      __syncthreads();              // ...and for everyone; also: all warps are done with tile kt-1
      const int nk = kt + STAGES - 1;
      if (nk < ktiles) load_tile(nk % STAGES, kbeg + nk * BK);  // overwrites tile kt-1's buffer
      cp_async_commit();
      compute_tile(kt % STAGES);
    }
    cp_async_wait<0>();
  }

  // Epilogue. Accumulator layout per m16n8 tile: lane holds rows g and g+8, cols 2*t4 and 2*t4+1.
  // Each quad of lanes writes a contiguous 32B segment, so stores are sector-efficient without
  // staging through smem.
  const int g = lane >> 2, t4 = lane & 3;
  float* out = out_mode == OUT_WORKSPACE ? ws + (size_t)blockIdx.z * M * N : C;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = bm + wm * WTM + mi * 16 + g + h * 8;
        const int c = bn + wn * WTN + ni * 8 + t4 * 2;
        if (r >= M) continue;
        const float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
        float* p = out + (size_t)r * N + c;
        if (out_mode == OUT_ATOMIC) {
          if (c < N) atomicAdd(p, v0);
          if (c + 1 < N) atomicAdd(p + 1, v1);
        } else if (c + 1 < N && (N & 1) == 0) {
          *reinterpret_cast<float2*>(p) = make_float2(v0, v1);
        } else {
          if (c < N) p[0] = v0;
          if (c + 1 < N) p[1] = v1;
        }
      }
}

// C = sum_z ws[z]. Vectorized when M*N % 4 == 0 (keeps every slice 16B-aligned).
template <bool VEC4>
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, float* __restrict__ C, size_t MN, int splits) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  if constexpr (VEC4) {
    const size_t n4 = MN / 4;
    const float4* w4 = reinterpret_cast<const float4*>(ws);
    float4* c4 = reinterpret_cast<float4*>(C);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride) {
      float4 s = w4[i];
      for (int z = 1; z < splits; ++z) {
        float4 t = w4[z * n4 + i];
        s.x += t.x, s.y += t.y, s.z += t.z, s.w += t.w;
      }
      c4[i] = s;
    }
  } else {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < MN; i += stride) {
      float s = ws[i];
      for (int z = 1; z < splits; ++z) s += ws[z * MN + i];
      C[i] = s;
    }
  }
}

inline void splitk_reduce(const float* ws, float* C, size_t MN, int splits, int sms, cudaStream_t s) {
  const bool v4 = MN % 4 == 0;
  const size_t work = v4 ? MN / 4 : MN;
  const int blocks = (int)std::min<size_t>((work + 255) / 256, (size_t)sms * 8);
  if (v4)
    splitk_reduce_kernel<true><<<blocks, 256, 0, s>>>(ws, C, MN, splits);
  else
    splitk_reduce_kernel<false><<<blocks, 256, 0, s>>>(ws, C, MN, splits);
}
