#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CUDA_CHECK(x)                                                                         \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #x, __FILE__, __LINE__,              \
              cudaGetErrorString(e_));                                                        \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

#define CUBLAS_CHECK(x)                                                                       \
  do {                                                                                        \
    cublasStatus_t s_ = (x);                                                                  \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                        \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__);            \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

__host__ __device__ constexpr inline int cdiv(int a, int b) { return (a + b - 1) / b; }

struct Shape {
  int M, N, K;
  double flops() const { return 2.0 * M * N * K; }
};

// Owning device buffer. Move-only.
template <typename T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  explicit DevBuf(size_t n_) : n(n_) {
    if (n) CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  }
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      if (p) cudaFree(p);
      p = o.p, n = o.n;
      o.p = nullptr, o.n = 0;
    }
    return *this;
  }
  size_t bytes() const { return n * sizeof(T); }
};

// Inputs are small integers in [-2, 2]. Products and partial sums are then exact in FP16/FP32
// (|sum| <= 4K < 2^24 for K <= 4M), so every correct kernel matches cuBLAS bit-for-bit
// regardless of summation order, split-K, or atomics. Any nonzero diff is a real bug.
__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
__device__ __forceinline__ void store_val(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_val(half* p, float v) { *p = __float2half(v); }

template <typename T>
__global__ void fill_int_kernel(T* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    store_val(p + i, float(int(hash_u32((uint32_t)i * 2654435761u ^ seed) % 5u) - 2));
}

template <typename T>
inline void fill_int(T* p, size_t n, uint32_t seed) {
  fill_int_kernel<<<1024, 256>>>(p, n, seed);
  CUDA_CHECK(cudaGetLastError());
}

// max |a - b|; NaN/Inf in either input counts as a huge error (fmaxf would silently drop NaN).
__global__ void max_abs_diff_kernel(const float* a, const float* b, size_t n, float* out) {
  float m = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float d = fabsf(a[i] - b[i]);
    if (!(d <= 3e38f)) d = 3e38f;
    m = fmaxf(m, d);
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(out), __float_as_int(m));
}
