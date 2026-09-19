#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "tc_gemm.cuh"

// Registry of tc_gemm instantiations: the search space for the autotuner. Each config is
// compiled twice: with 16B vector/cp.async loads (VEC) and with the scalar fallback.

using TcRawLaunch = void (*)(dim3, const half*, const half*, float*, float*, int, int, int, int, int, cudaStream_t);

struct TcConfig {
  std::string name;
  int BM, BN, BK, WM, WN, STAGES, MINB, threads;
  size_t smem;
  const void* kern[2];  // [0] scalar fallback, [1] vector path
  TcRawLaunch raw[2];
  // Filled at init from the compiled kernels:
  bool valid = true;
  int regs = 0, local_bytes = 0, blocks_per_sm = 0;
};

template <int BM, int BN, int BK, int WM, int WN, int STAGES, int MINB, bool VEC>
void tc_raw_launch(dim3 grid, const half* A, const half* B, float* C, float* ws, int M, int N, int K, int kps,
                   int mode, cudaStream_t s) {
  tc_gemm<BM, BN, BK, WM, WN, STAGES, VEC, MINB>
      <<<grid, WM * WN * 32, tc_smem_bytes(BM, BN, BK, STAGES), s>>>(A, B, C, ws, M, N, K, kps, mode);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES, int MINB = 1>
TcConfig make_tc_config() {
  TcConfig c;
  char buf[64];
  snprintf(buf, sizeof buf, "%dx%dx%d_w%dx%d_s%d", BM, BN, BK, WM, WN, STAGES);
  c.name = buf;
  if (MINB > 1) c.name += "_mb" + std::to_string(MINB);
  c.BM = BM, c.BN = BN, c.BK = BK, c.WM = WM, c.WN = WN, c.STAGES = STAGES, c.MINB = MINB;
  c.threads = WM * WN * 32;
  c.smem = tc_smem_bytes(BM, BN, BK, STAGES);
  c.kern[0] = (const void*)tc_gemm<BM, BN, BK, WM, WN, STAGES, false, MINB>;
  c.kern[1] = (const void*)tc_gemm<BM, BN, BK, WM, WN, STAGES, true, MINB>;
  c.raw[0] = tc_raw_launch<BM, BN, BK, WM, WN, STAGES, MINB, false>;
  c.raw[1] = tc_raw_launch<BM, BN, BK, WM, WN, STAGES, MINB, true>;
  return c;
}

// Names used by the ladder (k4 / k5) and the heuristic.
constexpr const char* K4_CONFIG = "128x128x32_w2x4_s1";
constexpr const char* K5_CONFIG = "128x128x32_w2x4_s3";

inline std::vector<TcConfig>& tc_configs() {
  static std::vector<TcConfig> v = [] {
    std::vector<TcConfig> c = {
        // --- the ladder: same tile, increasing pipeline depth ---
        make_tc_config<128, 128, 32, 2, 4, 1>(),  // k4: no pipelining
        make_tc_config<128, 128, 32, 2, 4, 2>(),  // double buffering
        make_tc_config<128, 128, 32, 2, 4, 3>(),  // k5 default
        make_tc_config<128, 128, 32, 2, 4, 4>(),
        make_tc_config<128, 128, 64, 2, 4, 3>(),
        // --- big tiles: highest arithmetic intensity, worst quantization ---
        make_tc_config<128, 128, 32, 2, 2, 3>(),  // 4 warps, 64x64 warp tiles: register heavy
        make_tc_config<256, 128, 32, 4, 2, 3>(),
        make_tc_config<128, 256, 32, 2, 4, 3>(),
        // --- medium / small tiles: better wave fill on small or odd shapes ---
        make_tc_config<128, 64, 32, 2, 2, 3>(),
        make_tc_config<64, 128, 32, 2, 2, 3>(),
        make_tc_config<64, 64, 32, 2, 2, 3>(),
        make_tc_config<64, 64, 64, 2, 2, 4>(),
        // --- skinny: M (or N) tiny, e.g. LLM decode ---
        make_tc_config<32, 128, 64, 1, 4, 3>(),
        make_tc_config<16, 128, 64, 1, 4, 4>(),
        make_tc_config<16, 256, 64, 1, 8, 4>(),
        make_tc_config<128, 32, 64, 4, 1, 3>(),
        // --- register-pressure experiment: same kernel, forced to fit more blocks per SM ---
        make_tc_config<128, 128, 32, 2, 4, 2, 2>(),
        make_tc_config<128, 128, 32, 2, 4, 2, 3>(),
    };
    int dev = 0, optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    for (auto& x : c) {
      if (x.smem > (size_t)optin) {
        x.valid = false;
        continue;
      }
      for (int k = 0; k < 2; ++k)
        CUDA_CHECK(cudaFuncSetAttribute(x.kern[k], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)x.smem));
      cudaFuncAttributes fa{};
      CUDA_CHECK(cudaFuncGetAttributes(&fa, x.kern[1]));
      x.regs = fa.numRegs;
      x.local_bytes = (int)fa.localSizeBytes;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&x.blocks_per_sm, x.kern[1], x.threads, x.smem));
      if (x.blocks_per_sm == 0) x.valid = false;
    }
    return c;
  }();
  return v;
}

inline const TcConfig* find_tc_config(const std::string& name) {
  for (auto& c : tc_configs())
    if (c.name == name) return c.valid ? &c : nullptr;
  return nullptr;
}

struct TcChoice {
  const TcConfig* cfg;
  int split;
};

// A deliberately simple, explainable picker: roughly what a hand-written dispatch table does.
// Its gap to the exhaustive oracle is the "last 15%" the writeup is about.
//   1. Skinny shapes get the skinny tiles.
//   2. Otherwise pick the tile that minimizes (waves x per-tile cost), where a wave is one round
//      of SMs x blocks/SM tiles and per-tile cost falls with tile size (arithmetic intensity).
//   3. If even that grid can't fill the machine and K is long, split K.
inline TcChoice heuristic_pick(int M, int N, int K, int sms) {
  std::vector<const char*> cands;
  if (M <= 16)
    cands = {"16x128x64_w1x4_s4", "16x256x64_w1x8_s4"};
  else if (M <= 32)
    cands = {"32x128x64_w1x4_s3"};
  else if (N <= 32)
    cands = {"128x32x64_w4x1_s3"};
  else
    cands = {"256x128x32_w4x2_s3", "128x256x32_w2x4_s3", K5_CONFIG,
             "128x64x32_w2x2_s3",  "64x128x32_w2x2_s3",  "64x64x32_w2x2_s3"};

  const TcConfig* best = nullptr;
  double best_cost = 1e300;
  for (const char* n : cands) {
    const TcConfig* c = find_tc_config(n);
    if (!c) continue;
    const double tiles = double(cdiv(M, c->BM)) * cdiv(N, c->BN);
    const double waves = std::ceil(tiles / (double(sms) * c->blocks_per_sm));
    const double intensity = double(c->BM) * c->BN / (c->BM + c->BN);
    const double eff = intensity / (intensity + 16.0);
    const double cost = waves * c->BM * c->BN / eff;
    if (cost < best_cost) best_cost = cost, best = c;
  }
  if (!best) best = find_tc_config(K5_CONFIG);

  const int tiles = cdiv(M, best->BM) * cdiv(N, best->BN);
  const int slots = sms * best->blocks_per_sm;
  int split = 1;
  while (split < 16 && tiles * split * 2 <= slots && K / (split * 2) >= 512) split *= 2;
  return {best, split};
}
