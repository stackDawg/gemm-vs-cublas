// bench: driver for the GEMM kernel ladder, the autotuner, and correctness self-tests.
//
//   bench --mode info     --csv results/configs.csv       config table (regs, spills, occupancy) + device.csv
//   bench --mode selftest                                  CPU-validated cuBLAS + every kernel/config on odd shapes
//   bench --mode ladder   --shapes shapes/square.txt --csv results/ladder.csv
//   bench --mode tune     --shapes a.txt,b.txt --csv results/tune.csv
//   bench --mode single   --kernel <k0|k1|k2|k3|cublas|cublas32|config-name> --shape M,N,K [--splitk S] [--atomic]
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "common.cuh"
#include "configs.cuh"
#include "k0_naive.cuh"
#include "k1_smem_tiled.cuh"
#include "k2_regblock.cuh"
#include "k3_wmma.cuh"
#include "reference.cuh"
#include "tc_gemm.cuh"

// k0/k1 are skipped above this many FLOPs: they would take seconds per launch, and a WDDM
// display GPU resets any kernel that runs longer than ~2 s (TDR).
constexpr double SLOW_KERNEL_MAX_FLOPS = 1.4e11;  // just covers 4097^3
constexpr size_t WORKSPACE_CAP_BYTES = size_t(1) << 30;
constexpr float TOL = 1e-3f;  // inputs are small integers, so correct results are exact

struct Args {
  std::string mode = "ladder";
  std::vector<std::string> shape_files;
  std::string csv;
  std::string device_csv = "results/device.csv";
  std::string kernel = K5_CONFIG;
  Shape shape{4096, 4096, 4096};
  int splitk = 1, iters = 10;
  bool atomic = false, check = true;
  double budget_ms = 200;
};

struct Ctx {
  cublasHandle_t h{};
  cudaDeviceProp prop{};
  int sms = 0;
  DevBuf<char> flushbuf;
  DevBuf<float> ws;
  DevBuf<float> errbuf;
  cudaEvent_t e0{}, e1{};

  Ctx() {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    sms = prop.multiProcessorCount;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));  // FP32 SGEMM stays true FP32 (no TF32)
    flushbuf = DevBuf<char>(std::max<size_t>(2 * (size_t)prop.l2CacheSize, size_t(8) << 20));
    errbuf = DevBuf<float>(1);
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
  }
  ~Ctx() {
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cublasDestroy(h);
  }
  float* workspace(size_t n) {
    if (ws.n < n) ws = DevBuf<float>(n);
    return ws.p;
  }
};

// ---------------------------------------------------------------------------------------------
// Timing and checking

// Median of cudaEvent-timed runs, L2 flushed before each. The first call is an untimed warmup
// (it also triggers any lazy allocation such as the split-K workspace).
static double time_ms(Ctx& ctx, const std::function<void()>& fn, double budget_ms, int min_reps, int max_reps) {
  fn();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> ts;
  int reps = max_reps;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaMemsetAsync(ctx.flushbuf.p, r & 0xff, ctx.flushbuf.bytes()));
    CUDA_CHECK(cudaEventRecord(ctx.e0));
    fn();
    CUDA_CHECK(cudaEventRecord(ctx.e1));
    CUDA_CHECK(cudaEventSynchronize(ctx.e1));
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, ctx.e0, ctx.e1));
    ts.push_back(t);
    if (r == 0) reps = std::clamp((int)(budget_ms / std::max(t, 1e-3f)), min_reps, max_reps);
  }
  CUDA_CHECK(cudaGetLastError());
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

static float max_abs_diff(Ctx& ctx, const float* a, const float* b, size_t n) {
  CUDA_CHECK(cudaMemset(ctx.errbuf.p, 0, sizeof(float)));
  max_abs_diff_kernel<<<256, 256>>>(a, b, n, ctx.errbuf.p);
  CUDA_CHECK(cudaGetLastError());
  float h = 0;
  CUDA_CHECK(cudaMemcpy(&h, ctx.errbuf.p, sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

// Poison C with NaNs, run once, compare against the reference. Poisoning means a kernel that
// silently writes nothing can't pass on a previous kernel's output.
static float run_and_check(Ctx& ctx, const std::function<void()>& fn, float* C, const float* Cref, size_t n) {
  CUDA_CHECK(cudaMemset(C, 0xff, n * sizeof(float)));
  fn();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return max_abs_diff(ctx, C, Cref, n);
}

// ---------------------------------------------------------------------------------------------
// Tensor-core dispatch (k4/k5/k6 all go through here)

enum class SplitMode { Workspace, Atomic };

static int effective_splits(const TcConfig& cfg, int K, int split, int* kps_out = nullptr) {
  const int kps = cdiv(cdiv(K, split), cfg.BK) * cfg.BK;
  if (kps_out) *kps_out = kps;
  return cdiv(K, kps);
}

static void run_tc(Ctx& ctx, const TcConfig& cfg, const half* A, const half* B, float* C, int M, int N, int K,
                   int split, SplitMode sm, bool force_scalar = false, cudaStream_t s = 0) {
  int kps = 0;
  const int gz = effective_splits(cfg, K, split, &kps);
  const dim3 grid(cdiv(N, cfg.BN), cdiv(M, cfg.BM), gz);
  const bool vec = !force_scalar && K % 8 == 0 && N % 8 == 0;
  int mode = OUT_DIRECT;
  float* ws = nullptr;
  if (gz > 1) {
    if (sm == SplitMode::Atomic) {
      mode = OUT_ATOMIC;
      CUDA_CHECK(cudaMemsetAsync(C, 0, (size_t)M * N * sizeof(float), s));
    } else {
      mode = OUT_WORKSPACE;
      ws = ctx.workspace((size_t)gz * M * N);
    }
  }
  cfg.raw[vec ? 1 : 0](grid, A, B, C, ws, M, N, K, kps, mode, s);
  if (mode == OUT_WORKSPACE) splitk_reduce(ws, C, (size_t)M * N, gz, ctx.sms, s);
}

// ---------------------------------------------------------------------------------------------
// Problems

struct Fp32Problem {
  Shape s{};
  DevBuf<float> A, B, C, Cref;
};
struct Fp16Problem {
  Shape s{};
  DevBuf<half> A, B;
  DevBuf<float> C, Cref;
};

static Fp32Problem make_fp32(Ctx& ctx, Shape s) {
  Fp32Problem p;
  p.s = s;
  p.A = DevBuf<float>((size_t)s.M * s.K);
  p.B = DevBuf<float>((size_t)s.K * s.N);
  p.C = DevBuf<float>((size_t)s.M * s.N);
  p.Cref = DevBuf<float>((size_t)s.M * s.N);
  fill_int(p.A.p, p.A.n, 1u);
  fill_int(p.B.p, p.B.n, 2u);
  cublas_sgemm(ctx.h, p.A.p, p.B.p, p.Cref.p, s.M, s.N, s.K);
  CUDA_CHECK(cudaDeviceSynchronize());
  return p;
}

static Fp16Problem make_fp16(Ctx& ctx, Shape s) {
  Fp16Problem p;
  p.s = s;
  p.A = DevBuf<half>((size_t)s.M * s.K);
  p.B = DevBuf<half>((size_t)s.K * s.N);
  p.C = DevBuf<float>((size_t)s.M * s.N);
  p.Cref = DevBuf<float>((size_t)s.M * s.N);
  fill_int(p.A.p, p.A.n, 1u);
  fill_int(p.B.p, p.B.n, 2u);
  cublas_hgemm(ctx.h, p.A.p, p.B.p, p.Cref.p, s.M, s.N, s.K);
  CUDA_CHECK(cudaDeviceSynchronize());
  return p;
}

// ---------------------------------------------------------------------------------------------
// CSV output

struct Row {
  std::string suite, family, kernel, config, splitmode = "none";
  Shape s{};
  int split = 1;
  double ms = 0, cublas_ms = 0;
  float err = 0;
  bool checked = false;
};

static double tflops(Shape s, double ms) { return ms > 0 ? s.flops() / (ms * 1e-3) / 1e12 : 0; }

struct Csv {
  FILE* f = nullptr;
  explicit Csv(const std::string& path) {
    if (path.empty()) return;
    f = fopen(path.c_str(), "w");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", path.c_str());
      exit(1);
    }
    fprintf(f, "suite,M,N,K,family,kernel,config,splitk,splitmode,ms,tflops,cublas_ms,cublas_tflops,pct_cublas,"
               "max_err,ok\n");
  }
  ~Csv() {
    if (f) fclose(f);
  }
  void write(const Row& r) {
    const double tf = tflops(r.s, r.ms), ctf = tflops(r.s, r.cublas_ms);
    const double pct = r.cublas_ms > 0 ? 100.0 * r.cublas_ms / r.ms : 0;
    const char* ok = !r.checked ? "-" : (r.err <= TOL ? "ok" : "FAIL");
    printf("%-9s %5d %5d %5d  %-9s %-22s k=%-2d %-9s %9.4f ms %7.3f TF %6.1f%%  %s\n", r.suite.c_str(), r.s.M,
           r.s.N, r.s.K, r.kernel.c_str(), r.config.c_str(), r.split, r.splitmode.c_str(), r.ms, tf, pct, ok);
    if (r.checked && r.err > TOL) printf("    ^^^ max_err = %g\n", r.err);
    if (f) {
      fprintf(f, "%s,%d,%d,%d,%s,%s,%s,%d,%s,%.6f,%.4f,%.6f,%.4f,%.2f,%g,%s\n", r.suite.c_str(), r.s.M, r.s.N,
              r.s.K, r.family.c_str(), r.kernel.c_str(), r.config.c_str(), r.split, r.splitmode.c_str(), r.ms, tf,
              r.cublas_ms, ctf, pct, r.err, ok);
      fflush(f);
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Shapes

struct SuiteShape {
  std::string suite;
  Shape s;
};

static std::vector<SuiteShape> load_shapes(const std::vector<std::string>& files) {
  std::vector<SuiteShape> out;
  for (const auto& f : files) {
    std::ifstream in(f);
    if (!in) {
      fprintf(stderr, "cannot open shapes file %s\n", f.c_str());
      exit(1);
    }
    std::string suite = f.substr(f.find_last_of("/\\") + 1);
    suite = suite.substr(0, suite.find('.'));
    std::string line;
    while (std::getline(in, line)) {
      line = line.substr(0, line.find('#'));
      std::istringstream ls(line);
      Shape s{};
      if (ls >> s.M >> s.N >> s.K) out.push_back({suite, s});
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Modes

static void gpu_warmup(Ctx& ctx) {
  // Bring a laptop GPU up to its sustained boost clock before measuring anything.
  Fp16Problem p = make_fp16(ctx, {2048, 2048, 2048});
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1)) {
    for (int i = 0; i < 20; ++i) cublas_hgemm(ctx.h, p.A.p, p.B.p, p.C.p, 2048, 2048, 2048);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

static void mode_ladder(Ctx& ctx, const Args& a) {
  Csv csv(a.csv);
  const double budget = a.budget_ms;
  gpu_warmup(ctx);
  for (const auto& ss : load_shapes(a.shape_files)) {
    const Shape s = ss.s;
    const size_t mn = (size_t)s.M * s.N;

    auto measure = [&](Row r, const std::function<void()>& fn, const std::function<void()>& ref, float* C,
                       const float* Cref) {
      if (a.check) {
        r.err = run_and_check(ctx, fn, C, Cref, mn);
        r.checked = true;
      }
      r.cublas_ms = time_ms(ctx, ref, budget / 2, 3, 30);  // interleaved: cuBLAS timed right before
      r.ms = time_ms(ctx, fn, budget, 5, 50);
      csv.write(r);
    };

    {  // FP32 SIMT ladder vs cublasSgemm
      Fp32Problem p = make_fp32(ctx, s);
      auto ref = [&] { cublas_sgemm(ctx.h, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); };
      Row base;
      base.suite = ss.suite, base.family = "fp32", base.s = s;
      Row r = base;
      r.kernel = "cublas", r.config = "cublasSgemm";
      r.ms = r.cublas_ms = time_ms(ctx, ref, budget, 5, 50);
      csv.write(r);
      if (s.flops() <= SLOW_KERNEL_MAX_FLOPS) {
        r = base, r.kernel = "k0", r.config = "naive";
        measure(r, [&] { k0_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); }, ref, p.C.p, p.Cref.p);
        r = base, r.kernel = "k1", r.config = "smem32";
        measure(r, [&] { k1_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); }, ref, p.C.p, p.Cref.p);
      }
      r = base, r.kernel = "k2", r.config = "128x128x8_reg8x8";
      measure(r, [&] { k2_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); }, ref, p.C.p, p.Cref.p);
    }
    {  // FP16 tensor-core ladder vs cublasGemmEx
      Fp16Problem p = make_fp16(ctx, s);
      auto ref = [&] { cublas_hgemm(ctx.h, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); };
      Row base;
      base.suite = ss.suite, base.family = "fp16", base.s = s;
      Row r = base;
      r.kernel = "cublas", r.config = "cublasGemmEx";
      r.ms = r.cublas_ms = time_ms(ctx, ref, budget, 5, 50);
      csv.write(r);

      r = base, r.kernel = "k3", r.config = "wmma_128x128x32";
      measure(r, [&] { k3_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); }, ref, p.C.p, p.Cref.p);

      const TcConfig* k4 = find_tc_config(K4_CONFIG);
      r = base, r.kernel = "k4", r.config = k4->name;
      measure(r, [&] { run_tc(ctx, *k4, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, 1, SplitMode::Workspace); }, ref,
              p.C.p, p.Cref.p);

      const TcConfig* k5 = find_tc_config(K5_CONFIG);
      r = base, r.kernel = "k5", r.config = k5->name;
      measure(r, [&] { run_tc(ctx, *k5, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, 1, SplitMode::Workspace); }, ref,
              p.C.p, p.Cref.p);

      const TcChoice hc = heuristic_pick(s.M, s.N, s.K, ctx.sms);
      r = base, r.kernel = "k6_heur", r.config = hc.cfg->name, r.split = hc.split;
      r.splitmode = hc.split > 1 ? "workspace" : "none";
      measure(r,
              [&] { run_tc(ctx, *hc.cfg, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, hc.split, SplitMode::Workspace); },
              ref, p.C.p, p.Cref.p);
    }
  }
}

// Exhaustive search over configs x split-K for each shape. Emits one row per variant, plus
// "oracle" (best variant) and "heuristic" rows.
static void mode_tune(Ctx& ctx, const Args& a) {
  Csv csv(a.csv);
  const double budget = a.budget_ms / 2;
  gpu_warmup(ctx);
  const int splits[] = {1, 2, 4, 8, 16};
  for (const auto& ss : load_shapes(a.shape_files)) {
    const Shape s = ss.s;
    const size_t mn = (size_t)s.M * s.N;
    Fp16Problem p = make_fp16(ctx, s);
    auto ref = [&] { cublas_hgemm(ctx.h, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); };
    const double cub0 = time_ms(ctx, ref, budget, 5, 30);

    std::vector<Row> rows;
    auto variant = [&](const TcConfig& cfg, int split, SplitMode sm, const char* kernel) {
      Row r;
      r.suite = ss.suite, r.family = "fp16", r.s = s, r.kernel = kernel, r.config = cfg.name, r.split = split;
      r.splitmode = split == 1 ? "none" : (sm == SplitMode::Atomic ? "atomic" : "workspace");
      auto fn = [&] { run_tc(ctx, cfg, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, split, sm); };
      if (a.check) {
        r.err = run_and_check(ctx, fn, p.C.p, p.Cref.p, mn);
        r.checked = true;
      }
      r.ms = time_ms(ctx, fn, budget, 3, 20);
      return r;
    };

    for (const auto& cfg : tc_configs()) {
      if (!cfg.valid) continue;
      const int tiles = cdiv(s.M, cfg.BM) * cdiv(s.N, cfg.BN);
      for (int split : splits) {
        if (split > 1) {
          if (effective_splits(cfg, s.K, split) != split) continue;               // duplicate of a smaller split
          if (tiles * split > 4 * ctx.sms * cfg.blocks_per_sm) continue;          // grid already fills the GPU
          if ((size_t)split * mn * sizeof(float) > WORKSPACE_CAP_BYTES) continue;
        }
        rows.push_back(variant(cfg, split, SplitMode::Workspace, "tc"));
        if (split > 1) rows.push_back(variant(cfg, split, SplitMode::Atomic, "tc"));
      }
    }
    const TcChoice hc = heuristic_pick(s.M, s.N, s.K, ctx.sms);
    Row heur = variant(*hc.cfg, hc.split, SplitMode::Workspace, "heuristic");

    const double cub1 = time_ms(ctx, ref, budget, 5, 30);
    const double cub = 0.5 * (cub0 + cub1);
    if (std::fabs(cub1 - cub0) / cub > 0.05)
      printf("    (cuBLAS drifted %.1f%% during this shape: %.4f -> %.4f ms; clocks/thermals)\n",
             100.0 * (cub1 - cub0) / cub0, cub0, cub1);

    Row cr;
    cr.suite = ss.suite, cr.family = "fp16", cr.s = s, cr.kernel = "cublas", cr.config = "cublasGemmEx";
    cr.ms = cr.cublas_ms = cub;
    csv.write(cr);
    const Row* best = nullptr;
    for (auto& r : rows) {
      r.cublas_ms = cub;
      csv.write(r);
      if ((!r.checked || r.err <= TOL) && (!best || r.ms < best->ms)) best = &r;
    }
    heur.cublas_ms = cub;
    csv.write(heur);
    if (best) {
      Row o = *best;
      o.kernel = "oracle";
      csv.write(o);
    }
  }
}

static void mode_info(Ctx& ctx, const Args& a) {
  const auto& p = ctx.prop;
  printf("%s  sm_%d%d  SMs=%d  L2=%d KB  smem/SM=%zu KB  smem/block(optin)=%zu KB  regs/SM=%d  maxThreads/SM=%d\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1024, p.sharedMemPerMultiprocessor / 1024,
         p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor);

  // Measured device-to-device copy bandwidth (read + write), for the roofline.
  double bw_gbs = 0;
  {
    const size_t bytes = size_t(256) << 20;
    DevBuf<char> x(bytes), y(bytes);
    const double ms = time_ms(ctx, [&] { CUDA_CHECK(cudaMemcpyAsync(y.p, x.p, bytes, cudaMemcpyDeviceToDevice)); },
                              300, 5, 30);
    bw_gbs = 2.0 * bytes / (ms * 1e-3) / 1e9;
  }
  gpu_warmup(ctx);
  double cub16 = 0, cub32 = 0;
  {
    Fp16Problem q = make_fp16(ctx, {4096, 4096, 4096});
    cub16 = tflops(q.s, time_ms(ctx, [&] { cublas_hgemm(ctx.h, q.A.p, q.B.p, q.C.p, 4096, 4096, 4096); }, 500, 5, 30));
  }
  {
    Fp32Problem q = make_fp32(ctx, {4096, 4096, 4096});
    cub32 = tflops(q.s, time_ms(ctx, [&] { cublas_sgemm(ctx.h, q.A.p, q.B.p, q.C.p, 4096, 4096, 4096); }, 500, 5, 30));
  }
  printf("measured copy bandwidth: %.1f GB/s   cuBLAS 4096^3: fp16->fp32 %.2f TFLOPS, fp32 %.2f TFLOPS\n", bw_gbs,
         cub16, cub32);
  if (FILE* f = fopen(a.device_csv.c_str(), "w")) {
    fprintf(f, "name,cc,sms,l2_bytes,smem_per_sm,smem_optin,regs_per_sm,max_threads_per_sm,copy_bw_gbs,"
               "cublas_fp16_tflops,cublas_fp32_tflops\n");
    fprintf(f, "\"%s\",%d%d,%d,%d,%zu,%zu,%d,%d,%.1f,%.3f,%.3f\n", p.name, p.major, p.minor, p.multiProcessorCount,
            p.l2CacheSize, p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.regsPerMultiprocessor,
            p.maxThreadsPerMultiProcessor, bw_gbs, cub16, cub32);
    fclose(f);
  }

  FILE* f = a.csv.empty() ? nullptr : fopen(a.csv.c_str(), "w");
  if (f)
    fprintf(f, "config,BM,BN,BK,WM,WN,stages,minb,threads,smem_bytes,regs,local_bytes,blocks_per_sm,warps_per_sm,"
               "occupancy_pct,valid\n");
  printf("\n%-24s %7s %5s %6s %8s %6s %9s\n", "config", "threads", "regs", "spill", "smem KB", "blk/SM", "occupancy");
  for (const auto& c : tc_configs()) {
    const int warps = c.blocks_per_sm * c.threads / 32;
    const double occ = 100.0 * warps * 32 / p.maxThreadsPerMultiProcessor;
    printf("%-24s %7d %5d %6d %8.1f %6d %8.1f%%%s\n", c.name.c_str(), c.threads, c.regs, c.local_bytes,
           c.smem / 1024.0, c.blocks_per_sm, occ, c.valid ? "" : "  (invalid on this GPU)");
    if (f)
      fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%.1f,%d\n", c.name.c_str(), c.BM, c.BN, c.BK, c.WM, c.WN,
              c.STAGES, c.MINB, c.threads, c.smem, c.regs, c.local_bytes, c.blocks_per_sm, warps, occ, c.valid);
  }
  if (f) fclose(f);
}

static int mode_selftest(Ctx& ctx, const Args&) {
  int pass = 0, fail = 0;
  auto report = [&](bool ok, const std::string& what, float err) {
    ok ? ++pass : ++fail;
    if (!ok) printf("FAIL  %-60s max_err=%g\n", what.c_str(), err);
  };

  // 1) cuBLAS wrappers vs. a CPU reference (validates the row-major trick).
  for (Shape s : {Shape{7, 5, 3}, Shape{33, 17, 19}, Shape{64, 64, 64}, Shape{100, 37, 50}}) {
    const size_t mn = (size_t)s.M * s.N;
    std::vector<float> got(mn);
    {
      Fp32Problem p = make_fp32(ctx, s);
      std::vector<float> A(p.A.n), B(p.B.n);
      CUDA_CHECK(cudaMemcpy(A.data(), p.A.p, p.A.bytes(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(B.data(), p.B.p, p.B.bytes(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got.data(), p.Cref.p, mn * 4, cudaMemcpyDeviceToHost));
      auto ref = cpu_gemm(A, B, s.M, s.N, s.K);
      float e = 0;
      for (size_t i = 0; i < mn; ++i) e = std::max(e, std::fabs(ref[i] - got[i]));
      report(e <= TOL, "cublas_sgemm vs CPU " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K), e);
    }
    {
      Fp16Problem p = make_fp16(ctx, s);
      std::vector<half> A(p.A.n), B(p.B.n);
      CUDA_CHECK(cudaMemcpy(A.data(), p.A.p, p.A.bytes(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(B.data(), p.B.p, p.B.bytes(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got.data(), p.Cref.p, mn * 4, cudaMemcpyDeviceToHost));
      auto ref = cpu_gemm(A, B, s.M, s.N, s.K);
      float e = 0;
      for (size_t i = 0; i < mn; ++i) e = std::max(e, std::fabs(ref[i] - got[i]));
      report(e <= TOL, "cublas_hgemm vs CPU " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K), e);
    }
  }

  // 2) Every kernel, every config, both load paths, several split-K factors and both split modes,
  //    on shapes chosen to hit every edge: M/N/K below, at, and just past tile boundaries,
  //    odd sizes, K % 8 != 0, and K large enough for deep split-K.
  const Shape shapes[] = {{1, 8, 8},        {7, 13, 5},       {33, 65, 17},   {128, 128, 128}, {130, 250, 72},
                          {255, 257, 264},  {300, 40, 1000},  {1, 4096, 512}, {517, 129, 4104}, {64, 64, 8192},
                          {200, 136, 4096}, {16, 1000, 2048}};
  for (Shape s : shapes) {
    const size_t mn = (size_t)s.M * s.N;
    const std::string tag = " @ " + std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
    {
      Fp32Problem p = make_fp32(ctx, s);
      auto chk = [&](const std::string& name, const std::function<void()>& fn) {
        float e = run_and_check(ctx, fn, p.C.p, p.Cref.p, mn);
        report(e <= TOL, name + tag, e);
      };
      chk("k0", [&] { k0_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); });
      chk("k1", [&] { k1_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); });
      chk("k2", [&] { k2_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); });
      chk("k2/scalar", [&] { k2_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, true); });
    }
    {
      Fp16Problem p = make_fp16(ctx, s);
      auto chk = [&](const std::string& name, const std::function<void()>& fn) {
        float e = run_and_check(ctx, fn, p.C.p, p.Cref.p, mn);
        report(e <= TOL, name + tag, e);
      };
      chk("k3", [&] { k3_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K); });
      chk("k3/scalar", [&] { k3_launch(p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, true); });
      for (const auto& cfg : tc_configs()) {
        if (!cfg.valid) continue;
        for (int scalar = 0; scalar < 2; ++scalar)
          for (int split : {1, 3, 8})
            for (SplitMode sm : {SplitMode::Workspace, SplitMode::Atomic}) {
              if (split == 1 && sm == SplitMode::Atomic) continue;
              const std::string name = cfg.name + (scalar ? "/scalar" : "") + " split=" + std::to_string(split) +
                                       (sm == SplitMode::Atomic ? "/atomic" : "");
              chk(name, [&] { run_tc(ctx, cfg, p.A.p, p.B.p, p.C.p, s.M, s.N, s.K, split, sm, scalar != 0); });
            }
      }
    }
  }
  printf("selftest: %d passed, %d failed\n", pass, fail);
  return fail ? 1 : 0;
}

// Run one kernel repeatedly with no checking, for Nsight Compute.
static void mode_single(Ctx& ctx, const Args& a) {
  const Shape s = a.shape;
  const std::string& k = a.kernel;
  std::function<void()> fn;
  Fp32Problem p32;
  Fp16Problem p16;
  if (k == "k0" || k == "k1" || k == "k2" || k == "cublas32") {
    p32 = make_fp32(ctx, s);
    if (k == "k0") fn = [&] { k0_launch(p32.A.p, p32.B.p, p32.C.p, s.M, s.N, s.K); };
    if (k == "k1") fn = [&] { k1_launch(p32.A.p, p32.B.p, p32.C.p, s.M, s.N, s.K); };
    if (k == "k2") fn = [&] { k2_launch(p32.A.p, p32.B.p, p32.C.p, s.M, s.N, s.K); };
    if (k == "cublas32") fn = [&] { cublas_sgemm(ctx.h, p32.A.p, p32.B.p, p32.C.p, s.M, s.N, s.K); };
  } else {
    p16 = make_fp16(ctx, s);
    if (k == "k3") {
      fn = [&] { k3_launch(p16.A.p, p16.B.p, p16.C.p, s.M, s.N, s.K); };
    } else if (k == "cublas") {
      fn = [&] { cublas_hgemm(ctx.h, p16.A.p, p16.B.p, p16.C.p, s.M, s.N, s.K); };
    } else {
      const TcConfig* cfg = find_tc_config(k == "k4" ? K4_CONFIG : k == "k5" ? K5_CONFIG : k);
      if (!cfg) {
        fprintf(stderr, "unknown kernel/config '%s' (see --mode info for config names)\n", k.c_str());
        exit(2);
      }
      const SplitMode sm = a.atomic ? SplitMode::Atomic : SplitMode::Workspace;
      fn = [&, cfg, sm] { run_tc(ctx, *cfg, p16.A.p, p16.B.p, p16.C.p, s.M, s.N, s.K, a.splitk, sm); };
    }
  }
  const double ms = time_ms(ctx, fn, 1e9, a.iters, a.iters);
  printf("%s %dx%dx%d: median %.4f ms, %.3f TFLOPS\n", k.c_str(), s.M, s.N, s.K, ms, tflops(s, ms));
}

// ---------------------------------------------------------------------------------------------

static void usage() {
  printf(
      "usage: bench --mode info|selftest|ladder|tune|single [options]\n"
      "  --shapes f1.txt[,f2.txt]   shape files (lines of 'M N K', # comments)\n"
      "  --csv out.csv              output CSV (ladder, tune, info)\n"
      "  --device-csv path          device summary written by info (default results/device.csv)\n"
      "  --budget ms                per-variant timing budget (default 200; tune uses half)\n"
      "  --no-check                 skip correctness checks\n"
      "  --kernel name              single: k0|k1|k2|k3|k4|k5|cublas|cublas32|<config name>\n"
      "  --shape M,N,K              single: problem shape\n"
      "  --splitk S  --atomic       single: split-K factor and mode\n"
      "  --iters N                  single: timed iterations\n");
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", k.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (k == "--mode") a.mode = next();
    else if (k == "--shapes") {
      std::stringstream ss(next());
      std::string f;
      while (std::getline(ss, f, ',')) a.shape_files.push_back(f);
    } else if (k == "--csv") a.csv = next();
    else if (k == "--device-csv") a.device_csv = next();
    else if (k == "--budget") a.budget_ms = std::stod(next());
    else if (k == "--no-check") a.check = false;
    else if (k == "--kernel") a.kernel = next();
    else if (k == "--shape") {
      if (sscanf(next().c_str(), "%d,%d,%d", &a.shape.M, &a.shape.N, &a.shape.K) != 3) {
        fprintf(stderr, "--shape expects M,N,K\n");
        exit(2);
      }
    } else if (k == "--splitk") a.splitk = std::stoi(next());
    else if (k == "--atomic") a.atomic = true;
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "-h" || k == "--help") {
      usage();
      exit(0);
    } else {
      fprintf(stderr, "unknown argument %s\n", k.c_str());
      usage();
      exit(2);
    }
  }
  return a;
}

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  Ctx ctx;
  if (a.mode == "info") mode_info(ctx, a);
  else if (a.mode == "selftest") return mode_selftest(ctx, a);
  else if (a.mode == "ladder" || a.mode == "tune") {
    if (a.shape_files.empty()) {
      fprintf(stderr, "--shapes is required for %s\n", a.mode.c_str());
      return 2;
    }
    a.mode == "ladder" ? mode_ladder(ctx, a) : mode_tune(ctx, a);
  } else if (a.mode == "single") mode_single(ctx, a);
  else {
    usage();
    return 2;
  }
  return 0;
}
