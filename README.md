# Chasing cuBLAS: a GEMM kernel ladder on an RTX 3050 Laptop GPU

Hand-written CUDA GEMMs, from naive up to a multistage tensor-core kernel with split-K, measured
against cuBLAS on square, skinny, and awkward shapes. See [WRITEUP.md](WRITEUP.md).

| stage | file | what it adds |
|---|---|---|
| k0 | `src/k0_naive.cuh` | one thread per output, FP32 |
| k1 | `src/k1_smem_tiled.cuh` | 32×32 shared-memory tiles |
| k2 | `src/k2_regblock.cuh` | 128×128 block tile, 8×8 register micro-tile, float4 |
| k3 | `src/k3_wmma.cuh` | tensor cores via WMMA (FP16 in, FP32 accumulate) |
| k4 | `src/tc_gemm.cuh` (STAGES=1) | raw `mma.sync` + `ldmatrix`, XOR-swizzled smem |
| k5 | `src/tc_gemm.cuh` (STAGES≥2) | `cp.async` multistage pipeline (double/triple buffering) |
| k6 | `src/tc_gemm.cuh` (gridDim.z>1) | split-K (workspace + reduce, or atomics) |
| — | `src/configs.cuh` | 18 tile configs, the autotuner's search space, and a heuristic picker |

## Setup (Windows)
1. Install **CUDA Toolkit 12.x** and **Visual Studio 2022 Build Tools** ("Desktop development with C++").
2. `pip install numpy pandas matplotlib`
3. Plug in the laptop and set Windows power mode to *Best performance*.

## Run
```powershell
./build.ps1                      # -> build/bench.exe, ptxas log in build/ptxas.log
./build/bench.exe --mode selftest
./scripts/run_all.ps1            # selftest, info, ladder, full autotune sweep, plots (~20-40 min)
./scripts/run_all.ps1 -Quick     # smaller timing budget
./scripts/profile.ps1            # Nsight Compute on k3/k4/k5/cuBLAS (needs perf-counter access)
```
Outputs: `results/*.csv`, `results/plots/*.png`, `results/summary.md`.

Profile a single kernel: `./build/bench.exe --mode single --kernel 64x128x32_w2x2_s3 --shape 1000,1000,1000 --splitk 2`.
