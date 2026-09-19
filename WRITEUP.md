# Chasing cuBLAS: why the last 15% is a shape problem, not a cleverness problem

> **Status: draft.** The analysis and models below are written ahead of the measurements. Every
> ⟦TBD⟧ is filled from `results/summary.md` and the plots after running `scripts/run_all.ps1`.
> If a measurement contradicts a prediction here, the measurement wins and the text changes.

## 0. Setup

- **GPU:** RTX 3050 Laptop (GA107, sm_86). ⟦TBD: SM count, L2, measured copy bandwidth from `results/device.csv`⟧.
- **Contract:** C = A·B, row-major, FP16 inputs, FP32 accumulate and output, compared against
  `cublasGemmEx(CUDA_R_16F, CUDA_R_16F → CUDA_R_32F, CUBLAS_COMPUTE_32F)`. The SIMT stages (k0–k2)
  are FP32 and are compared against `cublasSgemm`.
- **Methodology:** each timing is the median of 5–50 cudaEvent-timed runs with L2 flushed before
  each run. cuBLAS is re-timed right next to each kernel, so both see the same thermal state. A 1 s
  warmup runs before each sweep. Clocks are **not locked** (it's a laptop). `results/clocks.log`
  records SM clock, temperature, and throttle reasons at the start and end of each sweep.
- **Correctness:** inputs are integers in [-2, 2], so every correct kernel matches cuBLAS exactly,
  whatever the summation order, split-K, or atomics. The selftest covers every config × both load
  paths × split-K factors on shapes chosen to hit every edge case. Caveat: integer-valued data draws
  slightly less power than random data, which can shift boost clocks on a power-limited laptop.
  Both sides see the same data, so the *ratios* are fair.

## 1. The ladder

![ladder](results/plots/ladder_4096.png)

| step | what it fixes | expected bottleneck after |
|---|---|---|
| k0 naive | — | global-memory traffic: every FMA does 2 loads (L1/L2 absorb some) |
| k1 smem tiles | global reuse ×32 | shared-memory bandwidth: still 2 smem loads per FMA |
| k2 register blocking | 8×8 micro-tile: 16 smem loads per 64 FMAs | FP32 pipe; roughly where FP32 cuBLAS lives |
| k3 WMMA | tensor cores | load → sync → compute serialization; padded smem |
| k4 mma.sync + ldmatrix + swizzle | explicit fragment layout, conflict-free smem | global-load latency fully exposed (single buffer) |
| k5 cp.async, 3 stages | overlaps loads of tile k+2 with math on tile k | tile shape / occupancy / shape effects: the rest of this writeup |
| k6 heuristic + split-K | picks tile and split per shape | the heuristic's own blind spots (§5) |

⟦TBD: numbers per step at 4096³, and the one-line takeaway for each jump. The expected big jumps
are k1→k2 (register reuse), k2→k3 (tensor cores), and k4→k5 (latency hiding).⟧

**Bank conflicts, k3 vs k4.** WMMA uses +8-half padding. k4 instead XORs the 16-byte chunk index
with the row (`swz()` in `tc_gemm.cuh`), so the 8 row addresses of every `ldmatrix` phase land in 8
distinct bank groups. For BK=32 two rows share a 128B window, so the XOR uses `row/2`.
⟦TBD: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` from `scripts/profile.ps1`,
expected ≈0 for k4/k5.⟧

## 2. Tile quantization and wave quantization

A block tile of BM×BN always computes a full tile, so the machine does
`ceil(M/BM)·BM × ceil(N/BN)·BN` worth of work:

    tile efficiency = (M·N) / (ceil(M/BM)·BM · ceil(N/BN)·BN)

The tiles are then scheduled in *waves* of `slots = SMs × blocks_per_SM`. The last wave is
partially empty, but it still costs a full wave of time:

    wave efficiency = tiles / (ceil(tiles / slots) · slots)

Both effects multiply. The sawtooth sweep (M = K = 2048, N = 1024…1536) makes this visible:

![sawtooth](results/plots/sawtooth.png)

Worked example with 128×128 tiles, assuming 16 SMs × 2 blocks = 32 slots (the exact values come
from `configs.csv`/`device.csv` and are what the bottom panel of the plot uses):

| N | tiles | waves | tile eff. | wave eff. | product |
|---|---|---|---|---|---|
| 1024 | 16×8 = 128 | 4.0 → 4 | 1.00 | 1.00 | 1.00 |
| 1032 | 16×9 = 144 | 4.5 → 5 | 0.90 | 0.90 | 0.81 |
| 1152 | 16×9 = 144 | 4.5 → 5 | 1.00 | 0.90 | 0.90 |
| 1280 | 16×10 = 160 | 5.0 → 5 | 1.00 | 1.00 | 1.00 |

Increasing N by 8 columns (0.8% more work) costs up to ~19% of throughput with a fixed tile.
cuBLAS's curve is flatter because it doesn't use a fixed tile. ⟦TBD: measured range for fixed vs
autotuned vs cuBLAS.⟧

**Two kinds of awkward.** The awkward suite separates two effects:

- `4097×4096×4096` (only M off by one) keeps the vector path, so it is pure quantization. 33 tile
  rows instead of 32 gives tile efficiency 4097/4224 = 97% plus a wave penalty.
- `4096×4096×4095`, `4095³`, `4097³` have K or N not divisible by 8. Rows are then not 16B-aligned,
  so there is no `cp.async` and no 16B loads, and the kernel falls back to scalar loads. That
  **alignment cliff** is usually far larger than the quantization effect. cuBLAS has kernels
  specialized for alignments of 1/2/4/8, and we have exactly two paths.

⟦TBD: awkward table from summary.md.⟧

## 3. Occupancy vs. register pressure

Per SM on sm_86: 65,536 registers, ~100 KB shared memory, 1,536 threads, 16 blocks.
A 256-thread block therefore fits

- 2 blocks/SM if it uses ≤128 registers/thread, 3 blocks if ≤85,
- and only as many blocks as its shared memory allows: 3 stages × 16 KB = 48 KB → 2 blocks.

The accumulators alone are `WTM·WTN/32` registers per thread: 64 for a 64×32 warp tile and 128 for
64×64. Big warp tiles are what give tensor-core GEMMs their arithmetic intensity, and they are
register-hungry by construction.

The `_mb2`/`_mb3` configs are the *same kernel* with `__launch_bounds__(256, MINB)` forcing more
blocks per SM. `_mb3` caps registers at 85, below what the tile needs, and ptxas spills.

![occupancy](results/plots/occupancy.png)

⟦TBD: config table from summary.md. Expected finding: higher occupancy does not buy speed. The
best configs run at 8–16 warps/SM. Each warp issues MI×NI independent `mma`s per k-step, and
`cp.async` keeps STAGES−1 tiles in flight. That is enough parallelism *within* a warp to hide
latency, so extra warps only add register and smem pressure. Forcing occupancy via
`__launch_bounds__` trades a latency problem (already solved) for local-memory spills (a new one).⟧

## 4. Skinny shapes and split-K

For M ≤ 64 with N = K = 4096, the arithmetic intensity is ≈ M FLOPs/byte (the weight matrix dominates
traffic), far below the ridge point of ⟦TBD⟧ FLOPs/byte. These shapes are bandwidth problems, and
the right yardstick is GB/s, not TFLOPS:

![llm](results/plots/llm.png)

⟦TBD: fraction of copy bandwidth achieved by ours and by cuBLAS at M = 1, 16, 64.⟧

When the *output* is small but K is long (the skinny suite: 64×64×65536 has one 128×128 tile),
there are fewer tiles than SMs, and most of the GPU idles. Split-K cuts K into S slices, runs
`tiles × S` blocks, and sums the slices afterwards. The extra cost is `2·S·M·N·4` bytes of workspace
traffic plus a second launch, or atomics:

![splitk](results/plots/splitk.png)

⟦TBD: best split per shape. Expected: gains grow as tiles ≪ slots, then flatten or reverse once
`tiles × S` exceeds a couple of waves. Atomics win at small S (no second launch) and lose as S grows
(contention, and the result becomes nondeterministic in general; it happens to be exact here
because the test data are integers).⟧

## 5. Why the last 15% is autotuning

The heatmap shows, for each shape, every config's throughput relative to the best config for that
shape:

![heatmap](results/plots/config_heatmap.png)

Then the same kernels under three selection policies:

- **fixed**: the single config with the best geomean over *all* shapes,
- **heuristic**: `heuristic_pick()` in `configs.cuh`, a readable ~30-line rule (skinny → skinny
  tiles, otherwise minimize waves × per-tile cost, split K when the grid can't fill the GPU),
- **oracle**: exhaustive search per shape.

![gap](results/plots/autotune_gap.png)

⟦TBD: the table. The argument, which the data must support:⟧

1. On nice square shapes the kernel itself is within ⟦TBD⟧% of cuBLAS. Remaining gaps there are
   microarchitectural (epilogue, fragment double-buffering, instruction scheduling). Those are real,
   but small.
2. On awkward and skinny shapes, *the same kernel code* spans ⟦TBD⟧ depending only on template
   parameters and split-K, and the best choice changes from shape to shape (heatmap).
3. The oracle recovers most of the gap between fixed and cuBLAS. The heuristic recovers part of it.
   The oracle-vs-heuristic gap is the cost of *selection*, not of code quality.
4. cuBLAS ships a very large set of kernels plus per-architecture selection heuristics. It is winning
   the selection problem. We could match it on a given shape by tuning for that shape, and we would
   still need a selector to generalize.

## 6. Limitations and next steps

- **Kernel gaps:** no smem-staged epilogue (stores are 32B-sector-efficient, but not 128B-coalesced),
  no explicit register double-buffering of fragments across k-steps (left to the compiler's
  unrolling), and no swizzled block rasterization for L2 locality on large shapes.
- **Stream-K** would remove wave quantization by construction: work is split into equal slices of
  the MAC loop regardless of tile boundaries. It's the natural next step after §2.
- **Alignment specializations** (4B and 8B vector paths) would close part of the awkward-shape cliff.
- Hopper/Blackwell features (wgmma, TMA, clusters) don't exist on sm_86. The tile/wave/selection
  argument carries over unchanged.
- Laptop clocks aren't locked, so run-to-run noise is ⟦TBD from repeated runs⟧%. Differences
  smaller than that aren't claimed.

## Implementation notes

- All kernels are headers compiled into one binary (`src/bench.cu`). Per-config register, spill,
  and occupancy numbers come from `cudaFuncGetAttributes` / `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  (`bench --mode info`). The raw ptxas output is in `build/ptxas.log`.
- `tc_gemm.cuh` is one template covering k4 (STAGES=1), k5 (STAGES≥2), and k6 (gridDim.z>1). The
  ladder therefore isolates exactly one change per step.
