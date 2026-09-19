# Nsight Compute on selected kernels. Needs GPU performance-counter access: run as admin, or enable
# NVIDIA Control Panel > Developer > Manage GPU Performance Counters > "Allow access to all users".
param([string]$Shape = "4096,4096,4096")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force results\ncu | Out-Null

$metrics = @(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard",
    "smsp__average_warp_latency_issue_stalled_barrier",
    "smsp__average_warp_latency_issue_stalled_mio_throttle",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem"
) -join ","

# k3 (WMMA, padded smem), k4 (mma+ldmatrix, no pipeline), k5 (3-stage cp.async), register-limited
# variants, and cuBLAS for reference.
# Filter by kernel name, skipping the untimed warmup launch (and, for cuBLAS, the reference call).
$targets = [ordered]@{
    "k3"                     = @("regex:k3_wmma", 1)
    "k4"                     = @("regex:tc_gemm", 1)
    "128x128x32_w2x4_s2"     = @("regex:tc_gemm", 1)
    "k5"                     = @("regex:tc_gemm", 1)
    "128x128x32_w2x4_s2_mb2" = @("regex:tc_gemm", 1)
    "128x128x32_w2x4_s2_mb3" = @("regex:tc_gemm", 1)
    "cublas"                 = @("regex:gemm|xmma|cutlass", 2)
}
foreach ($k in $targets.Keys) {
    $filter, $skip = $targets[$k]
    $out = "results\ncu\$($k -replace '[^A-Za-z0-9_]', '_')"
    Write-Host "== $k"
    ncu -k $filter --launch-skip $skip --launch-count 1 --metrics $metrics --csv --page raw `
        .\build\bench.exe --mode single --kernel $k --shape $Shape --iters 3 | Out-File -Encoding utf8 "$out.csv"
    # Full report for the Nsight Compute GUI:
    ncu -k $filter --launch-skip $skip --launch-count 1 --set full -f -o $out `
        .\build\bench.exe --mode single --kernel $k --shape $Shape --iters 3 | Out-Null
}
Write-Host "Reports in results\ncu\"
