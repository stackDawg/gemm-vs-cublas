# Full pipeline: build, self-test, config info, ladder, autotune sweeps, plots.
# Plug the laptop in and set the Windows power mode to "Best performance" first.
param([switch]$SkipBuild, [switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipBuild) { ./build.ps1 }
$bench = ".\build\bench.exe"
New-Item -ItemType Directory -Force results | Out-Null

function Log-Clocks([string]$tag) {
    $q = nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader
    "$(Get-Date -Format s),$tag,$q" | Add-Content results\clocks.log
}

& $bench --mode selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed; not benchmarking broken kernels" }

& $bench --mode info --csv results\configs.csv --device-csv results\device.csv

$budget = if ($Quick) { 60 } else { 200 }
Log-Clocks "ladder-start"
& $bench --mode ladder --shapes shapes\square.txt,shapes\awkward.txt --csv results\ladder.csv --budget $budget
Log-Clocks "ladder-end"

Log-Clocks "tune-start"
& $bench --mode tune --shapes shapes\square.txt,shapes\awkward.txt,shapes\llm.txt,shapes\skinny.txt,shapes\sawtooth.txt --csv results\tune.csv --budget $budget
Log-Clocks "tune-end"

python scripts\plot.py
