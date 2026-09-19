# Builds build/bench.exe. Run from any PowerShell; locates MSVC via vswhere if cl.exe isn't on PATH.
#   ./build.ps1                 # sm_86 (RTX 30xx)
#   ./build.ps1 -Arch sm_89     # other GPUs
param([string]$Arch = "sm_86")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cl.exe not found and vswhere missing: install VS 2022 Build Tools (Desktop development with C++)." }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "No Visual Studio install with the C++ x64 toolset found." }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    Write-Host "Importing MSVC environment from $vcvars"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { throw "nvcc not found: install the CUDA Toolkit 12.x and reopen the shell." }

New-Item -ItemType Directory -Force build, results | Out-Null
$cmd = "nvcc -O3 -std=c++17 -arch=$Arch -lineinfo -Xptxas -v -Xcompiler /utf-8 -o build/bench.exe src/bench.cu -lcublas"
Write-Host $cmd
cmd /c "$cmd > build\ptxas.log 2>&1"
$code = $LASTEXITCODE
Get-Content build\ptxas.log | Select-String -Pattern "error|warning" | Where-Object { $_ -notmatch "ptxas info" } | ForEach-Object { Write-Host $_ }
if ($code -ne 0) { throw "nvcc failed (exit $code); see build\ptxas.log" }

# Surface any kernel that spills registers to local memory.
$spills = Get-Content build\ptxas.log | Select-String -Pattern "[1-9][0-9]* bytes spill (stores|loads)"
if ($spills) {
    Write-Host "`nKernels with register spills (see build\ptxas.log for which function each follows):" -ForegroundColor Yellow
    $spills | ForEach-Object { Write-Host "  $_" }
}
Write-Host "`nBuilt build\bench.exe"
