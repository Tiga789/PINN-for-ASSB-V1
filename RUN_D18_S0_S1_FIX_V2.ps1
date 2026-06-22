param(
    [string]$PythonExe = "python",
    [string]$Config = ".\configs\d18_s0_s1_fix.json",
    [string]$OutputRoot = "",
    [switch]$NoPlots
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Required = @(
    ".\configs\d18_s0_s1_fix.json",
    ".\scripts\d18_run_s0_s1_fix.py",
    ".\scripts\d18_s1_build_dense_casepack.py",
    ".\gv1\d18_cycleaware\dense_casepack.py"
)
foreach ($File in $Required) {
    if (-not (Test-Path $File -PathType Leaf)) {
        throw "Required D18-S0/S1-FIX V2 file is missing: $File"
    }
}

$Arguments = @(".\scripts\d18_run_s0_s1_fix.py", "--config", $Config)
if ($OutputRoot -ne "") { $Arguments += @("--output-root", $OutputRoot) }
if ($NoPlots) { $Arguments += "--no-plots" }

Write-Host "[D18-S0/S1-FIX V2] Diagnostic-only run; D18-S2 training is disabled."
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) { throw "D18-S0/S1-FIX V2 exited with code $LASTEXITCODE" }
Write-Host "[D18-S0/S1-FIX V2] Completed. Default output:"
Write-Host "E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle_fix"
