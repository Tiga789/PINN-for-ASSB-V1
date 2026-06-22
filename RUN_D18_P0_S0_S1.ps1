param(
    [string]$Config = ".\configs\d18_p0_s0_s1.json",
    [string]$OutputRoot = "",
    [switch]$NoPlots
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[D18] Project root: $ProjectRoot"
Write-Host "[D18] P0/S0/S1 only. This script does not launch training."

$python = Get-Command python -ErrorAction Stop
& $python.Source --version
if ($LASTEXITCODE -ne 0) { throw "Python preflight failed." }

$required = @(
    ".\configs\d18_p0_s0_s1.json",
    ".\scripts\d18_run_p0_s0_s1.py",
    ".\gv1\d18_cycleaware\model_scaffold.py",
    ".\gv1\d18_cycleaware\diagnostics.py"
)
foreach ($path in $required) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Required D18 file is missing: $path"
    }
}

$argsList = @(
    ".\scripts\d18_run_p0_s0_s1.py",
    "--config", $Config
)
if ($OutputRoot -ne "") {
    $argsList += @("--output-root", $OutputRoot)
}
if ($NoPlots) {
    $argsList += "--no-plots"
}

Write-Host "[D18] Running: python $($argsList -join ' ')"
& $python.Source @argsList
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "D18 P0/S0/S1 returned exit code $code"
}
Write-Host "[D18] Completed. Read D18_P0_S0_S1_OVERALL_STATUS.md in the configured output root."
exit 0
