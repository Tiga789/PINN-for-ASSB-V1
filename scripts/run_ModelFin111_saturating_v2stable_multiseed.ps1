param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProjectRoot = ".",
  [int[]]$Seeds = @(42, 7, 2026),
  [int]$Epochs = 5000,
  [string]$Device = "cuda",
  [switch]$AllowCPU,
  [switch]$SoftFailEval
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

$evalDirs = @()
$modelDirs = @()

foreach ($seed in $Seeds) {
  if ($seed -eq 42) {
    $workDir = "Data\assb111_saturating_v2stable"
    $modelDir = "ModelFin_111_saturating_v2stable"
    $evalDir = "EvalFin_111_saturating_v2stable_strict30_test70"
  } else {
    $workDir = "Data\assb111_saturating_v2stable_seed$seed"
    $modelDir = "ModelFin_111_saturating_v2stable_seed$seed"
    $evalDir = "EvalFin_111_saturating_v2stable_seed${seed}_strict30_test70"
  }

  Write-Host "`n==============================" -ForegroundColor Cyan
  Write-Host "ASSB-111 saturating_v2_stable seed=$seed" -ForegroundColor Cyan
  Write-Host "==============================" -ForegroundColor Cyan

  $args = @(
    "-PythonExe", $PythonExe,
    "-ProjectRoot", ".",
    "-InputFile", "input_assb111_strict30_saturating_v2stable",
    "-WorkDir", $workDir,
    "-ModelDir", $modelDir,
    "-EvalDir", $evalDir,
    "-Epochs", $Epochs,
    "-Device", $Device,
    "-Seed", $seed,
    "-SOHModelVariant", "saturating_v2_stable",
    "-InitializerFloorMode", "fixed_prior",
    "-FixedSOHFloor", 0.72,
    "-SOHFloorPrior", 0.72,
    "-FreezeSOHFloor", $true,
    "-FreezeSOH0", $false,
    "-LR", 5e-4,
    "-WeightDecay", 1e-5,
    "-ResidualBound", 0.006,
    "-RateCorrectionBound", 3.0,
    "-MinTrainR2ForBest", 0.990,
    "-MaxTrainMAEForBest", 0.0030,
    "-MaxValMAEForBest", 0.0020,
    "-RunOverdecayDiagnostics"
  )
  if ($AllowCPU) { $args += "-AllowCPU" }
  if ($SoftFailEval) { $args += "-SoftFailEval" }
  & .\scripts\run_ModelFin111_strict30.ps1 @args
  if ($LASTEXITCODE -ne 0) { throw "seed $seed run failed with exit code $LASTEXITCODE" }
  $evalDirs += $evalDir
  $modelDirs += $modelDir
}

$summaryDir = "EvalFin_111_saturating_v2stable_seed_stability"
New-Item -ItemType Directory -Force $summaryDir | Out-Null

$diagArgs = @(
  "scripts\diagnose_assb111_v2stable_stability.py",
  "--eval_dirs"
) + $evalDirs + @(
  "--model_dirs"
) + $modelDirs + @(
  "--output_dir", $summaryDir,
  "--target_r2", "0.98",
  "--max_r2_std", "0.01",
  "--max_mae_mean", "0.006",
  "--require_no_clamp"
)

Write-Host "`n==== ASSB-111 v2stable seed stability summary ====" -ForegroundColor Cyan
Write-Host ("python " + ($diagArgs -join " ")) -ForegroundColor DarkGray
& $PythonExe @diagArgs
if ($LASTEXITCODE -ne 0) {
  Write-Warning "v2stable seed stability summary did not meet target criteria; inspect $summaryDir"
}

Write-Host "`nCompleted ASSB-111 saturating_v2_stable multiseed runs." -ForegroundColor Green
Write-Host "Summary: $summaryDir"
