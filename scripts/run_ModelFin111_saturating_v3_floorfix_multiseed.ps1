param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProjectRoot = ".",
  [int[]]$Seeds = @(42, 7, 2026),
  [int]$Epochs = 5000,
  [string]$Device = "cuda",
  [double]$FixedSOHFloor = 0.72,
  [switch]$AllowCPU,
  [switch]$SoftFailEval
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

$evalDirs = @()
$modelDirs = @()

foreach ($seed in $Seeds) {
  $suffix = if ($seed -eq 42) { "" } else { "_seed$seed" }
  $workDir = "Data\assb111_saturating_v3_floorfix$suffix"
  $modelDir = "ModelFin_111_saturating_v3_floorfix$suffix"
  $evalDir = "EvalFin_111_saturating_v3_floorfix${suffix}_strict30_test70"

  Write-Host "`n========== ASSB111 saturating_v3_floorfix seed=$seed ==========" -ForegroundColor Yellow

  $runArgs = @(
    "-PythonExe", $PythonExe,
    "-ProjectRoot", ".",
    "-InputFile", "input_assb111_strict30_saturating_v3_floorfix",
    "-WorkDir", $workDir,
    "-ModelDir", $modelDir,
    "-EvalDir", $evalDir,
    "-Epochs", $Epochs,
    "-Device", $Device,
    "-Seed", $seed,
    "-SOHModelVariant", "saturating_v3",
    "-InitializerFloorMode", "fixed_prior",
    "-FixedSOHFloor", $FixedSOHFloor,
    "-FloorMin", "0.68",
    "-FloorMax", "0.78",
    "-SOHFloorPrior", $FixedSOHFloor,
    "-DamageRateScale", "5e-4",
    "-GateGamma", "1.0",
    "-ResidualBound", "0.004",
    "-SOHNumericMin", "0.60",
    "-WFloorPrior", "1.0",
    "-WTailGuard", "0.10",
    "-LR", "3e-4",
    "-RunOverdecayDiagnostics"
  )
  if ($AllowCPU) { $runArgs += "-AllowCPU" }
  if ($SoftFailEval) { $runArgs += "-SoftFailEval" }

  & .\scripts\run_ModelFin111_strict30.ps1 @runArgs
  if ($LASTEXITCODE -ne 0) { throw "run_ModelFin111_strict30.ps1 failed for seed=$seed" }
  $evalDirs += $evalDir
  $modelDirs += $modelDir

  & $PythonExe .\scripts\diagnose_assb111_v3_floorfix.py `
    --initializer_json "$workDir\saturating_v3_initializer.json" `
    --pred_csv "$evalDir\soh_pred_by_cycle.csv" `
    --output_json "$evalDir\floorfix_diagnostic.json" `
    --expected_floor $FixedSOHFloor
}

$summaryDir = "EvalFin_111_saturating_v3_floorfix_seed_stability"
New-Item -ItemType Directory -Force $summaryDir | Out-Null

& $PythonExe .\scripts\compare_assb111_saturating_v3_seeds.py `
  --eval_dirs $evalDirs `
  --output_dir $summaryDir `
  --target_r2 0.98 `
  --max_r2_std 0.01 `
  --max_mae_mean 0.006 `
  --require_no_clamp

& $PythonExe .\scripts\diagnose_assb111_saturating_v3_stability.py `
  --model_dirs $modelDirs `
  --eval_dirs $evalDirs `
  --output_dir $summaryDir

Write-Host "`nASSB111 saturating_v3_floorfix multi-seed run completed." -ForegroundColor Green
Write-Host "Seed stability summary: $summaryDir"
