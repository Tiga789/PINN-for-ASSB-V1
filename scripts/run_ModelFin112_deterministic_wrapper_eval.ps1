param(
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$SohModelDir = ".\ModelFin_112_deterministicSOH_ridge_g4",
  [string]$OutputModelDir = ".\ModelFin_112_deterministic_wrapper",
  [string]$DatasetCsv = ".\Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv",
  [string]$OutputEvalDir = ".\EvalFin_112_deterministic_wrapper",
  [string]$StateScorecardCsv = "",
  [string]$StateEvalNpz = "",
  [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $Root

if ($Clean) {
  Remove-Item $OutputModelDir -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $OutputEvalDir -Recurse -Force -ErrorAction SilentlyContinue
}

$buildArgs = @(
  ".\scripts\build_ModelFin112_single_model.py",
  "--soh_model_dir", $SohModelDir,
  "--output_model_dir", $OutputModelDir,
  "--dataset_csv", $DatasetCsv,
  "--state_core_dir", ".\ModelFin_107A"
)
if ($StateScorecardCsv -ne "") { $buildArgs += @("--state_scorecard_csv", $StateScorecardCsv) }
if ($StateEvalNpz -ne "") { $buildArgs += @("--state_eval_npz", $StateEvalNpz) }
if ($Clean) { $buildArgs += "--clean" }

Write-Host "=== Build deterministic wrapper ==="
& $Python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build_ModelFin112_single_model.py failed with exit code $LASTEXITCODE" }

$evalArgs = @(
  ".\evaluate_ModelFin112_unified_5targets.py",
  "--model_dir", $OutputModelDir,
  "--dataset_csv", $DatasetCsv,
  "--output_dir", $OutputEvalDir
)
if ($StateScorecardCsv -ne "") { $evalArgs += @("--state_scorecard_csv", $StateScorecardCsv) }
if ($StateEvalNpz -ne "") { $evalArgs += @("--state_eval_npz", $StateEvalNpz) }

Write-Host "=== Evaluate deterministic wrapper ==="
& $Python @evalArgs
if ($LASTEXITCODE -ne 0) { throw "evaluate_ModelFin112_unified_5targets.py failed with exit code $LASTEXITCODE" }

Write-Host "=== Compact summary ==="
Import-Csv (Join-Path $OutputEvalDir "five_target_compact_summary.csv") | Format-Table -AutoSize
