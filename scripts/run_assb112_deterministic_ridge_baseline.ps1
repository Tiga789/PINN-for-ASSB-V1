param(
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "",
  [string]$Dataset = ".\Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv",
  [string]$Manifest = ".\Data\assb111_seed42locked_repro_c00\split_manifest.json",
  [string]$OutDir = ".\ModelFin_112_deterministicSOH_ridge_g4",
  [string]$EvalDir = ".\EvalFin_112_deterministicSOH_ridge_g4",
  [string]$FeatureMode = "g4_all_strict",
  [ValidateSet("auto", "cuda", "cpu")]
  [string]$Device = "cuda",
  [ValidateSet("float64", "float32")]
  [string]$DType = "float64",
  [double]$GpuReserveGB = 2.0,
  [int]$GpuWorkRepeats = 4,
  [int]$TopKAverage = 3,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($Python)) {
  if (Test-Path "D:\Anaconda\envs\torchgpu\python.exe") {
    $Python = "D:\Anaconda\envs\torchgpu\python.exe"
  } elseif ($env:ASSB_PYTHON) {
    $Python = $env:ASSB_PYTHON
  } else {
    $Python = "python"
  }
}

if ($Clean) {
  Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $EvalDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "ASSB-112 deterministic ridge SOH baseline"
Write-Host "Root         = $Root"
Write-Host "Python       = $Python"
Write-Host "Dataset      = $Dataset"
Write-Host "Manifest     = $Manifest"
Write-Host "OutDir       = $OutDir"
Write-Host "FeatureMode  = $FeatureMode"
Write-Host "Device       = $Device"
Write-Host "DType        = $DType"
Write-Host "GpuReserveGB = $GpuReserveGB"
Write-Host "GpuWorkRepeats = $GpuWorkRepeats"

& $Python ".\scripts\train_assb112_deterministic_soh_baseline.py" `
  --dataset_csv $Dataset `
  --split_manifest_json $Manifest `
  --output_model_dir $OutDir `
  --feature_mode $FeatureMode `
  --device $Device `
  --dtype $DType `
  --gpu_reserve_gb $GpuReserveGB `
  --gpu_work_repeats $GpuWorkRepeats `
  --topk_average $TopKAverage `
  --selection_mode "visible_score" `
  --clip_soh_min 0.0 `
  --clip_soh_max 1.05 `
  --no_test_selection

if ($LASTEXITCODE -ne 0) {
  throw "deterministic ridge training failed with exit code $LASTEXITCODE"
}

& $Python ".\scripts\summarize_assb112_deterministic_baseline.py" `
  --model_dir $OutDir `
  --output_dir $EvalDir

Write-Host "[OK] deterministic ridge baseline finished."
Write-Host "Final report: $OutDir\metrics_soh_by_split_final_report.json"
Write-Host "Scorecard:    $EvalDir\deterministic_soh_scorecard.csv"
