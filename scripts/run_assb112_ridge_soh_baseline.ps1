param(
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$OutName = "ModelFin_112_ridgeSOH_g4_v7",
  [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $Root
$dataset = Join-Path $Root "Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv"
$manifest = Join-Path $Root "Data\assb111_seed42locked_repro_c00\split_manifest.json"
$outDir = Join-Path $Root $OutName
$logDir = Join-Path $Root "LogFin_112_ridgeSOH_v7"
New-Item -ItemType Directory -Force $logDir | Out-Null
$logPath = Join-Path $logDir "ridge_g4.log"

if ($Clean -and (Test-Path $outDir)) { Remove-Item $outDir -Recurse -Force }

Write-Host "ASSB-112 v7 ridge SOH baseline"
Write-Host "Purpose = deterministic fast train/val-only SOH baseline; no GPU needed"
Write-Host "Dataset = $dataset"
Write-Host "Manifest= $manifest"
Write-Host "OutDir  = $outDir"
Write-Host "Log     = $logPath"

& $Python (Join-Path $Root "scripts\train_assb112_ridge_soh_head.py") `
  --dataset_csv $dataset `
  --split_manifest_json $manifest `
  --output_model_dir $outDir `
  --feature_mode "g4_all_strict" `
  --candidate_tag "ridgeSOH_g4_v7" `
  --protocol_tag "ASSB112_v7_ridgeSOH_strict30_trainval_only" `
  --no_test_selection `
  2>&1 | Tee-Object -FilePath $logPath

if ($LASTEXITCODE -ne 0) {
  Write-Host "[FAILED] ridge baseline exit=$LASTEXITCODE"
  exit $LASTEXITCODE
}

$finalPath = Join-Path $outDir "metrics_soh_by_split_final_report.json"
if (Test-Path $finalPath) {
  $final = Get-Content $finalPath -Raw | ConvertFrom-Json
  $test = $final.metrics_by_split_after_selection.test
  Write-Host "[RIDGE TEST] R2=$($test.SOH_R2) MAE=$($test.SOH_MAE) RMSE=$($test.SOH_RMSE) BIAS=$($test.SOH_BIAS) corr=$($test.SOH_corr)"
}
Write-Host "[OK] ridge baseline finished."
