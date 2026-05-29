# GV1 D9.7 battery-8 outlier/regime diagnosis plotter.
# This script only generates plots and tables. It does not train or modify models.

$ErrorActionPreference = "Stop"

$python = "D:\Anaconda\envs\torchgpu\python.exe"
$summaryJson = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\d97_battery8_diagnosis_summary.json"
$outputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots"

if (-not (Test-Path $python)) {
  throw "Python executable not found: $python"
}
if (-not (Test-Path $summaryJson)) {
  throw "Diagnosis summary JSON not found: $summaryJson"
}
if (-not (Test-Path ".\scripts\gv1_plot_battery8_regime_d97.py")) {
  throw "Missing .\scripts\gv1_plot_battery8_regime_d97.py. Please add the D9.7 plot patch files to the project root."
}

& $python .\scripts\gv1_plot_battery8_regime_d97.py `
  --summary_json $summaryJson `
  --output_dir $outputDir `
  --max_runs 20

Write-Host ""
Write-Host "D9.7 plots saved to: $outputDir"
Write-Host "Open manifest:"
Write-Host "  $outputDir\d97_plot_manifest.json"
