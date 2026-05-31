param(
  [string]$ProjectRoot = "C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1",
  [string]$PythonExe = "D:/Anaconda/envs/torchgpu/python.exe",
  [string]$PredictionNpz = "E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96/prediction.npz",
  [string]$OutDir = "E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d10_p3_battery8_lightweight_correction",
  [double]$HoldoutFraction = 0.30,
  [switch]$MakePlots
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$argsList = @(
  "scripts/gv1_d10_p3_battery8_lightweight_correction.py",
  "--prediction_npz", $PredictionNpz,
  "--out_dir", $OutDir,
  "--holdout_fraction", $HoldoutFraction
)
if ($MakePlots) { $argsList += "--make_plots" }

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) { throw "D10-P3 battery-8 lightweight correction failed." }

Write-Host "D10-P3 outputs saved to: $OutDir"
Write-Host "Open recommendation:"
Write-Host "  $OutDir/D10_P3_RECOMMENDATION.md"
