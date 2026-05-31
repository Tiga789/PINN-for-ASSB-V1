param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$ProjectRoot = "",
  [string]$OutDir = "",
  [string]$ProfileManifest = "",
  [string]$CycleManifest = "",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$TargetBatchId = "Batch-1",
  [string]$TargetBatteryId = "battery-8",
  [string]$TargetProtocol = "2C",
  [switch]$MakePlots,
  [switch]$Strict
)

if ($ProjectRoot -eq "") {
  $ProjectRoot = (Get-Location).Path
}

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

$PythonArgs = @(
  "scripts\gv1_d11_b_regime_feature_distance_audit.py",
  "--cache_root", $CacheRoot,
  "--project_root", $ProjectRoot,
  "--target_batch_id", $TargetBatchId,
  "--target_battery_id", $TargetBatteryId,
  "--target_protocol", $TargetProtocol
)

if ($OutDir -ne "") {
  $PythonArgs += @("--out_dir", $OutDir)
}
if ($ProfileManifest -ne "") {
  $PythonArgs += @("--profile_manifest", $ProfileManifest)
}
if ($CycleManifest -ne "") {
  $PythonArgs += @("--cycle_manifest", $CycleManifest)
}
if ($MakePlots) {
  $PythonArgs += @("--make_plots")
}
if ($Strict) {
  $PythonArgs += @("--strict")
}

& $PythonExe @PythonArgs
if ($LASTEXITCODE -ne 0) {
  throw "D11-B regime feature distance audit failed."
}
