param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$TargetProfile = "Batch-1_2C_battery-8",
  [string]$OutputDir = "",
  [switch]$MakePlots,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $CacheRoot "xjtu_batch134_d12_battery8_data_diagnosis"
}

$scriptPath = Join-Path $ProjectRoot "scripts\gv1_d12_battery8_data_diagnosis.py"
if (!(Test-Path $scriptPath)) {
  throw "Missing script: $scriptPath"
}

$profilesRoot = Join-Path $CacheRoot "xjtu_batch134_replay_profiles"
if (!(Test-Path $profilesRoot)) {
  throw "Missing replay profiles root: $profilesRoot"
}

if ($Clean -and (Test-Path $OutputDir)) {
  Remove-Item $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$preflight = [ordered]@{
  stage = "D12 battery-8 data/profile anomaly diagnosis"
  ProjectRoot = $ProjectRoot
  CacheRoot = $CacheRoot
  ProfilesRoot = $profilesRoot
  PythonExe = $PythonExe
  TargetProfile = $TargetProfile
  OutputDir = $OutputDir
  MakePlots = [bool]$MakePlots
  Training = "disabled"
  MainlineOverwritten = $false
}
$preflight | ConvertTo-Json -Depth 6 | Write-Host

$argsList = @(
  $scriptPath,
  "--cache-root", $CacheRoot,
  "--profiles-root", $profilesRoot,
  "--target-profile", $TargetProfile,
  "--output-dir", $OutputDir
)
if ($MakePlots) {
  $argsList += "--make-plots"
}

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
  throw "D12 battery-8 diagnosis failed with exit code $LASTEXITCODE"
}

Write-Host "D12 battery-8 diagnosis done. OutputDir: $OutputDir"
Write-Host "Open:"
Write-Host "  $OutputDir\D12_B8_diagnostic_summary.json"
Write-Host "  $OutputDir\D12_B8_RECOMMENDATION.md"
