param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$SourceRunsRoot = "",
  [string]$OutputRunsRoot = "",
  [string]$ScorecardDir = "",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourceRunsRoot)) {
  $SourceRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks"
}
if ([string]::IsNullOrWhiteSpace($OutputRunsRoot)) {
  $OutputRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1i_s1e_soft_highfallback_6x40ks"
}
if ([string]::IsNullOrWhiteSpace($ScorecardDir)) {
  $ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1i_s1e_soft_highfallback_6x40ks_scorecard"
}

$preflight = [ordered]@{
  stage = "D12-S1I high-region local wrapper 6-profile 40ks"
  ProjectRoot = $ProjectRoot
  CacheRoot = $CacheRoot
  SourceRunsRoot = $SourceRunsRoot
  OutputRunsRoot = $OutputRunsRoot
  ScorecardDir = $ScorecardDir
  PythonExe = $PythonExe
  Training = $false
  MainlineOverwritten = $false
  SourceCandidate = "d12s1e_p2d_low_anchor_soft"
  OutputModes = @("baseline_d951", "d12s1i_high_region_revert_to_baseline", "d12s1i_high_region_delta_budget_20mV", "d12s1i_clip_4p35_plus_high_budget_20mV")
}
$preflight | ConvertTo-Json -Depth 5 | Write-Host

if (!(Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
if (!(Test-Path $PythonExe)) { throw "PythonExe not found: $PythonExe" }
if (!(Test-Path $SourceRunsRoot)) { throw "SourceRunsRoot not found: $SourceRunsRoot" }
$scriptPath = Join-Path $ProjectRoot "scripts\gv1_apply_d12_s1i_high_region_wrapper.py"
if (!(Test-Path $scriptPath)) { throw "Missing script: $scriptPath" }

$argsList = @(
  $scriptPath,
  "--source_runs_root", $SourceRunsRoot,
  "--output_runs_root", $OutputRunsRoot,
  "--scorecard_dir", $ScorecardDir,
  "--baseline_mode", "baseline_d951",
  "--candidate_mode", "d12s1e_p2d_low_anchor_soft"
)
if ($Clean) { $argsList += "--clean" }

& $PythonExe @argsList

Write-Host "D12-S1I 40ks wrapper done. Scorecard: $ScorecardDir"
