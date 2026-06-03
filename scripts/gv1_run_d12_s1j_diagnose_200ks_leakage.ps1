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
  $SourceRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_6x200ks"
}
if ([string]::IsNullOrWhiteSpace($OutputRunsRoot)) {
  $OutputRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1j_200ks_normal_rest_leakage_wrapper"
}
if ([string]::IsNullOrWhiteSpace($ScorecardDir)) {
  $ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1j_200ks_normal_rest_leakage_scorecard"
}

$ScriptPath = Join-Path $ProjectRoot "scripts\gv1_d12_s1j_diagnose_200ks_leakage.py"

$preflight = [ordered]@{
  stage = "D12-S1J 200ks normal/rest leakage diagnostic preflight"
  ProjectRoot = $ProjectRoot
  CacheRoot = $CacheRoot
  SourceRunsRoot = $SourceRunsRoot
  OutputRunsRoot = $OutputRunsRoot
  ScorecardDir = $ScorecardDir
  PythonExe = $PythonExe
  ScriptPath = $ScriptPath
  Training = "none"
  Reads = "D12-S1E baseline_d951 + d12s1e_p2d_low_anchor_soft 200ks predictions"
  Purpose = "Diagnose and locally wrap 200ks normal/rest/global leakage after S1I high fallback"
}
$preflight | ConvertTo-Json -Depth 5 | Write-Host

if (!(Test-Path $PythonExe)) { throw "PythonExe not found: $PythonExe" }
if (!(Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
if (!(Test-Path $ScriptPath)) { throw "D12-S1J script not found: $ScriptPath" }
if (!(Test-Path $SourceRunsRoot)) { throw "SourceRunsRoot not found: $SourceRunsRoot. Run S1E-soft+baseline 200ks first." }

$predCount = (Get-ChildItem $SourceRunsRoot -Recurse -Filter prediction.npz | Measure-Object).Count
if ($predCount -lt 12) {
  throw "Expected at least 12 prediction.npz files under $SourceRunsRoot, found $predCount. Need 6 profiles x 2 modes."
}

$argsList = @(
  $ScriptPath,
  "--source_runs_root", $SourceRunsRoot,
  "--output_runs_root", $OutputRunsRoot,
  "--output_dir", $ScorecardDir,
  "--baseline_mode", "baseline_d951",
  "--candidate_mode", "d12s1e_p2d_low_anchor_soft"
)
if ($Clean) { $argsList += "--clean" }

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) { throw "D12-S1J diagnostic failed with exit code $LASTEXITCODE" }

Write-Host "D12-S1J done. Scorecard: $ScorecardDir"
