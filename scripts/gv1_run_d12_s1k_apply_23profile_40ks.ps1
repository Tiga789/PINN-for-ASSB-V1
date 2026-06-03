param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$SourceRunsRoot = "",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SourceRunsRoot)) {
  $SourceRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks"
}
$OutputRunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1k_two_candidate_23x40ks_wrapper"
$ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1k_two_candidate_23x40ks_scorecard"

$Preflight = [ordered]@{
  stage = "D12-S1K two-candidate 23-profile 40ks wrapper confirmation"
  ProjectRoot = $ProjectRoot
  CacheRoot = $CacheRoot
  SourceRunsRoot = $SourceRunsRoot
  OutputRunsRoot = $OutputRunsRoot
  ScorecardDir = $ScorecardDir
  PythonExe = $PythonExe
  CandidateModes = @("d12s1k_low_only_revert_nonlow_to_baseline", "d12s1k_low_plus_transition_fade_to_baseline")
  NoTraining = $true
  Battery8ExpectedExcluded = $true
}
$Preflight | ConvertTo-Json -Depth 6 | Write-Host

if (-not (Test-Path -LiteralPath $SourceRunsRoot -PathType Container)) {
  throw "SourceRunsRoot not found: $SourceRunsRoot. Generate S1E-soft + baseline source predictions first with scripts\gv1_run_d12_s1k_generate_s1e_source_23profile_40ks_fast_parallel.ps1, or pass -SourceRunsRoot to an existing source prediction root."
}

$baseCount = (Get-ChildItem -LiteralPath $SourceRunsRoot -Recurse -Filter prediction.npz -File | Where-Object { $_.FullName -like "*baseline_d951*" } | Measure-Object).Count
$softCount = (Get-ChildItem -LiteralPath $SourceRunsRoot -Recurse -Filter prediction.npz -File | Where-Object { $_.FullName -like "*d12s1e_p2d_low_anchor_soft*" } | Measure-Object).Count
Write-Host "Source prediction count: baseline=$baseCount ; s1e_soft=$softCount"
if ($baseCount -lt 1 -or $softCount -lt 1) {
  throw "SourceRunsRoot lacks baseline_d951 or d12s1e_p2d_low_anchor_soft prediction.npz files."
}

& $PythonExe -m py_compile (Join-Path $ProjectRoot "scripts\gv1_apply_d12_s1k_two_candidate_wrapper.py")
if ($LASTEXITCODE -ne 0) { throw "py_compile failed for S1K wrapper" }

$cmd = @(
  (Join-Path $ProjectRoot "scripts\gv1_apply_d12_s1k_two_candidate_wrapper.py"),
  "--source_runs_root", $SourceRunsRoot,
  "--output_runs_root", $OutputRunsRoot,
  "--output_dir", $ScorecardDir,
  "--baseline_mode", "baseline_d951",
  "--candidate_mode", "d12s1e_p2d_low_anchor_soft"
)
if ($Clean) { $cmd += "--clean" }

& $PythonExe @cmd
if ($LASTEXITCODE -ne 0) { throw "D12-S1K 23-profile 40ks wrapper failed" }
Write-Host "D12-S1K 23-profile 40ks done. Scorecard: $ScorecardDir"
