
param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$P0Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2",
  [string]$P1Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard",
  [string]$D10P1Dir = "",
  [string]$D12S1K200ksDir = "",
  [string]$D12S1K40ksDir = "",
  [string]$D13SegmentDir = "",
  [switch]$StrictEvidence,
  [switch]$AllowWarn
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$LogPath = Join-Path $OutputDir "D14_P2_GENERALIZATION_SCORECARD_console.log"

$py = "python"
$argsList = @(
  ".\scripts\gv1_d14_p2_build_generalization_scorecard.py",
  "--project-root", $ProjectRoot,
  "--cache-root", $CacheRoot,
  "--p0-dir", $P0Dir,
  "--p1-dir", $P1Dir,
  "--output-dir", $OutputDir,
  "--allow-p0-p1-warn"
)
if ($D10P1Dir -ne "") { $argsList += @("--d10-p1-dir", $D10P1Dir) }
if ($D12S1K200ksDir -ne "") { $argsList += @("--d12-s1k-200ks-dir", $D12S1K200ksDir) }
if ($D12S1K40ksDir -ne "") { $argsList += @("--d12-s1k-40ks-dir", $D12S1K40ksDir) }
if ($D13SegmentDir -ne "") { $argsList += @("--d13-segment-dir", $D13SegmentDir) }
if ($StrictEvidence) { $argsList += "--strict-evidence" }

Write-Host "[D14-P2] Running scorecard builder..."
& $py @argsList 2>&1 | Tee-Object -FilePath $LogPath
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host "[D14-P2] Builder returned exit code $code. Check $LogPath"
  if ($StrictEvidence) { exit $code }
}

$verifyArgs = @(
  ".\scripts\gv1_d14_p2_verify_outputs.py",
  "--output-dir", $OutputDir,
  "--allow-warn"
)
if ($StrictEvidence) { $verifyArgs += "--strict" }
Write-Host "[D14-P2] Verifying outputs..."
& $py @verifyArgs 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "D14_P2_VERIFY_console.log")
$verifyCode = $LASTEXITCODE
if ($verifyCode -ne 0) { exit $verifyCode }

Write-Host "[D14-P2] Done. OutputDir=$OutputDir"
