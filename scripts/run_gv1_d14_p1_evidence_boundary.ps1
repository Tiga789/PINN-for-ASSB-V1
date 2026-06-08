param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$P0Dir = "",
  [string]$OutputDir = "",
  [string]$PythonExe = "python",
  [switch]$ReadmeOnly,
  [switch]$AllowWarn
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($P0Dir)) {
  $p0v2 = Join-Path $CacheRoot "xjtu_d14_p0_freeze_audit_v2"
  $p0v1 = Join-Path $CacheRoot "xjtu_d14_p0_freeze_audit"
  if (Test-Path $p0v2) { $P0Dir = $p0v2 } else { $P0Dir = $p0v1 }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $CacheRoot "xjtu_d14_p1_evidence_boundary"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$log = Join-Path $OutputDir "D14_P1_EVIDENCE_BOUNDARY_console.log"

$argsList = @(
  ".\scripts\gv1_d14_p1_generate_evidence_boundary_report.py",
  "--project-root", $ProjectRoot,
  "--cache-root", $CacheRoot,
  "--p0-dir", $P0Dir,
  "--output-dir", $OutputDir,
  "--config", ".\configs\d14_p1_evidence_boundary_config.json"
)

if ($ReadmeOnly) {
  $argsList += "--readme-only"
}

Write-Host "Running D14-P1 evidence-boundary audit..."
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "P0Dir       = $P0Dir"
Write-Host "OutputDir   = $OutputDir"

& $PythonExe @argsList 2>&1 | Tee-Object -FilePath $log

$verifyArgs = @(
  ".\scripts\gv1_d14_p1_verify_outputs.py",
  "--output-dir", $OutputDir
)
if ($AllowWarn) {
  $verifyArgs += "--allow-warn"
}

& $PythonExe @verifyArgs

Write-Host ""
Write-Host "D14-P1 outputs:"
Get-ChildItem $OutputDir | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
