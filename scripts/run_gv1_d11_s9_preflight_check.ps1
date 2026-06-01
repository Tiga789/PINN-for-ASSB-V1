param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot
Write-Host "D11-S9 preflight: trainable localized P2D-like correction head (post-hoc diagnostic)."
$required = @(
  "scripts\gv1_d11_s9_trainable_p2dlike_head_from_baseline.py",
  "scripts\gv1_d11_s9_scorecard_from_predictions.py",
  "scripts\run_gv1_d11_s9_trainable_p2dlike_correction.ps1",
  "scripts\run_gv1_d11_s9_collect_scorecard.ps1"
)
foreach ($rel in $required) {
  if (!(Test-Path (Join-Path $ProjectRoot $rel))) { throw "Missing required file: $rel" }
}
$roots = @(
  "$CacheRoot\xjtu_batch134_d11_s8_p2dlike_transport_correction",
  "$CacheRoot\xjtu_batch134_d11_s7_lowvoltage_escape",
  "$CacheRoot\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair"
)
$found = $false
foreach ($r in $roots) {
  if (Test-Path $r) {
    $n = (Get-ChildItem $r -Recurse -Filter prediction.npz -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "Candidate baseline root: $r ; prediction.npz count=$n"
    if ($n -gt 0) { $found = $true }
  }
}
if (-not $found) { throw "No prior prediction.npz root found for D11-S9. Run D11-S8/S7/S5C first." }
Write-Host "D11-S9 preflight PASS."
