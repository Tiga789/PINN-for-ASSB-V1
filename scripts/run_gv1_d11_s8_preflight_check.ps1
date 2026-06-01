param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$AfterPrepare
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

$required = @(
  "scripts\gv1_d11_s8_p2dlike_transport_correction_from_baseline.py",
  "scripts\gv1_d11_s8_prepare_p2dlike_commands.py",
  "scripts\gv1_d11_s8_scorecard_from_predictions.py"
)
foreach ($r in $required) {
  if (!(Test-Path $r)) { throw "Missing required file: $r" }
}

$baselineCandidates = @(
  "$CacheRoot\xjtu_batch134_d11_s7_lowvoltage_escape",
  "$CacheRoot\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair",
  "$CacheRoot\xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis",
  "$CacheRoot\xjtu_batch134_d11_s4_lowtail_correction_smoke",
  "$CacheRoot\xjtu_batch134_d12_s3_metadata_ablation"
)
$found = @()
foreach ($d in $baselineCandidates) {
  if (Test-Path $d) {
    $cnt = (Get-ChildItem $d -Recurse -Filter prediction.npz -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "baseline_d951|metadata_off|d10p1" -and $_.FullName -notmatch "battery[-_]8" } | Measure-Object).Count
    if ($cnt -gt 0) { $found += "$d ($cnt prediction files)" }
  }
}
if ($found.Count -eq 0) {
  throw "No baseline prediction.npz files found for D11-S8. Run D11-S7/S5C baseline first or provide baseline root in prepare script."
}
Write-Host "Baseline prediction sources found:"
$found | ForEach-Object { Write-Host "  $_" }

if ($AfterPrepare) {
  $cmdDir = Join-Path $CacheRoot "xjtu_batch134_d11_s8_p2dlike_transport_correction_commands"
  if (!(Test-Path $cmdDir)) { throw "Command directory not found: $cmdDir" }
  $bad = Select-String -Path "$cmdDir\*.ps1" -Pattern "--epochs\s+40000","--time_window_s\s+200000","--max_time_points\s+8192","--batch_size\s+2048","--metadata_mode\s+on","--enable_voltage_hard_clamp\s+True" -ErrorAction SilentlyContinue
  if ($bad) {
    $bad | Format-Table -AutoSize
    throw "D11-S8 generated commands contain unsafe old parameters."
  }
  Write-Host "AfterPrepare check passed. No unsafe training parameters found."
}
Write-Host "D11-S8 preflight passed."
