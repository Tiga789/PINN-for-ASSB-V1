param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$AfterPrepare
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

$required = @(
  "scripts\gv1_d11_s7_apply_lowvoltage_escape_patch.py",
  "scripts\gv1_d11_s7_prepare_lowvoltage_escape_commands.py",
  "scripts\gv1_d11_s7_scorecard_from_predictions.py",
  "scripts\run_gv1_d11_s7_apply_patch.ps1",
  "scripts\run_gv1_d11_s7_prepare_commands.ps1",
  "scripts\run_gv1_d11_s7_collect_scorecard.ps1",
  "gv1\output_transform.py",
  "scripts\gv1_train_conditioned_pinn.py"
)
foreach ($r in $required) {
  if (!(Test-Path $r)) { throw "Missing required path: $r" }
}

$txtTrain = Get-Content "scripts\gv1_train_conditioned_pinn.py" -Raw
$txtOut = Get-Content "gv1\output_transform.py" -Raw

if ($txtTrain -notmatch "D9\.5\.1|trend-first|rare-regime") {
  Write-Warning "Training script does not clearly show D9.5.1 mainline markers. Continue carefully."
}
if ($txtOut -match "enable_voltage_hard_clamp:\s*bool\s*=\s*True") {
  throw "Hard clamp appears enabled by default in output_transform.py"
}

# Before patch this can be absent. After patch/apply it should be present.
if (($txtTrain -notmatch "enable_low_voltage_escape") -or ($txtOut -notmatch "enable_low_voltage_escape")) {
  Write-Warning "D11-S7 low-voltage escape patch not detected yet. Run scripts\run_gv1_d11_s7_apply_patch.ps1 before prepare/execute."
}

if ($AfterPrepare) {
  $cmdDir = Join-Path $CacheRoot "xjtu_batch134_d11_s7_lowvoltage_escape_commands"
  if (!(Test-Path $cmdDir)) { throw "Generated command directory missing: $cmdDir" }
  $bad = Select-String -Path (Join-Path $cmdDir "*.ps1") -Pattern "epochs 40000","--epochs 40000","time_window_s 200000","--time_window_s 200000","max_time_points 8192","batch_size 2048","enable_voltage_hard_clamp True","metadata_on" -ErrorAction SilentlyContinue
  if ($bad) {
    $bad | Format-Table -AutoSize
    throw "D11-S7 generated commands contain unsafe old parameters."
  }
}

Write-Host "D11-S7 preflight passed."
