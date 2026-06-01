param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$AfterPrepare
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S5C preflight ===="
$required = @(
  "scripts\gv1_d11_s5c_prepare_lowtarget_amplitude_repair_commands.py",
  "scripts\gv1_d11_s5c_scorecard_from_predictions.py",
  "scripts\run_gv1_d11_s5c_prepare_commands.ps1",
  "scripts\run_gv1_d11_s5c_collect_scorecard.ps1",
  "scripts\gv1_train_conditioned_pinn.py",
  "gv1\output_transform.py",
  "gv1\losses.py",
  "gv1\trainer.py"
)
foreach ($p in $required) {
  if (-not (Test-Path $p)) { throw "Missing required file: $p" }
  Write-Host "OK: $p"
}

$trainText = Get-Content "scripts\gv1_train_conditioned_pinn.py" -Raw
$outText = Get-Content "gv1\output_transform.py" -Raw
if ($outText -match "enable_voltage_hard_clamp:\s*bool\s*=\s*True") { throw "Mainline output_transform appears to default hard clamp True." }
if ($trainText -notmatch "D9\.5\.1|trend-first|warmup") { Write-Warning "Training script does not visibly contain D9.5.1/trend/warmup marker." }

if ($AfterPrepare) {
  $cmdDir = Join-Path $CacheRoot "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands"
  if (-not (Test-Path $cmdDir)) { throw "Command directory not found: $cmdDir" }
  $bad = Select-String -Path (Join-Path $cmdDir "*.ps1") -Pattern "40000 epoch","epochs 40000","--epochs 40000","time_window_s 200000","--time_window_s 200000","max_time_points 8192","batch_size 2048","enable_voltage_hard_clamp True","metadata_on","battery-8" -ErrorAction SilentlyContinue
  if ($bad) {
    $bad | Format-Table -AutoSize | Out-String | Write-Host
    throw "D11-S5C generated scripts contain forbidden terms."
  }
  $summary = Join-Path $cmdDir "d11_s5c_command_preparation_summary.json"
  if (-not (Test-Path $summary)) { throw "Missing preparation summary: $summary" }
  Write-Host "AfterPrepare checks passed: $cmdDir"
}

Write-Host "D11-S5C preflight PASS"
