param(
  [string]$EvalDir = ".\EvalFin_111_seed42_locked_strict30_test70",
  [int]$WorstN = 10,
  [int]$TailN = 20
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$score = Join-Path $EvalDir "five_state_scorecard.csv"
$soh = Join-Path $EvalDir "soh_pred_by_cycle.csv"
$metrics = Join-Path $EvalDir "metrics_soh_by_split.json"
$diag = Join-Path $EvalDir "soh_overdecay_diagnostic.json"
$diagExt = Join-Path $EvalDir "soh_overdecay_diagnostic_external.json"
$guard = Join-Path $EvalDir "state_guard_and_acceptance.json"
$leak = Join-Path $EvalDir "leakage_audit.json"

Write-Host "==== ASSB-111 seed42-locked five-state scorecard ====" -ForegroundColor Cyan
if (Test-Path $score) { Import-Csv $score | Format-Table -AutoSize } else { Write-Warning "Missing $score" }

Write-Host "`n==== SOH metrics by split ====" -ForegroundColor Cyan
if (Test-Path $metrics) { Get-Content $metrics } else { Write-Warning "Missing $metrics" }

Write-Host "`n==== SOH overdecay diagnostics ====" -ForegroundColor Cyan
if (Test-Path $diag) { Get-Content $diag }
elseif (Test-Path $diagExt) { Get-Content $diagExt }
else { Write-Warning "Missing $diag" }

Write-Host "`n==== State guard and acceptance ====" -ForegroundColor Cyan
if (Test-Path $guard) { Get-Content $guard } else { Write-Warning "Missing $guard" }

Write-Host "`n==== Leakage audit ====" -ForegroundColor Cyan
if (Test-Path $leak) { Get-Content $leak } else { Write-Warning "Missing $leak" }

if (Test-Path $soh) {
  Write-Host "`n==== Held-out SOH worst cycles by absolute error ====" -ForegroundColor Cyan
  Import-Csv $soh | Where-Object { $_.split -eq "test" -and $_.SOH_obs -ne "" } |
    ForEach-Object {
      $_ | Add-Member -NotePropertyName abs_err -NotePropertyValue ([math]::Abs([double]$_.SOH_pred - [double]$_.SOH_obs)) -Force
      $_
    } |
    Sort-Object {[double]$_.abs_err} -Descending |
    Select-Object -First $WorstN cycle_id,split,SOH_obs,SOH_pred,abs_err,SOH_struct,SOH_base,soh_floor,remaining_degradable,damage_rate_gated,active_clamp_mask |
    Format-Table -AutoSize

  Write-Host "`n==== Last $TailN SOH cycles ====" -ForegroundColor Cyan
  Import-Csv $soh |
    Select-Object -Last $TailN cycle_id,split,SOH_obs,SOH_pred,SOH_struct,SOH_base,soh_floor,remaining_degradable,damage_rate_gated,active_clamp_mask |
    Format-Table -AutoSize
}
