param(
  [string]$ProjectRoot = ".",
  [string]$EvalDir = "EvalFin_111_seed42_locked_strict30_test70",
  [string]$ModelDir = "ModelFin_111_seed42_locked",
  [string]$SelectionDir = "EvalFin_111_seed42_locked_selection",
  [switch]$ShowTail
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

function Show-Json([string]$Path, [string]$Title) {
  Write-Host "`n==== $Title ====" -ForegroundColor Cyan
  if (Test-Path $Path) { Get-Content $Path } else { Write-Host "MISSING: $Path" -ForegroundColor Yellow }
}

Write-Host "ASSB-111 seed42-locked final metrics viewer" -ForegroundColor Green
Write-Host "Note: these are final reporting metrics only. Candidate selection must have been completed before this script is used."

Show-Json (Join-Path $SelectionDir "selected_candidate.json") "selected candidate"
Show-Json (Join-Path $SelectionDir "selection_audit.json") "selection audit"
Show-Json (Join-Path $ModelDir "train_summary.json") "train summary"
Show-Json (Join-Path $ModelDir "leakage_audit.json") "leakage audit"

Write-Host "`n==== five_state_scorecard.csv ====" -ForegroundColor Cyan
$score = Join-Path $EvalDir "five_state_scorecard.csv"
if (Test-Path $score) {
  Import-Csv $score | Format-Table variable,source,n,MAE,RMSE,NMAE,NRMSE,R2,corr -AutoSize
  Write-Host "`n---- SOH row ----" -ForegroundColor Cyan
  Import-Csv $score | Where-Object {$_.variable -eq "SOH"} | Format-List
} else {
  Write-Host "MISSING: $score" -ForegroundColor Yellow
}

Show-Json (Join-Path $EvalDir "soh_overdecay_diagnostic.json") "SOH overdecay diagnostic"

if ($ShowTail) {
  $pred = Join-Path $EvalDir "soh_pred_by_cycle.csv"
  Write-Host "`n==== final 25 SOH predictions ====" -ForegroundColor Cyan
  if (Test-Path $pred) {
    Import-Csv $pred | Select-Object -Last 25 cycle_id,split,SOH_obs,SOH_pred,SOH_error | Format-Table -AutoSize
  } else {
    Write-Host "MISSING: $pred" -ForegroundColor Yellow
  }
}
