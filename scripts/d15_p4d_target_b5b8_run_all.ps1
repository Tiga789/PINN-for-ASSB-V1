param(
  [switch]$AllowOverwrite,
  [int]$CandidateLimit = 0,
  [switch]$BackupOld
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Write-Host '[D15-P4D targeted] 0/3 selftest' -ForegroundColor Cyan
python scripts\d15_p4d_target_b5b8_selftest.py
if ($LASTEXITCODE -ne 0) { throw 'D15-P4D targeted selftest failed' }

Write-Host '[D15-P4D targeted] 1/3 regenerate and audit Batch-5_battery-8 candidates' -ForegroundColor Cyan
$fixArgs = @('scripts\d15_p4d_target_b5b8_fix.py')
if ($AllowOverwrite) { $fixArgs += '--allow-overwrite' }
if ($CandidateLimit -gt 0) { $fixArgs += @('--candidate-limit', [string]$CandidateLimit) }
if ($BackupOld) { $fixArgs += '--backup-old' }
python @fixArgs
$fixExit = $LASTEXITCODE
if ($fixExit -ne 0) {
  Write-Host "[D15-P4D targeted] fix returned nonzero=$fixExit; packing review diagnostics anyway." -ForegroundColor Yellow
}

Write-Host '[D15-P4D targeted] 2/3 pack review' -ForegroundColor Cyan
python scripts\d15_p4d_target_b5b8_pack_review.py

Write-Host '[D15-P4D targeted] DONE' -ForegroundColor Green
Write-Host 'Review zip: E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4d_batch5_battery8_targeted_fix_review.zip'
Write-Host 'Summary:    E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4d_batch5_battery8_targeted_fix\D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_SUMMARY.json'
if ($fixExit -ne 0) { exit $fixExit }
