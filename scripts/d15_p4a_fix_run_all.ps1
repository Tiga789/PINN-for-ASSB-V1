param(
  [switch]$AllowOverwrite
)
$ErrorActionPreference = "Stop"
Write-Host "[D15-P4A-fix] 0/3 selftest" -ForegroundColor Cyan
python scripts\d15_p4a_fix_selftest.py
Write-Host "[D15-P4A-fix] 1/3 fixed canonical replay readiness audit" -ForegroundColor Cyan
$overwriteArg = @()
if ($AllowOverwrite) { $overwriteArg += "--allow-overwrite" }
python scripts\d15_p4a_fix_replay_readiness_audit.py --config configs\d15_p4a_fix_replay_readiness_config.json @overwriteArg
Write-Host "[D15-P4A-fix] 2/3 pack review zip" -ForegroundColor Cyan
python scripts\d15_p4a_fix_pack_review.py --config configs\d15_p4a_fix_replay_readiness_config.json
Write-Host "[D15-P4A-fix] DONE" -ForegroundColor Green
Write-Host "Review zip: E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_fix_results_for_review.zip" -ForegroundColor Green
Write-Host "Scorecard:  E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_fix_replay_readiness_audit\D15_P4A_FIX_FINAL_SCORECARD.json" -ForegroundColor Green
