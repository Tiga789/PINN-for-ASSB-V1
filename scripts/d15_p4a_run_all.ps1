param(
  [switch]$AllowOverwrite,
  [string]$DatasetRoot = "E:\XJTU battery dataset",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_remaining32_replay_audit"
)
$ErrorActionPreference = "Stop"
$ReviewZip = Join-Path $CacheRoot "xjtu_d15_p4a_results_for_review.zip"
Write-Host "[D15-P4A] 0/3 selftest"
python scripts\d15_p4a_selftest.py
Write-Host "[D15-P4A] 1/3 audit remaining cells and replay profiles"
$argsAudit = @("scripts\d15_p4a_audit_remaining32.py", "--config", "configs\d15_p4a_remaining32_audit_config.json", "--dataset-root", $DatasetRoot, "--cache-root", $CacheRoot, "--out-dir", $OutDir)
if ($AllowOverwrite) { $argsAudit += "--allow-overwrite" }
python @argsAudit
Write-Host "[D15-P4A] 2/3 pack review zip"
python scripts\d15_p4a_pack_review.py --out-dir $OutDir --out-zip $ReviewZip
Write-Host "[D15-P4A] DONE"
Write-Host "Review zip: $ReviewZip"
Write-Host "Scorecard: $(Join-Path $OutDir 'D15_P4A_FINAL_SCORECARD.json')"
