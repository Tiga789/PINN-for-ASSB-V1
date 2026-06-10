param(
  [switch]$AllowOverwrite,
  [int]$Workers = 2,
  [ValidateSet('compressed','uncompressed')][string]$SaveMode = 'compressed',
  [switch]$SkipResourceSmoke
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$CacheRoot = 'E:\XJTU battery dataset\_gv1_cache'
$RunDir = Join-Path $CacheRoot 'xjtu_d15_p4c_batch56_replay_completion_scorecard'
$ReplayDir = Join-Path $CacheRoot 'xjtu_batch56_remaining14_replay_profiles_d15p4c'
$AuditDir = Join-Path $CacheRoot 'xjtu_d15_p4c_batch56_remaining14_replay_audit'
$ResourceDir = Join-Path $CacheRoot 'xjtu_d15_p4c_softlabel_resource_smoke'
$ReviewZip = Join-Path $CacheRoot 'xjtu_d15_p4c_results_for_review.zip'

if ($AllowOverwrite) {
  Remove-Item -Recurse -Force $RunDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ReplayDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $AuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ResourceDir -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZip -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force $ReplayDir | Out-Null
New-Item -ItemType Directory -Force $AuditDir | Out-Null
New-Item -ItemType Directory -Force $ResourceDir | Out-Null

$Config = 'configs\d15_p4c_batch56_remaining14_replay_config.json'
$PreJson = Join-Path $RunDir 'D15_P4C_PREFLIGHT_REPORT.json'
$BuildJson = Join-Path $ReplayDir 'D15_P4C_REPLAY_BUILD_REPORT.json'
$BuildManifest = Join-Path $ReplayDir 'xjtu_batch56_remaining14_replay_profile_manifest.csv'
$AuditJson = Join-Path $AuditDir 'D15_P4C_REPLAY_AUDIT_SUMMARY.json'
$ResourceJson = Join-Path $ResourceDir 'D15_P4C_RESOURCE_SMOKE_REPORT.json'
$ScoreJson = Join-Path $RunDir 'D15_P4C_FINAL_SCORECARD.json'

Write-Host '[D15-P4C] 0/6 selftest'
python scripts\d15_p4c_selftest.py

Write-Host '[D15-P4C] 1/6 preflight/discover Batch-5/6 remaining 14 raw files'
python scripts\d15_p4c_preflight.py --config $Config --out-dir $RunDir

Write-Host "[D15-P4C] 2/6 build remaining14 replay profiles; workers=$Workers save_mode=$SaveMode"
python scripts\d15_p4c_build_batch56_replay_profiles.py `
  --config $Config `
  --raw-manifest-csv (Join-Path $RunDir 'D15_P4C_RAW_TARGET_MANIFEST.csv') `
  --out-dir $ReplayDir `
  --workers $Workers `
  --save-mode $SaveMode `
  --allow-overwrite

Write-Host '[D15-P4C] 3/6 audit replay profiles'
python scripts\d15_p4c_audit_replay_profiles.py `
  --config $Config `
  --manifest-csv $BuildManifest `
  --out-dir $AuditDir

if ($SkipResourceSmoke) {
  '{"stage":"D15-P4C resource smoke","overall_status":"SKIPPED","torch_cuda_smoke":{"cuda_available":null}}' | Set-Content -Path $ResourceJson -Encoding UTF8
  Write-Host '[D15-P4C] 4/6 resource smoke skipped'
} else {
  Write-Host '[D15-P4C] 4/6 resource smoke for future P4D soft-label generation'
  python scripts\d15_p4c_softlabel_resource_smoke.py `
    --config $Config `
    --manifest-csv $BuildManifest `
    --out-dir $ResourceDir
}

Write-Host '[D15-P4C] 5/6 collect scorecard'
python scripts\d15_p4c_collect_scorecard.py `
  --config $Config `
  --preflight-json $PreJson `
  --build-json $BuildJson `
  --audit-json $AuditJson `
  --resource-json $ResourceJson `
  --out-json $ScoreJson

Write-Host '[D15-P4C] 6/6 pack review zip'
python scripts\d15_p4c_pack_review.py `
  --out-zip $ReviewZip `
  --paths $RunDir $ReplayDir $AuditDir $ResourceDir

Write-Host '[D15-P4C] DONE'
Write-Host "Review zip: $ReviewZip"
Write-Host "Scorecard: $ScoreJson"
