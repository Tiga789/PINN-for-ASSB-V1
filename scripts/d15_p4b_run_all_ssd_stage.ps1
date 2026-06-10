param(
  [switch]$AllowOverwrite,
  [int]$Workers = 8,
  [ValidateSet('compressed','uncompressed')]
  [string]$SaveMode = 'uncompressed',
  [string]$StagingRoot = 'C:\XJTU_gv1_cache_staging\d15_p4b_ready18',
  [switch]$ForceCopy,
  [switch]$MirrorFullOutputsToE
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$CacheRoot = 'E:\XJTU battery dataset\_gv1_cache'
$Config = 'configs\d15_p4b_ready18_generation_config.json'
$OriginalManifestCsv = Join-Path $CacheRoot 'xjtu_d15_p4a_fix_replay_readiness_audit\D15_P4A_FIX_P4B_INPUT_MANIFEST.csv'
$PriorJson = 'configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json'

$StageRoot = $StagingRoot
$StageManifestCsv = Join-Path $StageRoot 'D15_P4B_STAGED_P4B_INPUT_MANIFEST.csv'
$StageReportJson = Join-Path $StageRoot 'D15_P4B_SSD_STAGING_REPORT.json'
$OutSoftlabels = Join-Path $StageRoot 'xjtu_softlabels_p2dlite_rg_v1_d15p4b_ready18'
$AuditDir = Join-Path $StageRoot 'xjtu_d15_p4b_ready18_radial_audit'
$ScoreDir = Join-Path $StageRoot 'xjtu_d15_p4b_ready18_scorecard'
$ReviewZipLocal = Join-Path $StageRoot 'xjtu_d15_p4b_results_for_review.zip'
$ReviewZipE = Join-Path $CacheRoot 'xjtu_d15_p4b_results_for_review.zip'
$PreflightJson = Join-Path $ScoreDir 'D15_P4B_PREFLIGHT.json'
$ScorecardJson = Join-Path $ScoreDir 'D15_P4B_FINAL_SCORECARD.json'

if ($AllowOverwrite) {
  Remove-Item -Recurse -Force $OutSoftlabels -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $AuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ScoreDir -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZipLocal -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZipE -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $StageRoot | Out-Null
New-Item -ItemType Directory -Force $ScoreDir | Out-Null

Write-Host '[D15-P4B SSD] 0/7 selftest'
python scripts\d15_p4b_selftest.py

Write-Host '[D15-P4B SSD] 1/7 stage replay profiles to local SSD and rewrite manifest'
$stageArgs = @(
  'scripts\d15_p4b_prepare_ssd_staging.py',
  '--manifest-csv', $OriginalManifestCsv,
  '--staging-root', $StageRoot,
  '--out-manifest-csv', $StageManifestCsv,
  '--out-report-json', $StageReportJson
)
if ($ForceCopy) { $stageArgs += '--force-copy' }
python @stageArgs

Write-Host '[D15-P4B SSD] 2/7 preflight staged manifest'
python scripts\d15_p4b_preflight.py `
  --config $Config `
  --manifest-csv $StageManifestCsv `
  --prior-json $PriorJson `
  --out-json $PreflightJson

Write-Host "[D15-P4B SSD] 3/7 generate ready18 P2Dlite-RG soft labels on SSD; workers=$Workers; save_mode=$SaveMode"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$genArgs = @(
  'scripts\d15_p4b_generate_ready18_rg_softlabels.py',
  '--config', $Config,
  '--manifest-csv', $StageManifestCsv,
  '--prior-json', $PriorJson,
  '--output-dir', $OutSoftlabels,
  '--workers', [string]$Workers,
  '--save-mode', $SaveMode,
  '--allow-overwrite'
)
python @genArgs

Write-Host '[D15-P4B SSD] 4/7 radial audit for SSD-generated ready18 labels'
python scripts\d15_p0_radial_gradient_audit.py `
  --source-dir $OutSoftlabels `
  --prior-json $PriorJson `
  --out-dir $AuditDir `
  --allow-overwrite

Write-Host '[D15-P4B SSD] 5/7 collect scorecard'
python scripts\d15_p4b_collect_scorecard.py `
  --config $Config `
  --preflight-json $PreflightJson `
  --generation-dir $OutSoftlabels `
  --audit-dir $AuditDir `
  --out-json $ScorecardJson

Write-Host '[D15-P4B SSD] 6/7 pack review zip'
python scripts\d15_p4b_pack_review.py `
  --config $Config `
  --preflight-json $PreflightJson `
  --generation-dir $OutSoftlabels `
  --audit-dir $AuditDir `
  --scorecard-json $ScorecardJson `
  --out-zip $ReviewZipLocal

Copy-Item -Force $ReviewZipLocal $ReviewZipE
Write-Host "[D15-P4B SSD] copied review zip to: $ReviewZipE"

if ($MirrorFullOutputsToE) {
  $EOutSoftlabels = Join-Path $CacheRoot 'xjtu_softlabels_p2dlite_rg_v1_d15p4b_ready18'
  $EAuditDir = Join-Path $CacheRoot 'xjtu_d15_p4b_ready18_radial_audit'
  $EScoreDir = Join-Path $CacheRoot 'xjtu_d15_p4b_ready18_scorecard'
  Remove-Item -Recurse -Force $EOutSoftlabels -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $EAuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $EScoreDir -ErrorAction SilentlyContinue
  robocopy $OutSoftlabels $EOutSoftlabels /MIR /R:1 /W:1 | Out-Host
  robocopy $AuditDir $EAuditDir /MIR /R:1 /W:1 | Out-Host
  robocopy $ScoreDir $EScoreDir /MIR /R:1 /W:1 | Out-Host
}

Write-Host '[D15-P4B SSD] DONE'
Write-Host "Local staging root: $StageRoot"
Write-Host "Review zip: $ReviewZipE"
Write-Host "Scorecard: $ScorecardJson"
