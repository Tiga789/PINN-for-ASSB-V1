param(
  [switch]$AllowOverwrite,
  [int]$Workers = 2,
  [ValidateSet('compressed','uncompressed')]
  [string]$SaveMode = 'uncompressed'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$CacheRoot = 'E:\XJTU battery dataset\_gv1_cache'
$Config = 'configs\d15_p4b_ready18_generation_config.json'
$ManifestCsv = Join-Path $CacheRoot 'xjtu_d15_p4a_fix_replay_readiness_audit\D15_P4A_FIX_P4B_INPUT_MANIFEST.csv'
$PriorJson = 'configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json'
$OutSoftlabels = Join-Path $CacheRoot 'xjtu_softlabels_p2dlite_rg_v1_d15p4b_ready18'
$AuditDir = Join-Path $CacheRoot 'xjtu_d15_p4b_ready18_radial_audit'
$ScoreDir = Join-Path $CacheRoot 'xjtu_d15_p4b_ready18_scorecard'
$ReviewZip = Join-Path $CacheRoot 'xjtu_d15_p4b_results_for_review.zip'
$PreflightJson = Join-Path $ScoreDir 'D15_P4B_PREFLIGHT.json'
$ScorecardJson = Join-Path $ScoreDir 'D15_P4B_FINAL_SCORECARD.json'

if ((Test-Path $OutSoftlabels) -and -not $AllowOverwrite) {
  if ((Get-ChildItem $OutSoftlabels -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    throw "Output directory exists and is not empty: $OutSoftlabels. Use -AllowOverwrite for deliberate rerun."
  }
}
if ((Test-Path $AuditDir) -and -not $AllowOverwrite) {
  if ((Get-ChildItem $AuditDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    throw "Audit directory exists and is not empty: $AuditDir. Use -AllowOverwrite for deliberate rerun."
  }
}
if ((Test-Path $ScoreDir) -and -not $AllowOverwrite) {
  if ((Get-ChildItem $ScoreDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    throw "Scorecard directory exists and is not empty: $ScoreDir. Use -AllowOverwrite for deliberate rerun."
  }
}

if ($AllowOverwrite) {
  Remove-Item -Recurse -Force $OutSoftlabels -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $AuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ScoreDir -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZip -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $ScoreDir | Out-Null

Write-Host '[D15-P4B] 0/6 selftest'
python scripts\d15_p4b_selftest.py

Write-Host '[D15-P4B] 1/6 preflight'
python scripts\d15_p4b_preflight.py `
  --config $Config `
  --manifest-csv $ManifestCsv `
  --prior-json $PriorJson `
  --out-json $PreflightJson

Write-Host '[D15-P4B] 2/6 generate ready18 P2Dlite-RG soft labels'
$genArgs = @(
  'scripts\d15_p4b_generate_ready18_rg_softlabels.py',
  '--config', $Config,
  '--manifest-csv', $ManifestCsv,
  '--prior-json', $PriorJson,
  '--output-dir', $OutSoftlabels,
  '--workers', [string]$Workers,
  '--save-mode', $SaveMode
)
if ($AllowOverwrite) { $genArgs += '--allow-overwrite' }
python @genArgs

Write-Host '[D15-P4B] 3/6 radial audit for ready18 labels'
$auditArgs = @(
  'scripts\d15_p0_radial_gradient_audit.py',
  '--source-dir', $OutSoftlabels,
  '--prior-json', $PriorJson,
  '--out-dir', $AuditDir
)
if ($AllowOverwrite) { $auditArgs += '--allow-overwrite' }
python @auditArgs

Write-Host '[D15-P4B] 4/6 collect scorecard'
python scripts\d15_p4b_collect_scorecard.py `
  --config $Config `
  --preflight-json $PreflightJson `
  --generation-dir $OutSoftlabels `
  --audit-dir $AuditDir `
  --out-json $ScorecardJson

Write-Host '[D15-P4B] 5/6 pack review zip'
python scripts\d15_p4b_pack_review.py `
  --config $Config `
  --preflight-json $PreflightJson `
  --generation-dir $OutSoftlabels `
  --audit-dir $AuditDir `
  --scorecard-json $ScorecardJson `
  --out-zip $ReviewZip

Write-Host '[D15-P4B] DONE'
Write-Host "Review zip: $ReviewZip"
Write-Host "Scorecard: $ScorecardJson"
