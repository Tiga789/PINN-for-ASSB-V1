param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$RawRoot = "E:\XJTU battery dataset",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$AllowOverwrite,
  [switch]$Quick,
  [switch]$SkipNN
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$Config = "configs\d15_p3_batch2_applicability_config.json"
$NNConfig = "configs\d15_p3_batch2_nn_smoke_config.json"
$PriorJson = "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json"

$ReplayDir = Join-Path $CacheRoot "xjtu_batch2_replay_profiles_d15p3"
$SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p3_batch2_3cell"
$AuditDir = Join-Path $CacheRoot "xjtu_d15_p3_batch2_3cell_radial_audit"
$NNRunDir = Join-Path $CacheRoot "xjtu_d15_p3_batch2_3cell_rg_nn_smoke"
$ScorecardDir = Join-Path $CacheRoot "xjtu_d15_p3_batch2_applicability_scorecard"
$ReviewZip = Join-Path $CacheRoot "xjtu_d15_p3_results_for_review.zip"

if ($AllowOverwrite) {
  Remove-Item -Recurse -Force $ReplayDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $SoftlabelDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $AuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $NNRunDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ScorecardDir -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZip -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force $ReplayDir | Out-Null
New-Item -ItemType Directory -Force $ScorecardDir | Out-Null

Write-Host "[D15-P3] 0/9 selftest" -ForegroundColor Cyan
python scripts\d15_p3_selftest_batch2.py

$PreflightJson = Join-Path $ScorecardDir "D15_P3A_PREFLIGHT.json"
Write-Host "[D15-P3] 1/9 preflight" -ForegroundColor Cyan
python scripts\d15_p3a_preflight_batch2.py `
  --config $Config `
  --raw-root $RawRoot `
  --cache-root $CacheRoot `
  --out-json $PreflightJson

Write-Host "[D15-P3] 2/9 discover Batch-2 raw files" -ForegroundColor Cyan
python scripts\d15_p3a_discover_batch2.py `
  --config $Config `
  --raw-root $RawRoot `
  --out-dir $ReplayDir

Write-Host "[D15-P3] 3/9 build Batch-2 replay profiles" -ForegroundColor Cyan
$buildArgs = @(
  "scripts\d15_p3a_build_batch2_replay_profiles.py",
  "--config", $Config,
  "--raw-root", $RawRoot,
  "--out-dir", $ReplayDir
)
if ($AllowOverwrite) { $buildArgs += "--allow-overwrite" }
python @buildArgs

$ReplayManifest = Join-Path $ReplayDir "xjtu_batch2_replay_profile_manifest.csv"
Write-Host "[D15-P3] 4/9 select 3 representative Batch-2 cells" -ForegroundColor Cyan
python scripts\d15_p3b_select_batch2_representatives.py `
  --config $Config `
  --manifest-csv $ReplayManifest `
  --out-dir $ScorecardDir

$RepManifest = Join-Path $ScorecardDir "D15_P3B_BATCH2_REPRESENTATIVE_MANIFEST.csv"
Write-Host "[D15-P3] 5/9 generate 3-cell Batch-2 P2Dlite-RG soft labels" -ForegroundColor Cyan
$genArgs = @(
  "scripts\d15_p3c_generate_batch2_rg_softlabels.py",
  "--config", $Config,
  "--representative-manifest-csv", $RepManifest,
  "--prior-json", $PriorJson,
  "--output-dir", $SoftlabelDir
)
if ($AllowOverwrite) { $genArgs += "--allow-overwrite" }
python @genArgs

Write-Host "[D15-P3] 6/9 radial audit for Batch-2 generated RG labels" -ForegroundColor Cyan
python scripts\d15_p0_radial_gradient_audit.py `
  --source-dir $SoftlabelDir `
  --prior-json $PriorJson `
  --out-dir $AuditDir

$NNScorecardJson = Join-Path $NNRunDir "D15_P1_FINAL_SCORECARD.json"
if (-not $SkipNN) {
  Write-Host "[D15-P3] 7/9 Batch-2 3-cell NN smoke via D15-P1 trainer" -ForegroundColor Cyan
  $EffectiveNNConfig = $NNConfig
  if ($Quick) {
    $QuickConfig = Join-Path $ScorecardDir "d15_p3_batch2_nn_smoke_config_QUICK.json"
    $obj = Get-Content $NNConfig -Raw | ConvertFrom-Json
    $obj.training.epochs = 120
    $obj.data.max_time_points_per_profile_train = 2048
    $obj.data.max_time_points_per_profile_val = 512
    $obj | ConvertTo-Json -Depth 20 | Set-Content $QuickConfig -Encoding UTF8
    $EffectiveNNConfig = $QuickConfig
  }
  $trainArgs = @(
    "scripts\d15_p1_train_rg_closedset_nn_smoke.py",
    "--softlabel-dir", $SoftlabelDir,
    "--out-dir", $NNRunDir,
    "--config", $EffectiveNNConfig
  )
  if ($AllowOverwrite) { $trainArgs += "--allow-overwrite" }
  python @trainArgs
  python scripts\d15_p1_eval_rg_closedset_nn_smoke.py `
    --softlabel-dir $SoftlabelDir `
    --model-dir $NNRunDir `
    --out-dir (Join-Path $NNRunDir "eval_full_profiles") `
    --config $EffectiveNNConfig `
    --allow-overwrite
  python scripts\d15_p1_collect_scorecard.py `
    --run-dir $NNRunDir `
    --eval-dir (Join-Path $NNRunDir "eval_full_profiles") `
    --out-json $NNScorecardJson
} else {
  Write-Host "[D15-P3] 7/9 NN smoke skipped by -SkipNN" -ForegroundColor Yellow
  $NNScorecardJson = ""
}

Write-Host "[D15-P3] 8/9 collect scorecard" -ForegroundColor Cyan
$CollectArgs = @(
  "scripts\d15_p3_collect_scorecard.py",
  "--scorecard-dir", $ScorecardDir,
  "--preflight-json", $PreflightJson,
  "--discovery-json", (Join-Path $ReplayDir "D15_P3A_BATCH2_DISCOVERY_REPORT.json"),
  "--replay-json", (Join-Path $ReplayDir "D15_P3A_BATCH2_REPLAY_BUILD_REPORT.json"),
  "--selection-json", (Join-Path $ScorecardDir "D15_P3B_BATCH2_SELECTION_REPORT.json"),
  "--generation-json", (Join-Path $SoftlabelDir "D15_P3C_BATCH2_RG_GENERATION_REPORT.json"),
  "--radial-audit-json", (Join-Path $AuditDir "radial_gradient_audit_summary.json"),
  "--out-json", (Join-Path $ScorecardDir "D15_P3_FINAL_SCORECARD.json")
)
if (-not $SkipNN) { $CollectArgs += @("--nn-scorecard-json", $NNScorecardJson) }
python @CollectArgs

Write-Host "[D15-P3] 9/9 pack review zip" -ForegroundColor Cyan
python scripts\d15_p3_pack_review.py `
  --cache-root $CacheRoot `
  --out-zip $ReviewZip

Write-Host "[D15-P3] DONE" -ForegroundColor Green
Write-Host "Review zip: $ReviewZip" -ForegroundColor Green
Write-Host "Scorecard: $(Join-Path $ScorecardDir 'D15_P3_FINAL_SCORECARD.json')" -ForegroundColor Green
