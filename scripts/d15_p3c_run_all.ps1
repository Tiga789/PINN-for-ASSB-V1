param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$AllowOverwrite,
  [switch]$Quick,
  [switch]$SkipNN
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$Config = "configs\d15_p3c_batch2_15cell_applicability_config.json"
$NNConfig = "configs\d15_p3c_batch2_15cell_nn_config.json"
$BoundaryConfig = "configs\d15_p3c_boundary_repair_config.json"
$PriorJson = "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json"

$ReplayDir = Join-Path $CacheRoot "xjtu_batch2_replay_profiles_d15p3"
$ReplayManifest = Join-Path $ReplayDir "xjtu_batch2_replay_profile_manifest.csv"
$SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p3c_batch2_15cell"
$AuditDir = Join-Path $CacheRoot "xjtu_d15_p3c_batch2_15cell_radial_audit"
$NNRunDir = Join-Path $CacheRoot "xjtu_d15_p3c_batch2_15cell_rg_nn_benchmark"
$BoundaryDir = Join-Path $CacheRoot "xjtu_d15_p3c_batch2_15cell_boundary_projection_repair"
$ScorecardDir = Join-Path $CacheRoot "xjtu_d15_p3c_batch2_15cell_applicability_scorecard"
$ReviewZip = Join-Path $CacheRoot "xjtu_d15_p3c_results_for_review.zip"
$All15Manifest = Join-Path $ScorecardDir "D15_P3C_BATCH2_ALL15_MANIFEST.csv"
$All15ManifestJson = Join-Path $ScorecardDir "D15_P3C_BATCH2_ALL15_MANIFEST_REPORT.json"
$FinalScorecard = Join-Path $ScorecardDir "D15_P3C_FINAL_SCORECARD.json"

if ($AllowOverwrite) {
  Remove-Item -Recurse -Force $SoftlabelDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $AuditDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $NNRunDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $BoundaryDir -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force $ScorecardDir -ErrorAction SilentlyContinue
  Remove-Item -Force $ReviewZip -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force $ScorecardDir | Out-Null

Write-Host "[D15-P3C] 0/8 selftest" -ForegroundColor Cyan
python scripts\d15_p3c_selftest_15cell.py

Write-Host "[D15-P3C] 1/8 verify existing Batch-2 replay manifest and select all 15" -ForegroundColor Cyan
if (-not (Test-Path $ReplayManifest)) {
  throw "Missing Batch-2 replay manifest: $ReplayManifest. Run D15-P3 first to build replay profiles."
}
python scripts\d15_p3c_make_batch2_all15_manifest.py `
  --config $Config `
  --replay-manifest-csv $ReplayManifest `
  --out-csv $All15Manifest `
  --out-json $All15ManifestJson

Write-Host "[D15-P3C] 2/8 generate Batch-2 15-cell P2Dlite-RG soft labels" -ForegroundColor Cyan
$GenArgs = @(
  "scripts\d15_p3c_generate_batch2_15cell_rg_softlabels.py",
  "--config", $Config,
  "--all15-manifest-csv", $All15Manifest,
  "--prior-json", $PriorJson,
  "--output-dir", $SoftlabelDir
)
if ($AllowOverwrite) { $GenArgs += "--allow-overwrite" }
python @GenArgs

Write-Host "[D15-P3C] 3/8 radial audit for Batch-2 15-cell RG labels" -ForegroundColor Cyan
$AuditArgs = @(
  "scripts\d15_p0_radial_gradient_audit.py",
  "--source-dir", $SoftlabelDir,
  "--prior-json", $PriorJson,
  "--out-dir", $AuditDir
)
if ($AllowOverwrite) { $AuditArgs += "--allow-overwrite" }
python @AuditArgs

$NNScorecardJson = Join-Path $NNRunDir "D15_P1_FINAL_SCORECARD.json"
$BoundaryScorecardJson = Join-Path $BoundaryDir "D15_P3B_FINAL_SCORECARD.json"
if (-not $SkipNN) {
  Write-Host "[D15-P3C] 4/8 train 15-cell closed-set NN benchmark on GPU if available" -ForegroundColor Cyan
  $EffectiveNNConfig = $NNConfig
  $EffectiveBoundaryConfig = $BoundaryConfig
  if ($Quick) {
    $QuickNN = Join-Path $ScorecardDir "d15_p3c_batch2_15cell_nn_config_QUICK.json"
    $obj = Get-Content $NNConfig -Raw | ConvertFrom-Json
    $obj.training.epochs = 160
    $obj.data.max_time_points_per_profile_train = 2048
    $obj.data.max_time_points_per_profile_val = 512
    $obj.data.eval_stride = 16
    $obj | ConvertTo-Json -Depth 20 | Set-Content $QuickNN -Encoding UTF8
    $EffectiveNNConfig = $QuickNN

    $QuickBoundary = Join-Path $ScorecardDir "d15_p3c_boundary_repair_config_QUICK.json"
    $b = Get-Content $BoundaryConfig -Raw | ConvertFrom-Json
    $b.data.eval_stride = 16
    $b | ConvertTo-Json -Depth 20 | Set-Content $QuickBoundary -Encoding UTF8
    $EffectiveBoundaryConfig = $QuickBoundary
  }

  $TrainArgs = @(
    "scripts\d15_p1_train_rg_closedset_nn_smoke.py",
    "--softlabel-dir", $SoftlabelDir,
    "--out-dir", $NNRunDir,
    "--config", $EffectiveNNConfig
  )
  if ($AllowOverwrite) { $TrainArgs += "--allow-overwrite" }
  python @TrainArgs

  Write-Host "[D15-P3C] 5/8 evaluate raw 15-cell NN benchmark" -ForegroundColor Cyan
  python scripts\d15_p1_eval_rg_closedset_nn_smoke.py `
    --softlabel-dir $SoftlabelDir `
    --model-dir $NNRunDir `
    --out-dir (Join-Path $NNRunDir "eval_full_profiles") `
    --config $EffectiveNNConfig `
    --batch-size 262144 `
    --allow-overwrite

  python scripts\d15_p1_collect_scorecard.py `
    --run-dir $NNRunDir `
    --eval-dir (Join-Path $NNRunDir "eval_full_profiles") `
    --out-json $NNScorecardJson

  Write-Host "[D15-P3C] 6/8 apply theta projection repair and audit projected metrics" -ForegroundColor Cyan
  $BoundaryArgs = @(
    "scripts\d15_p3b_boundary_projection_repair.py",
    "--softlabel-dir", $SoftlabelDir,
    "--model-dir", $NNRunDir,
    "--out-dir", $BoundaryDir,
    "--config", $EffectiveBoundaryConfig
  )
  if ($AllowOverwrite) { $BoundaryArgs += "--allow-overwrite" }
  python @BoundaryArgs
  python scripts\d15_p3b_collect_scorecard.py `
    --repair-dir $BoundaryDir `
    --out-json $BoundaryScorecardJson
} else {
  Write-Host "[D15-P3C] 4-6/8 NN benchmark skipped by -SkipNN" -ForegroundColor Yellow
  $NNScorecardJson = ""
  $BoundaryScorecardJson = ""
}

Write-Host "[D15-P3C] 7/8 collect final Batch-2 15-cell scorecard" -ForegroundColor Cyan
$CollectArgs = @(
  "scripts\d15_p3c_collect_scorecard.py",
  "--scorecard-dir", $ScorecardDir,
  "--all15-manifest-json", $All15ManifestJson,
  "--generation-json", (Join-Path $SoftlabelDir "D15_P3C_BATCH2_15CELL_RG_GENERATION_REPORT.json"),
  "--radial-audit-json", (Join-Path $AuditDir "radial_gradient_audit_summary.json"),
  "--out-json", $FinalScorecard
)
if (-not $SkipNN) { $CollectArgs += @("--nn-scorecard-json", $NNScorecardJson, "--projection-scorecard-json", $BoundaryScorecardJson) }
python @CollectArgs

Write-Host "[D15-P3C] 8/8 pack review zip" -ForegroundColor Cyan
python scripts\d15_p3c_pack_review.py `
  --cache-root $CacheRoot `
  --out-zip $ReviewZip

Write-Host "[D15-P3C] DONE" -ForegroundColor Green
Write-Host "Review zip: $ReviewZip" -ForegroundColor Green
Write-Host "Scorecard: $FinalScorecard" -ForegroundColor Green
