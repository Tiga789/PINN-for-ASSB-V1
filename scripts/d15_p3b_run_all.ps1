param(
  [switch]$AllowOverwrite,
  [switch]$Quick,
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
$SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p3_batch2_3cell"
$ModelDir = Join-Path $CacheRoot "xjtu_d15_p3_batch2_3cell_rg_nn_smoke"
$OutDir = Join-Path $CacheRoot "xjtu_d15_p3b_batch2_boundary_projection_repair"
$ScorecardJson = Join-Path $OutDir "D15_P3B_FINAL_SCORECARD.json"
$ReviewZip = Join-Path $CacheRoot "xjtu_d15_p3b_results_for_review.zip"
$Config = "configs\d15_p3b_boundary_repair_config.json"

$BatchSize = 262144
$EvalStride = 1
if ($Quick) {
  $BatchSize = 65536
  $EvalStride = 8
  Write-Host "[D15-P3B] QUICK mode: eval_stride=$EvalStride batch_size=$BatchSize" -ForegroundColor Yellow
}

$OverwriteArgs = @()
if ($AllowOverwrite) { $OverwriteArgs += "--allow-overwrite" }

Write-Host "[D15-P3B] 0/4 selftest" -ForegroundColor Cyan
python scripts\d15_p3b_selftest_boundary_repair.py

Write-Host "[D15-P3B] 1/4 boundary projection repair eval" -ForegroundColor Cyan
python scripts\d15_p3b_boundary_projection_repair.py `
  --softlabel-dir "$SoftlabelDir" `
  --model-dir "$ModelDir" `
  --out-dir "$OutDir" `
  --config "$Config" `
  --device "$Device" `
  --batch-size $BatchSize `
  --eval-stride $EvalStride `
  @OverwriteArgs

Write-Host "[D15-P3B] 2/4 collect scorecard" -ForegroundColor Cyan
python scripts\d15_p3b_collect_scorecard.py `
  --repair-dir "$OutDir" `
  --out-json "$ScorecardJson"

Write-Host "[D15-P3B] 3/4 pack review zip" -ForegroundColor Cyan
python scripts\d15_p3b_pack_review.py `
  --repair-dir "$OutDir" `
  --scorecard-json "$ScorecardJson" `
  --out-zip "$ReviewZip"

Write-Host "[D15-P3B] DONE" -ForegroundColor Green
Write-Host "Review zip: $ReviewZip"
Write-Host "Scorecard: $ScorecardJson"
