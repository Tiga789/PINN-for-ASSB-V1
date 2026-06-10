param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelDir = "",
  [string]$RunDir = "",
  [string]$PriorJson = "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json",
  [string]$Config = "configs\d15_p2_precision_benchmark_config.json",
  [string]$Device = "auto",
  [switch]$AllowOverwrite,
  [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SoftlabelDir)) {
  $SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d15_p2_rg_precision_benchmark"
}
$EvalDir = Join-Path $RunDir "eval_full_profiles"
$AuditDir = Join-Path $RunDir "precision_audit"
$ReviewZip = Join-Path $CacheRoot "xjtu_d15_p2_results_for_review.zip"

Write-Host "[D15-P2] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D15-P2] SoftlabelDir=$SoftlabelDir" -ForegroundColor Cyan
Write-Host "[D15-P2] RunDir=$RunDir" -ForegroundColor Cyan

if ((Test-Path $RunDir) -and -not $AllowOverwrite) {
  throw "RunDir already exists: $RunDir. Use -AllowOverwrite to rerun."
}
if ((Test-Path $RunDir) -and $AllowOverwrite) {
  Remove-Item -Recurse -Force $RunDir
}
New-Item -ItemType Directory -Force $RunDir | Out-Null

Write-Host "[D15-P2] compile + selftest" -ForegroundColor Green
python -m compileall -q gv1 scripts
python scripts\d15_p2_selftest_precision_benchmark.py

Write-Host "[D15-P2] preflight" -ForegroundColor Green
python scripts\d15_p2_preflight.py `
  --softlabel-dir "$SoftlabelDir" `
  --prior-json "$PriorJson" `
  --config "$Config" `
  --out-json "$RunDir\D15_P2_PREFLIGHT.json"

Write-Host "[D15-P2] train precision benchmark" -ForegroundColor Green
$trainArgs = @(
  "scripts\d15_p2_train_rg_precision_benchmark.py",
  "--softlabel-dir", "$SoftlabelDir",
  "--out-dir", "$RunDir",
  "--config", "$Config",
  "--allow-overwrite"
)
if ($Quick) { $trainArgs += "--quick" }
if (-not [string]::IsNullOrWhiteSpace($Device)) { $trainArgs += @("--device", $Device) }
python @trainArgs

Write-Host "[D15-P2] eval full profiles with prediction dump" -ForegroundColor Green
python scripts\d15_p2_eval_rg_precision_benchmark.py `
  --softlabel-dir "$SoftlabelDir" `
  --model-dir "$RunDir" `
  --out-dir "$EvalDir" `
  --config "$Config" `
  --device "$Device" `
  --allow-overwrite

Write-Host "[D15-P2] precision audit" -ForegroundColor Green
python scripts\d15_p2_precision_audit.py `
  --softlabel-dir "$SoftlabelDir" `
  --eval-dir "$EvalDir" `
  --out-dir "$AuditDir" `
  --config "$Config" `
  --allow-overwrite

Write-Host "[D15-P2] final scorecard" -ForegroundColor Green
python scripts\d15_p2_collect_scorecard.py `
  --run-dir "$RunDir" `
  --eval-dir "$EvalDir" `
  --audit-dir "$AuditDir" `
  --out-json "$RunDir\D15_P2_FINAL_SCORECARD.json"

Write-Host "[D15-P2] pack review artifacts" -ForegroundColor Green
python scripts\d15_p2_pack_review.py `
  --run-dir "$RunDir" `
  --eval-dir "$EvalDir" `
  --audit-dir "$AuditDir" `
  --out-zip "$ReviewZip"

Write-Host "[D15-P2] DONE" -ForegroundColor Cyan
Write-Host "Review zip: $ReviewZip" -ForegroundColor Cyan
Write-Host "Final scorecard: $RunDir\D15_P2_FINAL_SCORECARD.json" -ForegroundColor Cyan
