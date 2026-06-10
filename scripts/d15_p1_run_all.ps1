param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell",
  [string]$PriorJson = "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json",
  [string]$ConfigJson = "configs\d15_p1_nn_smoke_config.json",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p1_rg_closedset_nn_smoke",
  [switch]$AllowOverwrite,
  [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Block
  )
  Write-Host "`n[D15-P1] === $Name ===" -ForegroundColor Cyan
  & $Block
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name, exit=$LASTEXITCODE"
  }
}

cd $ProjectRoot

$EvalDir = Join-Path $RunDir "eval_full_profiles"
$ScorecardJson = Join-Path $RunDir "D15_P1_FINAL_SCORECARD.json"
$PreflightJson = Join-Path $RunDir "D15_P1_PREFLIGHT.json"

if ((Test-Path $RunDir) -and (-not $AllowOverwrite)) {
  throw "RunDir already exists: $RunDir. Re-run with -AllowOverwrite only if you intentionally want to replace D15-P1 outputs."
}
if ($AllowOverwrite -and (Test-Path $RunDir)) {
  Remove-Item -Recurse -Force $RunDir
}
New-Item -ItemType Directory -Force $RunDir | Out-Null

Invoke-Step "compile python files" { python -m compileall -q gv1 scripts }
Invoke-Step "selftest NN utilities" { python scripts\d15_p1_selftest_nn_smoke.py }
Invoke-Step "preflight RG softlabels" {
  python scripts\d15_p1_preflight.py `
    --softlabel-dir "$SoftlabelDir" `
    --prior-json "$PriorJson" `
    --config "$ConfigJson" `
    --out-json "$PreflightJson"
}

$TrainArgs = @(
  "scripts\d15_p1_train_rg_closedset_nn_smoke.py",
  "--softlabel-dir", $SoftlabelDir,
  "--out-dir", $RunDir,
  "--config", $ConfigJson,
  "--allow-overwrite"
)
if ($Quick) { $TrainArgs += "--quick" }

Invoke-Step "train closed-set NN smoke" {
  python @TrainArgs
}

Invoke-Step "evaluate full profiles" {
  python scripts\d15_p1_eval_rg_closedset_nn_smoke.py `
    --softlabel-dir "$SoftlabelDir" `
    --model-dir "$RunDir" `
    --out-dir "$EvalDir" `
    --config "$ConfigJson" `
    --allow-overwrite
}

Invoke-Step "collect final scorecard" {
  python scripts\d15_p1_collect_scorecard.py `
    --run-dir "$RunDir" `
    --eval-dir "$EvalDir" `
    --out-json "$ScorecardJson"
}

Write-Host "`n[D15-P1] DONE" -ForegroundColor Green
Write-Host "RunDir: $RunDir"
Write-Host "Scorecard: $ScorecardJson"
