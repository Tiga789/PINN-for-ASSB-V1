param(
  [switch]$AllowOverwrite,
  [int]$EvalStride = 0,
  [int]$BatchSize = 0,
  [string]$Device = "",
  [ValidateSet("zero", "same_batch_first")]
  [string]$OnehotUnseenPolicy = "zero"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

$Config = "configs\d15_p5a_all55_existing_model_transfer_config.json"
$CfgObj = Get-Content $Config -Raw | ConvertFrom-Json
$OutDir = $CfgObj.output_dir

if ($AllowOverwrite -and (Test-Path $OutDir)) {
  Write-Host "[D15-P5A] Removing old output dir: $OutDir" -ForegroundColor Yellow
  Remove-Item -Recurse -Force $OutDir
}

Write-Host "[D15-P5A] 0/4 selftest" -ForegroundColor Cyan
python scripts\d15_p5a_selftest.py
if ($LASTEXITCODE -ne 0) { throw "D15-P5A selftest failed" }

Write-Host "[D15-P5A] 1/4 preflight" -ForegroundColor Cyan
python scripts\d15_p5a_preflight.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "D15-P5A preflight failed" }

Write-Host "[D15-P5A] 2/4 existing-model transfer evaluation" -ForegroundColor Cyan
$EvalArgs = @("scripts\d15_p5a_existing_model_transfer_eval.py", "--config", $Config, "--allow-overwrite", "--onehot-unseen-policy", $OnehotUnseenPolicy)
if ($EvalStride -gt 0) { $EvalArgs += @("--eval-stride", [string]$EvalStride) }
if ($BatchSize -gt 0) { $EvalArgs += @("--batch-size", [string]$BatchSize) }
if ($Device -ne "") { $EvalArgs += @("--device", $Device) }
python @EvalArgs
if ($LASTEXITCODE -ne 0) { throw "D15-P5A transfer evaluation failed" }

Write-Host "[D15-P5A] 3/4 pack review zip" -ForegroundColor Cyan
python scripts\d15_p5a_pack_review.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "D15-P5A pack review failed" }

Write-Host "[D15-P5A] DONE" -ForegroundColor Green
Write-Host "Review zip: $($CfgObj.review_zip)"
Write-Host "Scorecard : $OutDir\D15_P5A_FINAL_SCORECARD.json"
