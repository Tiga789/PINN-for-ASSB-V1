param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [switch]$AfterPrepare
)

$ErrorActionPreference = 'Stop'

Set-Location $ProjectRoot

Write-Host "==== D12-S3 preflight: required files ===="
$required = @(
  "scripts\gv1_train_conditioned_pinn.py",
  "scripts\gv1_train_conditioned_pinn_d12_metadata_runtime.py",
  "scripts\gv1_d12_s3_prepare_23profile_strict_commands.py",
  "scripts\gv1_d12_s3_scorecard_from_predictions.py",
  "scripts\run_gv1_d12_s3_prepare_commands.ps1",
  "scripts\run_gv1_d12_s3_collect_scorecard.ps1",
  "gv1\output_transform.py",
  "gv1\losses.py",
  "gv1\trainer.py",
  "gv1\model.py"
)
foreach ($rel in $required) {
  $path = Join-Path $ProjectRoot $rel
  if (-not (Test-Path $path)) { throw "Missing required file: $rel" }
  Write-Host "OK: $rel"
}

Write-Host "`n==== D12-S3 preflight: Python syntax ===="
& $PythonExe -m compileall gv1 scripts
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

Write-Host "`n==== D12-S3 preflight: mainline markers ===="
Select-String -Path "scripts\gv1_train_conditioned_pinn.py" -Pattern "D9.5.1","trend-first","warmup","rare" | Select-Object -First 20
Select-String -Path "gv1\output_transform.py" -Pattern "enable_voltage_hard_clamp: bool = False" | Select-Object -First 5

Write-Host "`n==== D12-S3 preflight: forbidden source-script parameters ===="
$bad = Select-String -Path "scripts\*.ps1","scripts\*.py" -Pattern "epochs=40000","time_window_s=200000","enable_voltage_hard_clamp=True" -ErrorAction SilentlyContinue
if ($bad) {
  Write-Warning "Potential forbidden strings found in source scripts. Inspect context below; README warnings may be acceptable, runnable commands are not."
  $bad
} else {
  Write-Host "OK: no forbidden source-script strings found."
}

if ($AfterPrepare) {
  $cmdDir = Join-Path $CacheRoot "xjtu_batch134_d12_s3_metadata_ablation_commands"
  Write-Host "`n==== D12-S3 preflight: generated command scripts ===="
  if (-not (Test-Path $cmdDir)) { throw "Generated command directory not found: $cmdDir" }
  $generated = Get-ChildItem $cmdDir -Filter "*.ps1"
  if ($generated.Count -lt 4) { throw "Expected at least 4 generated ps1 scripts, got $($generated.Count)" }
  $forbiddenGenerated = Select-String -Path (Join-Path $cmdDir "*.ps1") -Pattern "epochs 40000","time_window_s 200000","max_time_points 8192","batch_size 2048","_200ks" -ErrorAction SilentlyContinue
  if ($forbiddenGenerated) {
    $forbiddenGenerated
    throw "Forbidden long-run parameter found in generated D12-S3 command scripts."
  }
  Write-Host "OK: generated scripts are strict-smoke safe."
}

Write-Host "`nD12-S3 preflight PASS."

