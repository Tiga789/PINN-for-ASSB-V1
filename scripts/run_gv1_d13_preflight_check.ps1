param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D13 preflight ===="
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "PythonExe   = $PythonExe"

if (-not (Test-Path $PythonExe)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path "scripts\gv1_d13_segment_protocol_diagnosis.py")) { throw "Missing scripts\gv1_d13_segment_protocol_diagnosis.py" }

& $PythonExe -m compileall gv1 scripts
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }

$d10 = Join-Path $CacheRoot "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8"
$d12 = Join-Path $CacheRoot "xjtu_batch134_d12_s3_metadata_ablation_scorecard"

Write-Host "D10-P1 dir exists: $(Test-Path $d10) -- $d10"
Write-Host "D12-S3 scorecard dir exists: $(Test-Path $d12) -- $d12"

if (Test-Path $d12) {
  $s = Join-Path $d12 "d12_s3_scorecard_summary.json"
  Write-Host "D12 summary exists: $(Test-Path $s) -- $s"
}

Write-Host "D13 preflight finished. If at least D12-S3 exists, diagnosis can run."
