param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S6 preflight ===="
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "PythonExe   = $PythonExe"

if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (!(Test-Path (Join-Path $ProjectRoot "scripts\gv1_d11_s6_lowtarget_floor_capacity_audit.py"))) { throw "D11-S6 Python script missing." }

$predRoot = Join-Path $CacheRoot "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair"
$scoreRoot = Join-Path $CacheRoot "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_scorecard"
if (!(Test-Path $predRoot)) { throw "D11-S5C prediction root missing: $predRoot" }
if (!(Test-Path $scoreRoot)) { throw "D11-S5C scorecard root missing: $scoreRoot" }

$predCount = (Get-ChildItem $predRoot -Recurse -Filter prediction.npz -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "D11-S5C prediction.npz count = $predCount"
if ($predCount -lt 24) { throw "Expected at least 24 D11-S5C prediction.npz files. Found $predCount." }

& $PythonExe -m compileall gv1 scripts
if ($LASTEXITCODE -ne 0) { throw "compileall failed." }

Write-Host "D11-S6 preflight PASS."
