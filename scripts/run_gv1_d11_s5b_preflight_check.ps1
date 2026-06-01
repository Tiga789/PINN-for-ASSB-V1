param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S5B preflight ===="
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "PythonExe   = $PythonExe"

$requiredFiles = @(
  "scripts\gv1_d11_s5b_lowtarget_gate_sign_analysis.py",
  "scripts\run_gv1_d11_s5b_lowtarget_gate_sign_analysis.ps1"
)
foreach ($f in $requiredFiles) {
  if (!(Test-Path $f)) { throw "Missing required file: $f" }
  Write-Host "OK file: $f"
}

$s5aPred = Join-Path $CacheRoot "xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis"
$s5aScore = Join-Path $CacheRoot "xjtu_batch134_d11_s5a_lowtarget_sign_gate_scorecard"
if (!(Test-Path $s5aPred)) { throw "Missing S5A prediction root: $s5aPred" }
if (!(Test-Path $s5aScore)) { throw "Missing S5A scorecard dir: $s5aScore" }

$count = (Get-ChildItem $s5aPred -Recurse -Filter prediction.npz | Measure-Object).Count
Write-Host "S5A prediction.npz count = $count"
if ($count -lt 1) { throw "No S5A prediction.npz files found." }

& $PythonExe -m compileall scripts\gv1_d11_s5b_lowtarget_gate_sign_analysis.py
if ($LASTEXITCODE -ne 0) { throw "compileall failed for D11-S5B analysis script." }

Write-Host "D11-S5B preflight PASS."
