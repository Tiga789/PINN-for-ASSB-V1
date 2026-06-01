param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S6 low-target floor / model-capacity audit ===="
& $PythonExe scripts\gv1_d11_s6_lowtarget_floor_capacity_audit.py `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot

if ($LASTEXITCODE -ne 0) { throw "D11-S6 audit failed." }

Write-Host "D11-S6 audit completed."
Write-Host "Output: $CacheRoot\xjtu_batch134_d11_s6_lowtarget_floor_capacity_audit"
