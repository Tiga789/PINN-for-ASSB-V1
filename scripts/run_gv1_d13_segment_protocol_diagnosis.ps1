param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$OutDir = ""
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

if ($OutDir -eq "") {
  $OutDir = Join-Path $CacheRoot "xjtu_batch134_d13_segment_protocol_diagnosis"
}

Write-Host "==== Run D13 segment/protocol diagnosis ===="
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "OutDir      = $OutDir"

& $PythonExe scripts\gv1_d13_segment_protocol_diagnosis.py `
  --cache_root $CacheRoot `
  --out_dir $OutDir

if ($LASTEXITCODE -ne 0) { throw "D13 diagnosis failed" }

Write-Host "==== D13 output ===="
Write-Host $OutDir
Write-Host "Open recommendation:"
Write-Host (Join-Path $OutDir "D13_RECOMMENDATION.md")
