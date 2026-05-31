param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

& $PythonExe "scripts/gv1_d10_p1_prepare_23profile_200ks_plan.py" `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot

if ($LASTEXITCODE -ne 0) { throw "D10-P1 23-profile plan generation failed." }

$outDir = Join-Path $CacheRoot "xjtu_batch134_d10_p1_23profile_200ks_plan"
Write-Host ""
Write-Host "D10-P1 generated plan directory:" $outDir
Get-Content (Join-Path $outDir "d10_p1_prepare_23profile_plan_summary.json") -Raw
