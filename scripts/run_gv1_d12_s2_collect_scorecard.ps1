param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
& $Python "scripts\gv1_d12_s2_scorecard_from_predictions.py" --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D12-S2 scorecard collection failed." }
