param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
& $PythonExe "scripts\gv1_d12_rescue_true_smoke_metrics_and_scorecard.py" --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D12 TRUE SMOKE rescue scorecard failed." }
