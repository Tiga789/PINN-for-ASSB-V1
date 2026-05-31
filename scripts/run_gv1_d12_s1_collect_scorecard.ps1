param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [switch]$IncludeLegacyTrueSmoke
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$ArgList = @("scripts\gv1_d12_s1_scorecard_from_predictions.py", "--cache_root", $CacheRoot)
if ($IncludeLegacyTrueSmoke) { $ArgList += "--include_legacy_true_smoke" }
& $Python @ArgList
if ($LASTEXITCODE -ne 0) { throw "D12-S1 scorecard collection failed." }
