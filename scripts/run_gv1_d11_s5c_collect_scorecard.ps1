param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

& $PythonExe "scripts\gv1_d11_s5c_scorecard_from_predictions.py" --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D11-S5C scorecard collection failed." }
