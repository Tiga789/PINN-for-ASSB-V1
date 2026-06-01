param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot
& $PythonExe ".\scripts\gv1_d11_s9_trainable_p2dlike_head_from_baseline.py" `
  --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D11-S9 trainable correction generation failed." }
