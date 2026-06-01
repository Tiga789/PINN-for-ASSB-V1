param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot
& $PythonExe ".\scripts\gv1_d11_s9_scorecard_from_predictions.py" `
  --prediction_root "$CacheRoot\xjtu_batch134_d11_s9_trainable_p2dlike_correction" `
  --out_dir "$CacheRoot\xjtu_batch134_d11_s9_trainable_p2dlike_correction_scorecard"
if ($LASTEXITCODE -ne 0) { throw "D11-S9 scorecard collection failed." }
