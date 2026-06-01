param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot
$predRoot = Join-Path $CacheRoot "xjtu_batch134_d11_s8_p2dlike_transport_correction"
$outDir = Join-Path $CacheRoot "xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard"
& $PythonExe "scripts\gv1_d11_s8_scorecard_from_predictions.py" --prediction_root $predRoot --out_dir $outDir
if ($LASTEXITCODE -ne 0) { throw "D11-S8 scorecard collection failed." }
