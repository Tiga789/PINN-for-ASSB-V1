param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S5B low-target gate/sign analysis ===="
& $PythonExe scripts\gv1_d11_s5b_lowtarget_gate_sign_analysis.py --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D11-S5B gate/sign analysis failed." }

$outDir = Join-Path $CacheRoot "xjtu_batch134_d11_s5b_lowtarget_gate_sign_analysis"
Write-Host "D11-S5B output: $outDir"
Write-Host "Open recommendation:"
Write-Host "  notepad `"$outDir\D11_S5B_RECOMMENDATION.md`""
