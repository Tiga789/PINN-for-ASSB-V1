param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [switch]$MakePlots
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$argsList = @(
  "scripts/gv1_d10_p0_battery8_regime_judgement.py",
  "--cache_root", $CacheRoot
)
if ($MakePlots) { $argsList += "--make_plots" }

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) { throw "D10-P0 judgement failed." }

$outDir = Join-Path $CacheRoot "xjtu_batch134_d10_p0_battery8_regime_judgement"
Write-Host ""
Write-Host "D10-P0 output directory:" $outDir
Write-Host "Recommendation:"
Get-Content (Join-Path $outDir "D10_P0_RECOMMENDATION.md") -Raw
