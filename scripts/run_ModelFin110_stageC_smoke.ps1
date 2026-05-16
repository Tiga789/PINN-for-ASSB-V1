param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$InputFile = ".\input_assb_ModelFin110_agingStageC"
)

$ErrorActionPreference = "Stop"
Write-Host "Stage C smoke requires the second package with complete modified training files." -ForegroundColor Yellow
Write-Host "This script only clears ModelFin_110/LogFin_110 and runs main.py with $InputFile." -ForegroundColor Yellow
Remove-Item -Recurse -Force ".\ModelFin_110" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\LogFin_110" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\DataFin_110" -ErrorAction SilentlyContinue
& $PythonExe .\main.py -i $InputFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Stage C smoke finished. Check ModelFin_110/best.pt, aging_state.pt, aging_config.json." -ForegroundColor Green
