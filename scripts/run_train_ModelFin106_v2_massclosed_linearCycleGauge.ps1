$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "ModelFin_106 is a calibrated wrapper, not a new SGD/L-BFGS training run."
Write-Host "This script builds ModelFin_106 from ModelFin_105 + linear-cycle common-mode gauge."
.\scripts\run_build_ModelFin106_linearCycleGauge.ps1
