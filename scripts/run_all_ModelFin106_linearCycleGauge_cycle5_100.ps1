$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

.\scripts\run_build_ModelFin106_linearCycleGauge.ps1
.\scripts\check_ModelFin106_linearGauge_config.ps1
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_100_linearGauge.ps1
