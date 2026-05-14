$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Get-Content .\ModelFin_104\config.json | Select-String "ASSB_SOFT_LABEL_DIR|ASSB_CYCLE_FROM|ASSB_CYCLE_TO|USE_I_CBAR|ZERO_MEAN|CBAR_BASELINE|CURRENT_POTENTIAL|POTENTIAL_BASELINE|LONG_SEQUENCE_MODE"
