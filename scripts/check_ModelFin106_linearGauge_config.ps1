$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (!(Test-Path ".\ModelFin_106\best.pt")) { throw "Missing .\ModelFin_106\best.pt" }
if (!(Test-Path ".\ModelFin_106\config.json")) { throw "Missing .\ModelFin_106\config.json" }
if (!(Test-Path ".\ModelFin_106\gauge_config.json")) { throw "Missing .\ModelFin_106\gauge_config.json" }

Write-Host "==== ModelFin_106 gauge config ===="
$g = Get-Content ".\ModelFin_106\gauge_config.json" -Raw | ConvertFrom-Json
$g | Select-Object model_id, model_name, calibration_method, calib_cycle_from, calib_cycle_to, apply_cycle_from, apply_cycle_to, linear_bias_slope_V_per_cycle, linear_bias_intercept_V, offset_formula | Format-List

Write-Host "==== ModelFin_106 config key check ===="
Get-Content ".\ModelFin_106\config.json" | Select-String "ASSB_MODELFIN_WRAPPER_ID|ASSB_COMMON_MODE_GAUGE|ASSB_SOFT_LABEL_DIR|CBAR_BASELINE|USE_I_CBAR|ZERO_MEAN|CURRENT_POTENTIAL|MAX_BATCH_SIZE_DATA|activeData"
