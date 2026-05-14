$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (!(Test-Path ".\ModelFin_105\config.json")) {
  throw "ModelFin_105\config.json was not found. Run training first."
}

Get-Content .\ModelFin_105\config.json | Select-String "ASSB_SOFT_LABEL_DIR|ASSB_CYCLE_FROM|ASSB_CYCLE_TO|alpha|activeData|MAX_BATCH_SIZE_DATA|USE_I_CBAR|ZERO_MEAN|CBAR_BASELINE|CURRENT_POTENTIAL|POTENTIAL_BASELINE|LONG_SEQUENCE_MODE|w_phie_dat|w_phis_c_dat|LOAD_MODEL"
