$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "=== util/spm_assb_train_discharge.py keyword check ==="
Select-String -Path .\util\spm_assb_train_discharge.py `
  -Pattern "t_global_s|cycle_id|time_scale_s|ASSB_LONG_SEQUENCE|dynamic_from_continuous_solution_npz|assb_soft_lable_cycle5-522_v1"

Write-Host "=== input keyword check ==="
Select-String -Path .\input_assb_cycles5to522_v4_continuous_ID102 `
  -Pattern "ASSB_LONG_SEQUENCE|dynamic_from_continuous_solution_npz|assb_soft_lable_cycle5-20_v1_smoke|soft_label_phis_c"

Write-Host "=== smoke soft-label folder check ==="
if (Test-Path .\Data\assb_soft_lable_cycle5-20_v1_smoke\solution.npz) {
  Write-Host "[OK] Data\assb_soft_lable_cycle5-20_v1_smoke\solution.npz exists"
} else {
  Write-Host "[WARN] smoke slice not found yet. Run scripts\01_make_102_smoke_softlabels.ps1 first."
}
