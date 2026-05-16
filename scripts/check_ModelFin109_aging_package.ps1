# ASSB ModelFin_109 aging mechanism package check
# Run from project root:
#   .\scripts\check_ModelFin109_aging_package.ps1

param(
    [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$CycleTableCsv = ".\Data\assb_aging_ModelFin109\cycle_table.csv",
    [string]$CapacityTargetCsv = ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv"
)

$ErrorActionPreference = "Stop"

Write-Host "[ASSB109 check] project root: $(Get-Location)"

$required = @(
    ".\scripts\prepare_assb_aging_cycle_table.py",
    ".\scripts\run_ModelFin109_aging_smoke.ps1",
    ".\scripts\run_ModelFin109_aging_train_eval.ps1",
    ".\scripts\check_ModelFin109_aging_package.ps1",
    ".\util\assb_cycle_table.py",
    ".\util\assb_aging_state.py",
    ".\util\assb_aging_physics.py",
    ".\util\assb_capacity_from_states.py",
    ".\evaluate_assb_ModelFin109_aging.py",
    ".\input_assb_ModelFin109_aging"
)

foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        throw "Missing required ModelFin109 new file: $p"
    }
}
Write-Host "[ASSB109 check] required new files: OK"

& $PythonExe -m py_compile `
    .\scripts\prepare_assb_aging_cycle_table.py `
    .\util\assb_cycle_table.py `
    .\util\assb_aging_state.py `
    .\util\assb_aging_physics.py `
    .\util\assb_capacity_from_states.py `
    .\evaluate_assb_ModelFin109_aging.py
Write-Host "[ASSB109 check] py_compile: OK"

Select-String -Path .\util\assb_aging_state.py -Pattern "AgingMechanismHead|f_lam_c|theta_window_c|soh_mech" | Out-Host
Select-String -Path .\util\assb_aging_physics.py -Pattern "effective_volume_at_t|aged_surface_flux|aged_theta_window|aged_terminal_shift|assert_fixed_material_identity" | Out-Host
Select-String -Path .\input_assb_ModelFin109_aging -Pattern "USE_ASSB_AGING_MECHANISM|DATA_LOSS|MAX_BATCH_SIZE_DATA|AGING_USE_LAM_C|AGING_USE_R_OHM|AGING_USE_THETA_WINDOW_C" | Out-Host

# Guard against accidental dependency on separately backed-up old files.
$bad = Select-String -Path @(
    ".\scripts\prepare_assb_aging_cycle_table.py",
    ".\util\assb_cycle_table.py",
    ".\util\assb_aging_state.py",
    ".\util\assb_aging_physics.py",
    ".\util\assb_capacity_from_states.py",
    ".\evaluate_assb_ModelFin109_aging.py"
) -Pattern "107A_base|_107A_base|_base\.py|load_base" -SimpleMatch -ErrorAction SilentlyContinue
if ($bad) {
    $bad | Out-Host
    throw "Detected dependency on backed-up/base files. This package must not use that pattern."
}
Write-Host "[ASSB109 check] no backup/base-file dependency pattern detected."

if (Test-Path $CycleTableCsv) {
    Write-Host "[ASSB109 check] cycle table exists: $CycleTableCsv"
} else {
    Write-Host "[ASSB109 check] cycle table not found yet: $CycleTableCsv"
    Write-Host "                 Run scripts\prepare_assb_aging_cycle_table.py before training."
}

if (Test-Path $CapacityTargetCsv) {
    Write-Host "[ASSB109 check] capacity target exists: $CapacityTargetCsv"
} else {
    Write-Host "[ASSB109 check] capacity target not found yet: $CapacityTargetCsv"
    Write-Host "                 Generate capacity_soh_targets.csv before mechanism training/eval."
}

Write-Host "[ASSB109 check] done."
