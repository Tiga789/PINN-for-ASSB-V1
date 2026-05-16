<#
Run ASSB ModelFin_108 capacity/SOH preparation and evaluation.

This first-step script adds no original soft-label data loss. It prepares the
cycle-level capacity target from ZHB_ASSB_NCM811.xlsx step sheet and can run a
standalone capacity-head smoke fit. Running main.py requires the later patches
that register the capacity head inside init_pinn/_losses/myNN.
#>

param(
    [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
    [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$ZhbXlsx = "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\ZHB_ASSB_NCM811.xlsx",
    [string]$CapacityTargetCsv = "Data\assb_capacity_targets_cycle5_521_from_step.csv",
    [string]$CapacityTargetJson = "Data\assb_capacity_targets_cycle5_521_from_step_summary.json",
    [switch]$RunStandaloneCapacitySmoke,
    [switch]$RunMain
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[ASSB-108] ProjectRoot = $ProjectRoot"
Write-Host "[ASSB-108] PythonExe   = $PythonExe"
Write-Host "[ASSB-108] Source xlsx = $ZhbXlsx"

if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (!(Test-Path $ZhbXlsx)) { throw "ZHB xlsx not found: $ZhbXlsx" }

# Avoid stale long-sequence summary overriding current paths.
Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force "Data" | Out-Null

Write-Host "`n[ASSB-108] Step 1/3: prepare capacity/SOH target from step sheet..."
& $PythonExe .\scripts\prepare_assb_capacity_soh_targets.py `
    --source_path $ZhbXlsx `
    --excel_sheet step `
    --cycle_from 5 `
    --cycle_to 522 `
    --qref_cycle_from 5 `
    --qref_cycle_to 20 `
    --complete_v_max 2.20 `
    --min_q_ah 0.00025 `
    --exclude_incomplete `
    --output_csv $CapacityTargetCsv `
    --output_json $CapacityTargetJson

if (!(Test-Path $CapacityTargetCsv)) { throw "capacity target CSV was not created: $CapacityTargetCsv" }
if (!(Test-Path $CapacityTargetJson)) { throw "capacity target JSON was not created: $CapacityTargetJson" }

Write-Host "`n[ASSB-108] Capacity target summary:"
Get-Content $CapacityTargetJson | Select-String "q_ref_mAh|n_train_cycles|n_incomplete_cycles|cycle_min|cycle_max|Q_dis_min_mAh|Q_dis_max_mAh|SOH_min|SOH_max"

Write-Host "`n[ASSB-108] Step 2/3: static checks for no data-loss setting and new capacity files..."
Get-Content .\input_assb_ModelFin108_capacityPhysicsOnly | Select-String "DATA_LOSS|ALPHA_DATA|MAX_BATCH_SIZE_DATA|CAPACITY_|ASSB_USE_CAPACITY_AGING|USE_ASSB_CAPACITY_AGING"
Get-ChildItem .\scripts\prepare_assb_capacity_soh_targets.py, .\util\assb_capacity_targets.py, .\util\aging_assb_capacity.py, .\evaluate_assb_capacity_curve.py | Format-Table Name, Length

if ($RunStandaloneCapacitySmoke) {
    Write-Host "`n[ASSB-108] Step 3/3: standalone capacity-head smoke fit/eval..."
    & $PythonExe .\evaluate_assb_capacity_curve.py `
        --model_dir ModelFin_108_capacityHeadStandalone `
        --capacity_target_csv $CapacityTargetCsv `
        --output_dir EvalFin_108_capacity_curve_standaloneSmoke `
        --fit_standalone `
        --fit_epochs 3000 `
        --fit_lr 0.002 `
        --device cpu

    Write-Host "`n[ASSB-108] Standalone capacity metrics:"
    Get-Content .\EvalFin_108_capacity_curve_standaloneSmoke\metrics_capacity_global.json | Select-String "train_Q_MAE_mAh|train_SOH_MAE|train_Q_R2|all_Q_MAE_mAh|all_SOH_MAE"
}
else {
    Write-Host "`n[ASSB-108] Standalone capacity smoke skipped. Add -RunStandaloneCapacitySmoke to verify target/evaluator loop now."
}

if ($RunMain) {
    Write-Host "`n[ASSB-108] Running main.py. This requires later patches to init_pinn/_losses/myNN so the capacity head enters training."
    & $PythonExe .\main.py -i input_assb_ModelFin108_capacityPhysicsOnly
}
else {
    Write-Host "`n[ASSB-108] Main training skipped. After the integration patch, run:"
    Write-Host "  $PythonExe .\main.py -i input_assb_ModelFin108_capacityPhysicsOnly"
}

Write-Host "`n[ASSB-108] Done."
