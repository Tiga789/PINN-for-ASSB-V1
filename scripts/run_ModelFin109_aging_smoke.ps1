# ASSB ModelFin_109 aging mechanism smoke run
# Run from project root after installing the first and second ModelFin109 packages.

param(
    [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$SolutionNpz = "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz",
    [string]$CapacityTargetCsv = ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv",
    [string]$CycleTableCsv = ".\Data\assb_aging_ModelFin109\cycle_table.csv",
    [string]$CycleTableJson = ".\Data\assb_aging_ModelFin109\cycle_table_summary.json",
    [int]$CycleFrom = 5,
    [int]$CycleTo = 30,
    [int]$TrainTo = 20,
    [int]$ValTo = 25,
    [string]$InputFile = ".\input_assb_ModelFin109_aging"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force ".\Data\assb_aging_ModelFin109" | Out-Null

if (-not (Test-Path $CycleTableCsv)) {
    & $PythonExe .\scripts\prepare_assb_aging_cycle_table.py `
      --solution_npz $SolutionNpz `
      --capacity_target_csv $CapacityTargetCsv `
      --cycle_from $CycleFrom `
      --cycle_to $CycleTo `
      --train_to $TrainTo `
      --val_to $ValTo `
      --output_csv $CycleTableCsv `
      --output_json $CycleTableJson
}

.\scripts\check_ModelFin109_aging_package.ps1 -PythonExe $PythonExe -CycleTableCsv $CycleTableCsv -CapacityTargetCsv $CapacityTargetCsv

Remove-Item -Recurse -Force ".\ModelFin_109", ".\LogFin_109", ".\DataFin_109" -ErrorAction SilentlyContinue

Write-Host "[ASSB109 smoke] starting training with $InputFile"
& $PythonExe .\main.py -i $InputFile

Write-Host "[ASSB109 smoke] training finished; checking key outputs"
Test-Path ".\ModelFin_109\best.pt"
Test-Path ".\ModelFin_109\config.json"
Test-Path ".\ModelFin_109\aging_state.pt"
Test-Path ".\ModelFin_109\aging_config.json"
