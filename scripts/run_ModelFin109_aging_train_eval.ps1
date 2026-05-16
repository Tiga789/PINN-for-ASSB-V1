# ASSB ModelFin_109 aging mechanism train + evaluate flow
# Run from project root after installing the full ModelFin109 code package.

param(
    [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$SolutionNpz = "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz",
    [string]$CapacityTargetCsv = ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv",
    [string]$CycleTableCsv = ".\Data\assb_aging_ModelFin109\cycle_table.csv",
    [string]$CycleTableJson = ".\Data\assb_aging_ModelFin109\cycle_table_summary.json",
    [string]$InputFile = ".\input_assb_ModelFin109_aging",
    [string]$OutputDir = ".\EvalFin_109_aging_mechanism",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force ".\Data\assb_aging_ModelFin109" | Out-Null

if (-not (Test-Path $CycleTableCsv)) {
    & $PythonExe .\scripts\prepare_assb_aging_cycle_table.py `
      --solution_npz $SolutionNpz `
      --capacity_target_csv $CapacityTargetCsv `
      --cycle_from 5 `
      --cycle_to 522 `
      --train_to 300 `
      --val_to 420 `
      --output_csv $CycleTableCsv `
      --output_json $CycleTableJson
}

.\scripts\check_ModelFin109_aging_package.ps1 -PythonExe $PythonExe -CycleTableCsv $CycleTableCsv -CapacityTargetCsv $CapacityTargetCsv

Remove-Item -Recurse -Force ".\ModelFin_109", ".\LogFin_109", ".\DataFin_109", $OutputDir -ErrorAction SilentlyContinue

& $PythonExe .\main.py -i $InputFile

& $PythonExe .\evaluate_assb_ModelFin109_aging.py `
  --model_dir ".\ModelFin_109" `
  --solution_npz $SolutionNpz `
  --capacity_target_csv $CapacityTargetCsv `
  --cycle_table_csv $CycleTableCsv `
  --output_dir $OutputDir `
  --device $Device

Get-Content "$OutputDir\metrics_capacity_by_split.json"
Get-Content "$OutputDir\metrics_states_global.json"
