param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$SolutionNpz = "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz",
  [string]$CapacityTargetCsv = ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv",
  [string]$CycleTableCsv = ".\Data\assb_aging_fix1\cycle_table.csv",
  [string]$CycleTableJson = ".\Data\assb_aging_fix1\cycle_table_summary.json",
  [string]$OutputDir = ".\ModelFin_110_stageB",
  [string]$EvalDir = ".\EvalFin_110_stageB_aging",
  [string]$Device = "cuda",
  [int]$CycleFrom = 5,
  [int]$CycleTo = 522,
  [int]$TrainTo = 300,
  [int]$ValTo = 420
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force ".\Data\assb_aging_fix1" | Out-Null

if (-not (Test-Path $CycleTableCsv)) {
  & $PythonExe .\scripts\prepare_assb_aging_fix1_cycle_table.py `
    --solution_npz $SolutionNpz `
    --capacity_target_csv $CapacityTargetCsv `
    --cycle_from $CycleFrom `
    --cycle_to $CycleTo `
    --train_to $TrainTo `
    --val_to $ValTo `
    --output_csv $CycleTableCsv `
    --output_json $CycleTableJson
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Remove-Item -Recurse -Force $OutputDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $EvalDir -ErrorAction SilentlyContinue

& $PythonExe .\scripts\train_assb_aging_stageB.py `
  --cycle_table_csv $CycleTableCsv `
  --capacity_target_csv $CapacityTargetCsv `
  --output_dir $OutputDir `
  --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe .\evaluate_assb_aging_fix1.py `
  --aging_model_dir $OutputDir `
  --cycle_table_csv $CycleTableCsv `
  --capacity_target_csv $CapacityTargetCsv `
  --output_dir $EvalDir `
  --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Stage B finished. See $OutputDir and $EvalDir" -ForegroundColor Green
