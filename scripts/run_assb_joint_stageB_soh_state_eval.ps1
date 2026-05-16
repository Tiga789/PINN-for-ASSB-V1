
param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$StageBEvalDir = ".\EvalFin_110_stageB_aging",
  [string]$ReferenceNpz = "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz",
  [string]$StateEvalDir = ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only",
  [string]$StatePredictionNpz = "",
  [string]$CycleTableCsv = ".\Data\assb_aging_fix1\cycle_table.csv",
  [string]$OutputDir = ".\EvalFin_110_joint_StageB_SOH_107A_states_fix2"
)

if (-not (Test-Path $StageBEvalDir)) {
  Write-Error "StageBEvalDir not found: $StageBEvalDir"
  exit 1
}

$argsList = @(
  ".\evaluate_assb_joint_states_soh.py",
  "--stageB_eval_dir", $StageBEvalDir,
  "--output_dir", $OutputDir
)

if ($ReferenceNpz -and (Test-Path $ReferenceNpz)) {
  $argsList += @("--reference_npz", $ReferenceNpz)
} elseif ($ReferenceNpz) {
  Write-Warning "ReferenceNpz not found. This is okay if the state npz contains *_true fields: $ReferenceNpz"
}

if ($CycleTableCsv -and (Test-Path $CycleTableCsv)) {
  $argsList += @("--cycle_table_csv", $CycleTableCsv)
}

if ($StatePredictionNpz -and (Test-Path $StatePredictionNpz)) {
  $argsList += @("--state_prediction_npz", $StatePredictionNpz)
} elseif ($StateEvalDir -and (Test-Path $StateEvalDir)) {
  $argsList += @("--state_eval_dir", $StateEvalDir)
} else {
  Write-Error "No state prediction npz or state eval dir found. Provide -StatePredictionNpz or -StateEvalDir."
  exit 1
}

& $PythonExe @argsList
