param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProjectRoot = ".",
  [string]$InputFile = "input_assb111_strict30_saturating_v2_seed42locked",
  [string]$CapacityTargetCsv = "Data\assb_capacity_soh_targets\capacity_soh_targets.csv",
  [string]$SolutionNpz = "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz",
  [string]$StateModelDir = "ModelFin_107A",
  [string]$StateEvalDir = "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only",
  [string]$StateEvalNpz = "",
  [string]$CycleTableCsv = "Data\assb_aging_fix1\cycle_table.csv",
  [string]$WorkDir = "Data\assb111_seed42_locked",
  [string]$ModelDir = "ModelFin_111_seed42_locked",
  [string]$EvalDir = "EvalFin_111_seed42_locked_strict30_test70",
  [string]$Device = "cuda",
  [int]$Epochs = 5000,
  [int]$Seed = 42,
  [string]$SOHModelVariant = "saturating_v2",
  [double]$FloorMin = 0.65,
  [double]$FloorMax = 0.85,
  [double]$SOHFloorPrior = 0.72,
  [double]$DamageRateScale = 5e-4,
  [double]$GateGamma = 1.0,
  [double]$ResidualBound = 0.008,
  [double]$SOHNumericMin = 0.60,
  [double]$WFloorPrior = 0.02,
  [double]$WTailGuard = 0.05,
  [double]$LR = 2e-3,
  [double]$WeightDecay = 1e-5,
  [int]$Patience = 600,
  [double]$MinTrainR2ForBest = -1000000000.0,
  [double]$MaxTrainMAEForBest = 1000000000.0,
  [double]$MaxValMAEForBest = 1000000000.0,
  [string]$CandidateTag = "",
  [string]$SelectionMode = "visible_train_val_only",
  [string]$ProtocolTag = "",
  [double]$Dropout = 0.05,
  [int]$CheckpointInterval = 50,
  [double]$EmaDecay = 0.995,
  [int]$TopKCheckpoints = 5,
  [switch]$UseEMA,
  [switch]$TopKCheckpointAvg,
  [switch]$AllowCPU,
  [switch]$SoftFailEval,
  [switch]$RunOverdecayDiagnostics
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

# Prevent old all-cycle/cycle5 summary env vars from contaminating this strict30 run.
Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_SOFT_LABEL_DIR -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_OCP_DIR -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force $WorkDir | Out-Null
New-Item -ItemType Directory -Force $ModelDir | Out-Null
New-Item -ItemType Directory -Force $EvalDir | Out-Null

function Invoke-Step {
  param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Name,
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [object[]]$PyArgs
  )
  Write-Host "`n==== $Name ====" -ForegroundColor Cyan
  if ($null -eq $PyArgs -or $PyArgs.Count -eq 0) { throw "$Name failed: no Python script/arguments were passed." }
  if ($PyArgs.Count -eq 1 -and $PyArgs[0] -is [System.Array]) { $PyArgs = @($PyArgs[0]) }
  $PyArgs = @($PyArgs | ForEach-Object { [string]$_ })
  Write-Host ("python " + ($PyArgs -join " ")) -ForegroundColor DarkGray
  & $PythonExe @PyArgs
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$splitJson = Join-Path $WorkDir "split_manifest.json"
$splitCsv = Join-Path $WorkDir "split_manifest.csv"
$featuresCsv = Join-Path $WorkDir "features_107A_cycle.csv"
$featureSummary = Join-Path $WorkDir "feature_summary.json"
$featureSchema = Join-Path $WorkDir "feature_schema.json"
$featureScaler = Join-Path $WorkDir "feature_scaler.json"
$datasetCsv = Join-Path $WorkDir "dataset_strict30.csv"
$maskedTrainCsv = Join-Path $WorkDir "masked_train_dataset.csv"
$preAudit = Join-Path $WorkDir "leakage_audit_pretrain.json"

Invoke-Step "ASSB111 split manifest" @(
  "scripts\make_assb111_split_manifest.py",
  "--capacity_target_csv", $CapacityTargetCsv,
  "--output_json", $splitJson,
  "--output_csv", $splitCsv
)

$extractArgs = @(
  "scripts\extract_assb111_107A_features.py",
  "--solution_npz", $SolutionNpz,
  "--state_eval_dir", $StateEvalDir,
  "--cycle_table_csv", $CycleTableCsv,
  "--split_manifest_json", $splitJson,
  "--output_csv", $featuresCsv,
  "--output_json", $featureSummary,
  "--cycle_from", "5",
  "--cycle_to", "521"
)
if (-not [string]::IsNullOrWhiteSpace($StateEvalNpz)) { $extractArgs += @("--state_eval_npz", $StateEvalNpz) }
Invoke-Step "ASSB111 107A feature extraction" $extractArgs

Invoke-Step "ASSB111 dataset build" @(
  "scripts\build_assb111_dataset.py",
  "--features_csv", $featuresCsv,
  "--capacity_target_csv", $CapacityTargetCsv,
  "--split_manifest_json", $splitJson,
  "--dataset_csv", $datasetCsv,
  "--masked_train_dataset_csv", $maskedTrainCsv,
  "--schema_json", $featureSchema,
  "--scaler_json", $featureScaler,
  "--audit_json", $preAudit,
  "--feature_mode", "p1_107a_strict",
  "--scaler_scope", "train"
)

Invoke-Step "ASSB111 pretrain leakage audit" @(
  "scripts\audit_assb111_leakage.py",
  "--dataset_csv", $datasetCsv,
  "--split_manifest_json", $splitJson,
  "--feature_mode", "p1_107a_strict",
  "--scaler_json", $featureScaler,
  "--output_json", $preAudit
)

$trainArgs = @(
  "scripts\train_assb111_soh_head.py",
  "--dataset_csv", $datasetCsv,
  "--split_manifest_json", $splitJson,
  "--scaler_json", $featureScaler,
  "--output_model_dir", $ModelDir,
  "--feature_mode", "p1_107a_strict",
  "--device", $Device,
  "--epochs", $Epochs,
  "--seed", $Seed,
  "--lr", $LR,
  "--weight_decay", $WeightDecay,
  "--patience", $Patience,
  "--soh_model_variant", $SOHModelVariant,
  "--floor_min", $FloorMin,
  "--floor_max", $FloorMax,
  "--soh_floor_prior", $SOHFloorPrior,
  "--damage_rate_scale", $DamageRateScale,
  "--gate_gamma", $GateGamma,
  "--residual_bound", $ResidualBound,
  "--soh_numeric_min", $SOHNumericMin,
  "--w_floor_prior", $WFloorPrior,
  "--w_tail_guard", $WTailGuard,
  "--dropout", $Dropout,
  "--min_train_r2_for_best", $MinTrainR2ForBest,
  "--max_train_mae_for_best", $MaxTrainMAEForBest,
  "--max_val_mae_for_best", $MaxValMAEForBest,
  "--checkpoint_interval", $CheckpointInterval,
  "--ema_decay", $EmaDecay,
  "--top_k_checkpoints", $TopKCheckpoints,
  "--selection_mode", $SelectionMode,
  "--best_selection_mode", "seed42_locked_visible_guard",
  "--visible_score_mode", "val_mae_train_guard",
  "--write_training_summary",
  "--no_test_selection",
  "--save_epoch_checkpoints"
)
if ($AllowCPU) { $trainArgs += "--allow_cpu" }
if ($CandidateTag -ne "") { $trainArgs += @("--candidate_tag", $CandidateTag) }
if ($ProtocolTag -ne "") { $trainArgs += @("--protocol_tag", $ProtocolTag) }
if ($Seed -eq 42 -and ($ProtocolTag -match "seed42" -or $InputFile -match "seed42" -or $CandidateTag -ne "")) {
  $trainArgs += @("--seed_locked", "--locked_seed_value", "42")
}
if ($UseEMA) { $trainArgs += "--enable_ema" }
if ($TopKCheckpointAvg) { $trainArgs += "--enable_swa_topk" }
Invoke-Step "ASSB111 SOH head training" $trainArgs

Invoke-Step "ASSB111 post-train leakage audit" @(
  "scripts\audit_assb111_leakage.py",
  "--dataset_csv", $datasetCsv,
  "--split_manifest_json", $splitJson,
  "--feature_mode", "p1_107a_strict",
  "--scaler_json", (Join-Path $ModelDir "feature_scaler.json"),
  "--train_history_csv", (Join-Path $ModelDir "train_history.csv"),
  "--output_json", (Join-Path $ModelDir "leakage_audit.json")
)

Invoke-Step "ASSB111 package manifest" @(
  "scripts\build_ModelFin111_package.py",
  "--model_dir", $ModelDir,
  "--state_model_dir", $StateModelDir,
  "--state_eval_dir", $StateEvalDir,
  "--dataset_csv", $datasetCsv,
  "--features_csv", $featuresCsv,
  "--split_manifest_json", $splitJson,
  "--input_file", $InputFile,
  "--feature_schema_json", (Join-Path $ModelDir "feature_schema.json"),
  "--feature_scaler_json", (Join-Path $ModelDir "feature_scaler.json"),
  "--leakage_audit_json", (Join-Path $ModelDir "leakage_audit.json"),
  "--training_summary_json", (Join-Path $ModelDir "train_summary.json"),
  "--checkpoint_manifest_csv", (Join-Path $ModelDir "checkpoint_manifest.csv"),
  "--selected_model_pt", (Join-Path $ModelDir "selected_model.pt")
)

$evalArgs = @(
  "scripts\evaluate_assb111_five_state.py",
  "--model_dir", $ModelDir,
  "--state_eval_dir", $StateEvalDir,
  "--dataset_csv", $datasetCsv,
  "--split_manifest_json", $splitJson,
  "--output_dir", $EvalDir
)
if (-not [string]::IsNullOrWhiteSpace($StateEvalNpz)) { $evalArgs += @("--state_eval_npz", $StateEvalNpz) }
if ($AllowCPU) { $evalArgs += "--allow_cpu" }
if ($SoftFailEval) { $evalArgs += "--soft_fail" }
Invoke-Step "ASSB111 five-state evaluation" $evalArgs

if ($RunOverdecayDiagnostics) {
  Invoke-Step "ASSB111 SOH overdecay diagnostics" @(
    "scripts\diagnose_assb111_soh_overdecay.py",
    "--pred_csv", (Join-Path $EvalDir "soh_pred_by_cycle.csv"),
    "--output_dir", $EvalDir
  )
}

Write-Host "`nASSB111 strict30 seed42-compatible saturating pipeline completed with guarded checkpoint only." -ForegroundColor Green
Write-Host "Model: $ModelDir"
Write-Host "Eval:  $EvalDir"
