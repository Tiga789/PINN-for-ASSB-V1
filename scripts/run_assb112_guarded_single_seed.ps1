param(
  [int]$Seed = 7,
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$Device = "cuda",
  [string]$DType = "float32",
  [int]$Epochs = 2000,
  [int]$EvalEvery = 10,
  [int]$PrintEvery = 50,
  [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $Root
$dataset = Join-Path $Root "Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv"
$manifest = Join-Path $Root "Data\assb111_seed42locked_repro_c00\split_manifest.json"
$outDir = Join-Path $Root ("ModelFin_112_v7_softscore_seed{0}" -f $Seed)
$logDir = Join-Path $Root "LogFin_112_v7_single_seed"
New-Item -ItemType Directory -Force $logDir | Out-Null
$logPath = Join-Path $logDir ("seed_{0}.log" -f $Seed)

if ($Clean -and (Test-Path $outDir)) { Remove-Item $outDir -Recurse -Force }

Write-Host "ASSB-112 v7 single-seed run"
Write-Host "Selection = softscore, hard guard audit-only"
Write-Host "Seed      = $Seed"
Write-Host "Epochs    = $Epochs"
Write-Host "OutDir    = $outDir"

& $Python (Join-Path $Root "scripts\train_assb111_soh_head.py") `
  --dataset_csv $dataset `
  --split_manifest_json $manifest `
  --output_model_dir $outDir `
  --feature_mode "g4_all_strict" `
  --soh_model_variant "robust_saturating" `
  --seed $Seed `
  --device $Device `
  --epochs $Epochs `
  --lr 2e-3 `
  --weight_decay 1e-5 `
  --hidden_dim 48 `
  --hidden_layers 2 `
  --dropout 0.05 `
  --feature_dropout 0.05 `
  --soh_floor_prior 0.72 `
  --soh_numeric_min 0.60 `
  --min_train_r2_for_best 0.990 `
  --max_train_mae_for_best 0.0030 `
  --max_val_mae_for_best 0.0030 `
  --min_val_r2_for_best 0.80 `
  --min_val_corr_for_best 0.95 `
  --max_val_bias_for_best 0.0030 `
  --max_val_tail_bias_for_best 0.0040 `
  --max_val_slope_mae_for_best 0.0020 `
  --min_val_range_ratio_for_best 0.40 `
  --max_val_range_ratio_for_best 1.80 `
  --max_visible_monotonic_penalty_for_best 5.0e-5 `
  --selection_strategy softscore `
  --patience 800 `
  --min_epochs_before_patience 1200 `
  --eval_every $EvalEvery `
  --print_every $PrintEvery `
  --dtype $DType `
  --cuda_matmul_precision high `
  --no_test_selection `
  --progress_json "progress.json" `
  --candidate_tag ("v7_softscore_seed{0}" -f $Seed) `
  --protocol_tag "ASSB112_v7_softscore_trainval_only" `
  2>&1 | Tee-Object -FilePath $logPath

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[OK] seed=$Seed finished."
