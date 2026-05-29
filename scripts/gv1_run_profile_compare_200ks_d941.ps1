& "$PSScriptRoot\gv1_run_profile_compare_d941.ps1" `
  -OutRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_200ks_d941" `
  -Epochs 1400 `
  -BatchSize 4096 `
  -MaxTimePoints 16384 `
  -PredictionTimePoints 8192 `
  -TimeWindowS 200000 `
  -Lr 0.0007 `
  -ProfileAdaptiveMode auto
