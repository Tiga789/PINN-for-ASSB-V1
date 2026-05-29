& "$PSScriptRoot\gv1_run_profile_compare_d94.ps1" `
  -OutRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d94" `
  -Epochs 1200 `
  -BatchSize 4096 `
  -MaxTimePoints 8192 `
  -PredictionTimePoints 4096 `
  -TimeWindowS 40000 `
  -Lr 0.0007 `
  -Device cuda `
  -ProfileAdaptiveMode auto
