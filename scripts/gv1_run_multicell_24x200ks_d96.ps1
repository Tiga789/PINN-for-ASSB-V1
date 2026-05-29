# Run this only if 24x40ks passes. It can take significantly longer.
& "$PSScriptRoot\gv1_run_multicell_verify_d96.ps1" `
  -OutRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x200ks_d96" `
  -ProtocolQuota "all" `
  -MaxProfiles 24 `
  -Epochs 1400 `
  -BatchSize 4096 `
  -MaxTimePoints 16384 `
  -PredictionTimePoints 8192 `
  -TimeWindowS 200000 `
  -Lr 0.0007 `
  -ProfileAdaptiveMode auto
