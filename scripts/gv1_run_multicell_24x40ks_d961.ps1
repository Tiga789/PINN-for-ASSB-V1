# D9.6.1 verification: all 24 profiles, 40ks. Auto mode only changes late 2C profiles.
& "$PSScriptRoot\gv1_run_multicell_verify_d961.ps1" `
  -OutRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d961" `
  -ProtocolQuota "all" `
  -MaxProfiles 24 `
  -Epochs 1000 `
  -BatchSize 4096 `
  -MaxTimePoints 8192 `
  -PredictionTimePoints 4096 `
  -TimeWindowS 40000 `
  -Lr 0.0007 `
  -ProfileAdaptiveMode auto
