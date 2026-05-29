# Run only after the D9.6.1 borderline battery-8 / late-2C repair checks pass.
& "$PSScriptRoot\gv1_run_multicell_verify_d961.ps1" `
  -OutRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x200ks_d961" `
  -ProtocolQuota "all" `
  -MaxProfiles 24 `
  -Epochs 1400 `
  -BatchSize 4096 `
  -MaxTimePoints 16384 `
  -PredictionTimePoints 8192 `
  -TimeWindowS 200000 `
  -Lr 0.00055 `
  -ProfileAdaptiveMode auto
