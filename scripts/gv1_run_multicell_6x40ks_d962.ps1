$root = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_6x40ks_d962"
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_verify_d962.ps1 `
  -OutRoot $root `
  -ProtocolQuota "2C:2,R2.5:2,R3:2" `
  -MaxProfiles 6 `
  -Epochs 1000 `
  -MaxTimePoints 8192 `
  -PredictionTimePoints 4096 `
  -TimeWindowS 40000 `
  -Lr 0.0007 `
  -ProfileAdaptiveMode auto
