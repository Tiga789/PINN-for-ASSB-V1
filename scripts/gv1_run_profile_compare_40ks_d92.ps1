param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d92",
  [string]$Device = "cuda"
)

Set-Location $ProjectRoot
& .\scripts\gv1_run_profile_compare_d92.ps1 `
  -ProjectRoot $ProjectRoot `
  -Python $Python `
  -ProfileRoot $ProfileRoot `
  -OutRoot $OutRoot `
  -Epochs 1000 `
  -BatchSize 4096 `
  -MaxTimePoints 8192 `
  -PredictionTimePoints 4096 `
  -TimeWindowS 40000 `
  -Lr 0.0008 `
  -Device $Device
