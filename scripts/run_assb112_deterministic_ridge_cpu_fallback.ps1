param(
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [switch]$Clean
)
Set-Location $Root
.\scripts\run_assb112_deterministic_ridge_baseline.ps1 -Device cpu -GpuReserveGB 0 -GpuWorkRepeats 1 -Clean:$Clean
