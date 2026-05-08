# Optional check script. Run from project root.
$ErrorActionPreference = "Stop"
$targets = @(
  ".\util\init_pinn.py",
  ".\util\_losses.py",
  ".\util\_rescale.py",
  ".\util\spm_assb_train_discharge.py",
  ".\evaluate_assb_pinn_vs_softlabels.py"
)
Write-Host "[CHECK] Wrapper traces:" -ForegroundColor Cyan
Select-String -Path $targets -Pattern "__pre102|Run the installer script|_load_legacy_module|ModelFin_102 wrapper" -ErrorAction SilentlyContinue
Write-Host "[CHECK] Expected: no output above." -ForegroundColor Green
Write-Host "[CHECK] Required Model102 markers:" -ForegroundColor Cyan
Select-String -Path ".\util\spm_assb_train_discharge.py", ".\evaluate_assb_pinn_vs_softlabels.py", ".\input_assb_cycles5to522_v4_continuous_ID102" -Pattern "cycle5-522_v1|ModelFin_102|soft_lable_cycle5-522_v1|soft_labels_only|t_global_s|cycle_id" -ErrorAction SilentlyContinue
