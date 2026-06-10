param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SourceSoftlabels = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3",
  [switch]$AllowOverwrite
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$PriorJson = Join-Path $ProjectRoot "configs\P2Dlite_prior_xjtu_lr18650la_rg_v1.json"
$OutSoftlabels = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell"
$AuditOld = Join-Path $CacheRoot "xjtu_d15_p0_radial_gradient_audit_p2dlite_v1"
$AuditRG = Join-Path $CacheRoot "xjtu_d15_p0_radial_gradient_audit_p2dlite_rg_v1"
$CompareDir = Join-Path $CacheRoot "xjtu_d15_p0_radial_gradient_rg_comparison"
$PreflightJson = Join-Path $CacheRoot "xjtu_d15_p0_preflight.json"

Write-Host "[D15-P0] ProjectRoot      = $ProjectRoot"
Write-Host "[D15-P0] CacheRoot        = $CacheRoot"
Write-Host "[D15-P0] SourceSoftlabels = $SourceSoftlabels"
Write-Host "[D15-P0] OutSoftlabels    = $OutSoftlabels"
Write-Host "[D15-P0] PriorJson        = $PriorJson"

python scripts\d15_p0_selftest_rg_solver.py

python scripts\d15_p0_preflight.py `
  --source-dir "$SourceSoftlabels" `
  --prior-json "$PriorJson" `
  --output-softlabels-dir "$OutSoftlabels" `
  --out-json "$PreflightJson"

$OverwriteArg = @()
if ($AllowOverwrite) { $OverwriteArg = @("--allow-overwrite") }

Write-Host "[D15-P0] Step 1/4: audit old P2Dlite v1 radial gradients"
python scripts\d15_p0_radial_gradient_audit.py `
  --source-dir "$SourceSoftlabels" `
  --prior-json "$PriorJson" `
  --out-dir "$AuditOld" `
  @OverwriteArg

Write-Host "[D15-P0] Step 2/4: generate P2Dlite-RG v1 soft labels into new directory"
python scripts\d15_p0_generate_p2dlite_rg_softlabels.py `
  --source-dir "$SourceSoftlabels" `
  --prior-json "$PriorJson" `
  --output-dir "$OutSoftlabels" `
  @OverwriteArg

Write-Host "[D15-P0] Step 3/4: audit new P2Dlite-RG radial gradients"
python scripts\d15_p0_radial_gradient_audit.py `
  --source-dir "$OutSoftlabels" `
  --prior-json "$PriorJson" `
  --out-dir "$AuditRG" `
  @OverwriteArg

Write-Host "[D15-P0] Step 4/4: compare old vs RG radial audit"
python scripts\d15_p0_compare_radial_audits.py `
  --old-audit-dir "$AuditOld" `
  --rg-audit-dir "$AuditRG" `
  --out-dir "$CompareDir" `
  @OverwriteArg

Write-Host "[D15-P0] DONE"
Write-Host "[D15-P0] Old audit:    $AuditOld"
Write-Host "[D15-P0] RG labels:    $OutSoftlabels"
Write-Host "[D15-P0] RG audit:     $AuditRG"
Write-Host "[D15-P0] Comparison:   $CompareDir"
