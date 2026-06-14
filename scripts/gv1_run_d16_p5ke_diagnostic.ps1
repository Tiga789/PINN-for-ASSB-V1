param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$P5KCRunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST\C_train10",
  [string]$P5KDRunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kd_train10_generator_aligned_hard_cbar_ocp_FAST\D_train10_prior_balanced",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke_diagnostic_first_audit",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5ke_diag_mmap_cache",
  [int]$MaxDeepProfiles = 10,
  [int]$SamplePointsPerProfile = 12000,
  [switch]$SkipDeepSoftlabelAudit
)

$ErrorActionPreference = "Stop"

Write-Host "[D16-P5K-E] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-E] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-E] P5K-C RunDir=$P5KCRunDir"
Write-Host "[D16-P5K-E] P5K-D RunDir=$P5KDRunDir"
Write-Host "[D16-P5K-E] OutDir=$OutDir"
Write-Host "[D16-P5K-E] MaxDeepProfiles=$MaxDeepProfiles SamplePointsPerProfile=$SamplePointsPerProfile"

Set-Location $ProjectRoot
python -m py_compile scripts\gv1_d16_p5ke_diagnostic_first_audit.py

$ReportFile = Join-Path $OutDir "D16_P5K_E_DIAGNOSTIC_REPORT.md"
$argsList = @(
  "scripts\gv1_d16_p5ke_diagnostic_first_audit.py",
  "--project-root", $ProjectRoot,
  "--softlabel-root", $SoftlabelRoot,
  "--p5kc-run-dir", $P5KCRunDir,
  "--p5kd-run-dir", $P5KDRunDir,
  "--out-dir", $OutDir,
  "--report-file", $ReportFile,
  "--mmap-cache-root", $MmapCacheRoot,
  "--max-deep-profiles", "$MaxDeepProfiles",
  "--sample-points-per-profile", "$SamplePointsPerProfile"
)
if ($SkipDeepSoftlabelAudit) { $argsList += "--skip-deep-softlabel-audit" }

python @argsList

Write-Host "[D16-P5K-E] DONE"
Write-Host "Report: $ReportFile"
