param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke2_provenance_baseline_only_audit",
  [string]$Models = "P5K-C,P5K-D",
  [int]$LimitProfiles = 0,
  [int]$ChunkSize = 200000,
  [int]$SampleStride = 1
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5K-E2] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D16-P5K-E2] CacheRoot=$CacheRoot" -ForegroundColor Cyan
Write-Host "[D16-P5K-E2] SoftlabelRoot=$SoftlabelRoot" -ForegroundColor Cyan
Write-Host "[D16-P5K-E2] OutDir=$OutDir" -ForegroundColor Cyan
Write-Host "[D16-P5K-E2] Models=$Models LimitProfiles=$LimitProfiles ChunkSize=$ChunkSize SampleStride=$SampleStride" -ForegroundColor Cyan

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path "scripts\gv1_d16_p5ke2_provenance_baseline_audit.py")) { throw "Missing scripts\gv1_d16_p5ke2_provenance_baseline_audit.py" }

python -m py_compile scripts\gv1_d16_p5ke2_provenance_baseline_audit.py

python scripts\gv1_d16_p5ke2_provenance_baseline_audit.py `
  --project-root "$ProjectRoot" `
  --cache-root "$CacheRoot" `
  --softlabel-root "$SoftlabelRoot" `
  --out-dir "$OutDir" `
  --models "$Models" `
  --limit-profiles $LimitProfiles `
  --chunk-size $ChunkSize `
  --sample-stride $SampleStride

Write-Host "[D16-P5K-E2] DONE" -ForegroundColor Green
Write-Host "Report: $OutDir\D16_P5K_E2_PROVENANCE_BASELINE_AUDIT_REPORT.md" -ForegroundColor Yellow
