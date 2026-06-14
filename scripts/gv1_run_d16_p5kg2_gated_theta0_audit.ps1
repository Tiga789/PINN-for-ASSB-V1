param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$G1ByProfile = "",
  [string]$OutDir = "",
  [double]$TolerateEvalMae = 0.002,
  [double]$TolerateEvalR2 = 0.02,
  [switch]$AllowOverwrite
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $CacheRoot "xjtu_d16_p5kg2_gated_theta0_adapter_audit"
}

Write-Host "[D16-P5K-G2 gated theta0 audit] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G2 gated theta0 audit] CacheRoot=$CacheRoot"
Write-Host "[D16-P5K-G2 gated theta0 audit] G1ByProfile=$G1ByProfile"
Write-Host "[D16-P5K-G2 gated theta0 audit] OutDir=$OutDir"

if (-not (Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $OutDir | Out-Null

$pyArgs = @(
  "scripts\gv1_d16_p5kg2_gated_theta0_adapter_audit.py",
  "--cache-root", $CacheRoot,
  "--out-dir", $OutDir,
  "--tolerate-eval-mae", "$TolerateEvalMae",
  "--tolerate-eval-r2", "$TolerateEvalR2"
)
if (-not [string]::IsNullOrWhiteSpace($G1ByProfile)) { $pyArgs += @("--g1-by-profile", $G1ByProfile) }
if ($AllowOverwrite) { $pyArgs += @("--allow-overwrite") }

python @pyArgs
$ec = $LASTEXITCODE
if ($ec -ne 0) {
  Write-Host "[D16-P5K-G2 gated theta0 audit] Python exited with code $ec"
  exit $ec
}

Write-Host "[D16-P5K-G2 gated theta0 audit] DONE"
Write-Host "Report: $(Join-Path $OutDir 'D16_P5KG2_GATED_THETA0_ADAPTER_AUDIT_REPORT.md')"
