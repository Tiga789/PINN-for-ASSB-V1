[CmdletBinding(PositionalBinding=$false)]
param(
  [switch]$AllowOverwrite,
  [string]$G1ByProfile = "",
  [string]$G0ByProfile = "",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg3_theta0_adapter_v2_audit"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Location).Path

if ($OutDir -eq "\" -or $OutDir.Trim() -eq "") {
  throw "Invalid OutDir='$OutDir'. Do not append a trailing backslash after the command. Use PowerShell backtick (`) for line continuation, not '\'."
}

Write-Host "[D16-P5K-G3 v3] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G3 v3] OutDir=$OutDir"
if ($G1ByProfile -ne "") { Write-Host "[D16-P5K-G3 v3] G1ByProfile=$G1ByProfile" }
if ($G0ByProfile -ne "") { Write-Host "[D16-P5K-G3 v3] G0ByProfile=$G0ByProfile" }

$py = Join-Path $ProjectRoot "scripts\gv1_d16_p5kg3_theta0_adapter_v2_audit.py"
if (-not (Test-Path $py)) { throw "Missing Python script: $py" }

$argsList = @($py, "--out-dir", $OutDir)
if ($AllowOverwrite) { $argsList += "--allow-overwrite" }
if ($G1ByProfile -ne "") { $argsList += @("--g1-by-profile", $G1ByProfile) }
if ($G0ByProfile -ne "") { $argsList += @("--g0-by-profile", $G0ByProfile) }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "P5K-G3 v3 audit failed with exit code $LASTEXITCODE" }

Write-Host "[D16-P5K-G3 v3] DONE"
Write-Host "Report: $OutDir\D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md"
