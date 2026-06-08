param(
    [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
    [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
    [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit",
    [switch]$StrictCache,
    [switch]$StrictASSB,
    [string]$BaselineFingerprint = ""
)

$ErrorActionPreference = "Stop"

Write-Host "[D14-P0] ProjectRoot = $ProjectRoot"
Write-Host "[D14-P0] CacheRoot   = $CacheRoot"
Write-Host "[D14-P0] OutputDir   = $OutputDir"

if (!(Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}
if (!(Test-Path $CacheRoot)) {
    throw "CacheRoot not found: $CacheRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$LogPath = Join-Path $OutputDir "D14_P0_FREEZE_AUDIT_console.log"

$ScriptPath = Join-Path $ProjectRoot "scripts\gv1_d14_p0_freeze_mainline_audit.py"
if (!(Test-Path $ScriptPath)) {
    throw "Audit script not found. Put scripts\gv1_d14_p0_freeze_mainline_audit.py under ProjectRoot first. Missing: $ScriptPath"
}

$ArgsList = @(
    $ScriptPath,
    "--project-root", $ProjectRoot,
    "--cache-root", $CacheRoot,
    "--output-dir", $OutputDir
)

if ($StrictCache) { $ArgsList += "--strict-cache" }
if ($StrictASSB) { $ArgsList += "--strict-assb" }
if ($BaselineFingerprint -ne "") {
    $ArgsList += @("--baseline-fingerprint", $BaselineFingerprint)
}

Write-Host "[D14-P0] Running audit..."
& python @ArgsList 2>&1 | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

$AuditJson = Join-Path $OutputDir "D14_P0_FREEZE_AUDIT.json"
$Verifier = Join-Path $ProjectRoot "scripts\gv1_d14_p0_verify_outputs.py"
if (Test-Path $Verifier) {
    Write-Host "[D14-P0] Verifying audit JSON..."
    & python $Verifier --audit-json $AuditJson --allow-warn
}

Write-Host "[D14-P0] Output files:"
Write-Host "  $AuditJson"
Write-Host "  $(Join-Path $OutputDir 'D14_P0_FREEZE_AUDIT.md')"
Write-Host "  $(Join-Path $OutputDir 'D14_P0_FILE_FINGERPRINTS.csv')"
Write-Host "  $(Join-Path $OutputDir 'D14_P0_SCORECARD_INDEX.csv')"
Write-Host "  $LogPath"

exit $ExitCode
