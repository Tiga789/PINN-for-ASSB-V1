param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$DataRoot = "E:\XJTU battery dataset",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$P0Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2",
  [string]$P1Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2",
  [string]$P2Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit",
  [string[]]$Batches = @("Batch-5", "Batch-6"),
  [int]$MaxFilesToInspect = 0,
  [switch]$AllowWarn
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Avoid PowerShell 5 treating Python warning stderr records as terminating
# NativeCommandError when streamed through Tee-Object. We still preserve the
# real Python exit code and fail on nonzero exit codes below.
$env:PYTHONWARNINGS = "ignore:Could not infer format:UserWarning"

$AuditScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3_batch56_feasibility_audit.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3_verify_outputs.py"
$ConfigPath = Join-Path $ProjectRoot "configs\d14_p3_batch56_feasibility_config.json"

if (!(Test-Path $AuditScript)) {
  throw "Missing audit script: $AuditScript"
}
if (!(Test-Path $VerifyScript)) {
  throw "Missing verify script: $VerifyScript"
}

$batchArgs = @()
foreach ($b in $Batches) { $batchArgs += $b }

$allowArg = @()
if ($AllowWarn) { $allowArg += "--allow_warn" }

Write-Host "[D14-P3] Running Batch-5/6 feasibility audit..."
$oldEapForPython = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& python $AuditScript `
  --project_root $ProjectRoot `
  --data_root $DataRoot `
  --cache_root $CacheRoot `
  --output_dir $OutputDir `
  --config $ConfigPath `
  --p0_dir $P0Dir `
  --p1_dir $P1Dir `
  --p2_dir $P2Dir `
  --batches $batchArgs `
  --max_files_to_inspect $MaxFilesToInspect `
  @allowArg 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "D14_P3_BATCH56_FEASIBILITY_AUDIT_console.log")
$exit1 = $LASTEXITCODE
$ErrorActionPreference = $oldEapForPython
if ($exit1 -ne 0) {
  Write-Host "[D14-P3] Audit returned exit code $exit1. Running verifier anyway to preserve diagnostics..."
}

Write-Host "[D14-P3] Verifying outputs..."
$oldEapForVerify = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& python $VerifyScript `
  --output_dir $OutputDir `
  @allowArg 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "D14_P3_VERIFY_console.log")
$exit2 = $LASTEXITCODE
$ErrorActionPreference = $oldEapForVerify
if ($exit1 -ne 0) { exit $exit1 }
if ($exit2 -ne 0) { exit $exit2 }

Write-Host "[D14-P3] Done. OutputDir=$OutputDir"
