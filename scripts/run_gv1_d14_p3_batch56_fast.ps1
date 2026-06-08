param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$DataRoot = "E:\XJTU battery dataset",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$P0Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2",
  [string]$P1Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2",
  [string]$P2Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit_fast",
  [string[]]$Batches = @("Batch-5", "Batch-6"),
  [switch]$AllowWarn
)

$ErrorActionPreference = "Stop"

function Quote-Arg([string]$s) {
  return '"' + ($s -replace '"','\"') + '"'
}

function Run-PythonLogged {
  param(
    [string]$PythonExe,
    [string[]]$Arguments,
    [string]$StdoutPath,
    [string]$StderrPath,
    [string]$CombinedLogPath,
    [string]$Label
  )

  if (Test-Path $StdoutPath) { Remove-Item $StdoutPath -Force }
  if (Test-Path $StderrPath) { Remove-Item $StderrPath -Force }

  Add-Content -Path $CombinedLogPath -Value "[$Label] START $(Get-Date -Format o)"
  Add-Content -Path $CombinedLogPath -Value "[$Label] python=$PythonExe"
  Add-Content -Path $CombinedLogPath -Value "[$Label] args=$($Arguments -join ' ')"

  $argLine = ($Arguments | ForEach-Object { Quote-Arg $_ }) -join " "
  $p = Start-Process -FilePath $PythonExe -ArgumentList $argLine -Wait -PassThru -NoNewWindow `
       -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath

  Add-Content -Path $CombinedLogPath -Value "[$Label] EXIT_CODE=$($p.ExitCode)"
  Add-Content -Path $CombinedLogPath -Value "[$Label] STDOUT:"
  if (Test-Path $StdoutPath) { Get-Content $StdoutPath | Add-Content -Path $CombinedLogPath }
  Add-Content -Path $CombinedLogPath -Value "[$Label] STDERR:"
  if (Test-Path $StderrPath) { Get-Content $StderrPath | Add-Content -Path $CombinedLogPath }

  return $p.ExitCode
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$AuditScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3_batch56_fast_feasibility_audit.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3_fast_verify_outputs.py"
$ConfigPath = Join-Path $ProjectRoot "configs\d14_p3_fast_feasibility_config.json"

if (!(Test-Path $AuditScript)) { throw "Missing audit script: $AuditScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing verify script: $VerifyScript" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P3_FAST_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P3 FAST] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$AuditArgs = @(
  $AuditScript,
  "--project_root", $ProjectRoot,
  "--data_root", $DataRoot,
  "--cache_root", $CacheRoot,
  "--output_dir", $OutputDir,
  "--config", $ConfigPath,
  "--p0_dir", $P0Dir,
  "--p1_dir", $P1Dir,
  "--p2_dir", $P2Dir,
  "--batches"
) + $Batches + $allowArgs

Write-Host "[D14-P3 FAST] Running shallow feasibility audit..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $AuditArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P3_FAST_AUDIT_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P3_FAST_AUDIT_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "AUDIT"

Get-Content (Join-Path $OutputDir "D14_P3_FAST_AUDIT_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P3 FAST] Verifying outputs..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P3_FAST_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P3_FAST_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"

Get-Content (Join-Path $OutputDir "D14_P3_FAST_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P3 FAST audit failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P3 FAST verify failed with exit code $exit2. See $CombinedLog" }

Write-Host "[D14-P3 FAST] Done. OutputDir=$OutputDir"
