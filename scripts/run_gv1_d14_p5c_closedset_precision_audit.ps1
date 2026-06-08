param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$P5BOutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision_v2",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5c_8cell_closedset_precision_audit",
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

$ConfigPath = Join-Path $ProjectRoot "configs\d14_p5c_closedset_precision_audit_config.json"
$AuditScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5c_audit_closedset_precision.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5c_verify_audit_outputs.py"

if (!(Test-Path $ConfigPath)) { throw "Missing P5C config: $ConfigPath" }
if (!(Test-Path $AuditScript)) { throw "Missing P5C audit script: $AuditScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing P5C verify script: $VerifyScript" }
if (!(Test-Path $P5BOutputDir)) { throw "Missing P5BOutputDir: $P5BOutputDir" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P5C_CLOSEDSET_PRECISION_AUDIT_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P5C] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$AuditArgs = @(
  $AuditScript,
  "--p5b_output_dir", $P5BOutputDir,
  "--config", $ConfigPath,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5C] Auditing closed-set precision benchmark..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $AuditArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5C_AUDIT_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5C_AUDIT_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "AUDIT"
Get-Content (Join-Path $OutputDir "D14_P5C_AUDIT_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5C] Verifying audit outputs..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5C_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5C_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"
Get-Content (Join-Path $OutputDir "D14_P5C_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P5C audit failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P5C verify failed with exit code $exit2. See $CombinedLog" }

Write-Host "[D14-P5C] Done. OutputDir=$OutputDir"
