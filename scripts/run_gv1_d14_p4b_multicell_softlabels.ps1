param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PriorFile = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3",
  [int]$MaxPointsPerProfile = 100000,
  [int]$NR = 17,
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

$GenerateScript = Join-Path $ProjectRoot "scripts\gv1_d14_p4b_generate_multicell_softlabels.py"
$AuditScript = Join-Path $ProjectRoot "scripts\gv1_d14_p4b_audit_outputs.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p4b_verify_outputs.py"
$ConfigPath = Join-Path $ProjectRoot "configs\d14_p4b_xjtu_p2dlite_multicell_softlabel_config.json"

if (!(Test-Path $GenerateScript)) { throw "Missing P4B generator script: $GenerateScript" }
if (!(Test-Path $AuditScript)) { throw "Missing P4B audit script: $AuditScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing P4B verify script: $VerifyScript" }
if (!(Test-Path $PriorFile)) { throw "Missing P2Dlite prior file: $PriorFile" }
if (!(Test-Path $ConfigPath)) { throw "Missing P4B config: $ConfigPath" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P4B_SOFTLABEL_MULTIPROFILE_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P4B-v3] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$GenerateArgs = @(
  $GenerateScript,
  "--project_root", $ProjectRoot,
  "--cache_root", $CacheRoot,
  "--prior_file", $PriorFile,
  "--config", $ConfigPath,
  "--output_dir", $OutputDir,
  "--max_points_per_profile", "$MaxPointsPerProfile",
  "--n_r", "$NR"
) + $allowArgs

Write-Host "[D14-P4B-v3] Running batch/protocol fixed controlled multi-profile P2Dlite soft-label generation..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $GenerateArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4B_GENERATE_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4B_GENERATE_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "GENERATE"

Get-Content (Join-Path $OutputDir "D14_P4B_GENERATE_stdout.log") -ErrorAction SilentlyContinue

$AuditArgs = @(
  $AuditScript,
  "--output_dir", $OutputDir,
  "--prior_file", $PriorFile,
  "--config", $ConfigPath
) + $allowArgs

Write-Host "[D14-P4B-v3] Re-auditing generated soft labels..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $AuditArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4B_AUDIT_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4B_AUDIT_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "AUDIT"

Get-Content (Join-Path $OutputDir "D14_P4B_AUDIT_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P4B-v3] Verifying outputs..."
$exit3 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4B_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4B_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"

Get-Content (Join-Path $OutputDir "D14_P4B_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P4B-v3 generator failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P4B-v3 audit failed with exit code $exit2. See $CombinedLog" }
if ($exit3 -ne 0) { throw "D14-P4B-v3 verify failed with exit code $exit3. See $CombinedLog" }

Write-Host "[D14-P4B-v3] Done. OutputDir=$OutputDir"
