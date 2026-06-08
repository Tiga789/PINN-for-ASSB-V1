param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PriorFile = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_smoke_p4a",
  [int]$MaxProfilesTotal = 2,
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

$GenerateScript = Join-Path $ProjectRoot "scripts\gv1_generate_xjtu_p2dlite_softlabels.py"
$AuditScript = Join-Path $ProjectRoot "scripts\gv1_audit_xjtu_p2dlite_softlabels.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p4a_verify_outputs.py"
$ConfigPath = Join-Path $ProjectRoot "configs\d14_p4a_xjtu_softlabel_patch_config.json"

if (!(Test-Path $GenerateScript)) { throw "Missing generator script: $GenerateScript" }
if (!(Test-Path $AuditScript)) { throw "Missing audit script: $AuditScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing P4A verify script: $VerifyScript" }
if (!(Test-Path $PriorFile)) { throw "Missing P2Dlite prior file: $PriorFile" }
if (!(Test-Path $ConfigPath)) { throw "Missing P4A config: $ConfigPath" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P4A_SOFTLABEL_SMOKE_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P4A] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$GenerateArgs = @(
  $GenerateScript,
  "--project_root", $ProjectRoot,
  "--cache_root", $CacheRoot,
  "--prior_file", $PriorFile,
  "--config", $ConfigPath,
  "--output_dir", $OutputDir,
  "--max_profiles_total", "$MaxProfilesTotal",
  "--max_points_per_profile", "$MaxPointsPerProfile",
  "--n_r", "$NR"
) + $allowArgs

Write-Host "[D14-P4A] Regenerating bounded/metadata-fixed XJTU P2Dlite soft labels..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $GenerateArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4A_GENERATE_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4A_GENERATE_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "GENERATE"

Get-Content (Join-Path $OutputDir "D14_P4A_GENERATE_stdout.log") -ErrorAction SilentlyContinue

$AuditArgs = @(
  $AuditScript,
  "--output_dir", $OutputDir,
  "--prior_file", $PriorFile,
  "--upper_warn_V", "4.25",
  "--upper_fail_V", "4.35",
  "--lower_warn_V", "2.45",
  "--lower_fail_V", "2.35"
) + $allowArgs

Write-Host "[D14-P4A] Running metadata and voltage-bound audit..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $AuditArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4A_AUDIT_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4A_AUDIT_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "AUDIT"

Get-Content (Join-Path $OutputDir "D14_P4A_AUDIT_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P4A] Verifying P4A outputs..."
$exit3 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P4A_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P4A_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"

Get-Content (Join-Path $OutputDir "D14_P4A_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P4A generator failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P4A audit failed with exit code $exit2. See $CombinedLog" }
if ($exit3 -ne 0) { throw "D14-P4A verify failed with exit code $exit3. See $CombinedLog" }

Write-Host "[D14-P4A] Done. OutputDir=$OutputDir"
