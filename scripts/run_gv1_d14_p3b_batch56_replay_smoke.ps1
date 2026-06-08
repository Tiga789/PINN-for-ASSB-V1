param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$DataRoot = "E:\XJTU battery dataset",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$P3FastDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit_fast",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3b_batch56_replay_smoke",
  [int]$FilesPerBatch = 1,
  [int]$MaxSubrecordsPerFile = 30,
  [int]$MaxTotalPointsPerProfile = 120000,
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

$BuildScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3b_build_batch56_replay_smoke.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p3b_verify_outputs.py"
$ConfigPath = Join-Path $ProjectRoot "configs\d14_p3b_batch56_replay_smoke_config.json"

if (!(Test-Path $BuildScript)) { throw "Missing build script: $BuildScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing verify script: $VerifyScript" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P3B_REPLAY_SMOKE_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P3B] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$BuildArgs = @(
  $BuildScript,
  "--project_root", $ProjectRoot,
  "--data_root", $DataRoot,
  "--cache_root", $CacheRoot,
  "--output_dir", $OutputDir,
  "--config", $ConfigPath,
  "--p3_fast_dir", $P3FastDir,
  "--files_per_batch", "$FilesPerBatch",
  "--max_subrecords_per_file", "$MaxSubrecordsPerFile",
  "--max_total_points_per_profile", "$MaxTotalPointsPerProfile"
) + $allowArgs

Write-Host "[D14-P3B] Running controlled Batch-5/6 replay-profile smoke..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $BuildArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P3B_AUDIT_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P3B_AUDIT_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "BUILD"

Get-Content (Join-Path $OutputDir "D14_P3B_AUDIT_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P3B] Verifying outputs..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P3B_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P3B_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"

Get-Content (Join-Path $OutputDir "D14_P3B_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P3B build failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P3B verify failed with exit code $exit2. See $CombinedLog" }

Write-Host "[D14-P3B] Done. OutputDir=$OutputDir"
