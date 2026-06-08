param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5_p2dlite_nn_smoke",
  [switch]$RepairOnly,
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

if (!(Test-Path $OutputDir)) { throw "Missing D14-P5 OutputDir: $OutputDir" }

$ConfigPath = Join-Path $ProjectRoot "configs\d14_p5_xjtu_p2dlite_nn_smoke_config.json"
$ManifestCsv = Join-Path $OutputDir "D14_P5_SOFTLABEL_NN_MANIFEST.csv"
$ModelDir = Join-Path $OutputDir "ModelFin_D14_P5_p2dlite_nn_smoke"
$EvalScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5a_eval_p2dlite_softlabel_nn.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5_verify_outputs.py"

if (!(Test-Path $ConfigPath)) { throw "Missing P5 config: $ConfigPath" }
if (!(Test-Path $ManifestCsv)) { throw "Missing P5 manifest: $ManifestCsv" }
if (!(Test-Path $ModelDir)) { throw "Missing P5 model dir: $ModelDir" }
if (!(Test-Path $EvalScript)) { throw "Missing P5A eval script: $EvalScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing P5A verify script: $VerifyScript" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P5A_EVAL_VERIFY_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P5A] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$repairArgs = @()
if ($RepairOnly) { $repairArgs += "--repair_only" }

$EvalArgs = @(
  $EvalScript,
  "--project_root", $ProjectRoot,
  "--config", $ConfigPath,
  "--manifest_csv", $ManifestCsv,
  "--model_dir", $ModelDir,
  "--output_dir", $OutputDir
) + $repairArgs + $allowArgs

Write-Host "[D14-P5A] Running eval/report repair..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $EvalArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5A_EVAL_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5A_EVAL_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "EVAL_P5A"
Get-Content (Join-Path $OutputDir "D14_P5A_EVAL_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5A] Verifying repaired outputs..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5A_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5A_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY_P5A"
Get-Content (Join-Path $OutputDir "D14_P5A_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P5A eval/report repair failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P5A verify failed with exit code $exit2. See $CombinedLog" }

Write-Host "[D14-P5A] Done. OutputDir=$OutputDir"
