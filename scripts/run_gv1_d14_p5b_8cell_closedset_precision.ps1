param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision",
  [int]$Epochs = 500,
  [int]$BatchSize = 65536,
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

$ConfigPath = Join-Path $ProjectRoot "configs\d14_p5b_xjtu_p2dlite_8cell_closedset_precision_config.json"
$BuildScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5b_build_closedset_manifest.py"
$TrainScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5b_train_closedset_precision.py"
$EvalScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5b_eval_closedset_precision.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5b_verify_outputs.py"

if (!(Test-Path $ConfigPath)) { throw "Missing config: $ConfigPath" }
if (!(Test-Path $BuildScript)) { throw "Missing manifest script: $BuildScript" }
if (!(Test-Path $TrainScript)) { throw "Missing training script: $TrainScript" }
if (!(Test-Path $EvalScript)) { throw "Missing eval script: $EvalScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing verify script: $VerifyScript" }
if (!(Test-Path $SoftlabelRoot)) { throw "Missing SoftlabelRoot: $SoftlabelRoot" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P5B_CLOSEDSET_PRECISION_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P5B-v2] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$BuildArgs = @(
  $BuildScript,
  "--project_root", $ProjectRoot,
  "--softlabel_root", $SoftlabelRoot,
  "--config", $ConfigPath,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5B-v2] Building closed-set manifest..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $BuildArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5B_MANIFEST_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5B_MANIFEST_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "MANIFEST"
Get-Content (Join-Path $OutputDir "D14_P5B_MANIFEST_stdout.log") -ErrorAction SilentlyContinue
if ($exit1 -ne 0) { throw "D14-P5B-v2 manifest failed with exit code $exit1. See $CombinedLog" }

$ManifestCsv = Join-Path $OutputDir "D14_P5B_CLOSEDSET_MANIFEST.csv"

$TrainArgs = @(
  $TrainScript,
  "--project_root", $ProjectRoot,
  "--config", $ConfigPath,
  "--manifest_csv", $ManifestCsv,
  "--output_dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize"
) + $allowArgs

Write-Host "[D14-P5B-v2] Training closed-set precision model..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $TrainArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5B_TRAIN_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5B_TRAIN_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "TRAIN"
Get-Content (Join-Path $OutputDir "D14_P5B_TRAIN_stdout.log") -ErrorAction SilentlyContinue
if ($exit2 -ne 0) { throw "D14-P5B-v2 training failed with exit code $exit2. See $CombinedLog" }

$ModelDir = Join-Path $OutputDir "ModelFin_D14_P5B_8cell_closedset_precision"

$EvalArgs = @(
  $EvalScript,
  "--project_root", $ProjectRoot,
  "--config", $ConfigPath,
  "--manifest_csv", $ManifestCsv,
  "--model_dir", $ModelDir,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5B-v2] Evaluating closed-set precision model..."
$exit3 = Run-PythonLogged -PythonExe $PythonExe -Arguments $EvalArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5B_EVAL_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5B_EVAL_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "EVAL"
Get-Content (Join-Path $OutputDir "D14_P5B_EVAL_stdout.log") -ErrorAction SilentlyContinue
if ($exit3 -ne 0) { throw "D14-P5B-v2 evaluation failed with exit code $exit3. See $CombinedLog" }

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5B-v2] Verifying outputs..."
$exit4 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5B_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5B_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"
Get-Content (Join-Path $OutputDir "D14_P5B_VERIFY_stdout.log") -ErrorAction SilentlyContinue
if ($exit4 -ne 0) { throw "D14-P5B-v2 verify failed with exit code $exit4. See $CombinedLog" }

Write-Host "[D14-P5B-v2] Done. OutputDir=$OutputDir"
