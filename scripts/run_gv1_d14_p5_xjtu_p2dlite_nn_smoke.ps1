param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3",
  [string]$PriorFile = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json",
  [string]$OutputDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5_p2dlite_nn_smoke",
  [int]$Epochs = 120,
  [int]$BatchSize = 2048,
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

$ConfigPath = Join-Path $ProjectRoot "configs\d14_p5_xjtu_p2dlite_nn_smoke_config.json"
$BuildScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5_build_softlabel_manifest.py"
$TrainScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5_train_p2dlite_softlabel_nn.py"
$EvalScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5_eval_p2dlite_softlabel_nn.py"
$VerifyScript = Join-Path $ProjectRoot "scripts\gv1_d14_p5_verify_outputs.py"

if (!(Test-Path $ConfigPath)) { throw "Missing config: $ConfigPath" }
if (!(Test-Path $BuildScript)) { throw "Missing manifest builder: $BuildScript" }
if (!(Test-Path $TrainScript)) { throw "Missing trainer: $TrainScript" }
if (!(Test-Path $EvalScript)) { throw "Missing evaluator: $EvalScript" }
if (!(Test-Path $VerifyScript)) { throw "Missing verifier: $VerifyScript" }
if (!(Test-Path $SoftlabelRoot)) { throw "Missing SoftlabelRoot: $SoftlabelRoot" }
if (!(Test-Path $PriorFile)) { throw "Missing PriorFile: $PriorFile" }

$PythonExe = (Get-Command python -ErrorAction Stop).Source
$CombinedLog = Join-Path $OutputDir "D14_P5_NN_SMOKE_console.log"
Set-Content -Path $CombinedLog -Value "[D14-P5] Log created $(Get-Date -Format o)"

$allowArgs = @()
if ($AllowWarn) { $allowArgs += "--allow_warn" }

$BuildArgs = @(
  $BuildScript,
  "--project_root", $ProjectRoot,
  "--softlabel_root", $SoftlabelRoot,
  "--config", $ConfigPath,
  "--output_dir", $OutputDir,
  "--prior_file", $PriorFile
) + $allowArgs

Write-Host "[D14-P5] Building soft-label NN manifest..."
$exit1 = Run-PythonLogged -PythonExe $PythonExe -Arguments $BuildArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5_MANIFEST_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5_MANIFEST_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "MANIFEST"
Get-Content (Join-Path $OutputDir "D14_P5_MANIFEST_stdout.log") -ErrorAction SilentlyContinue

$ManifestCsv = Join-Path $OutputDir "D14_P5_SOFTLABEL_NN_MANIFEST.csv"

$TrainArgs = @(
  $TrainScript,
  "--project_root", $ProjectRoot,
  "--config", $ConfigPath,
  "--manifest_csv", $ManifestCsv,
  "--output_dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize"
) + $allowArgs

Write-Host "[D14-P5] Training NN smoke model..."
$exit2 = Run-PythonLogged -PythonExe $PythonExe -Arguments $TrainArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5_TRAIN_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5_TRAIN_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "TRAIN"
Get-Content (Join-Path $OutputDir "D14_P5_TRAIN_stdout.log") -ErrorAction SilentlyContinue

$ModelDir = Join-Path $OutputDir "ModelFin_D14_P5_p2dlite_nn_smoke"

$EvalArgs = @(
  $EvalScript,
  "--project_root", $ProjectRoot,
  "--config", $ConfigPath,
  "--manifest_csv", $ManifestCsv,
  "--model_dir", $ModelDir,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5] Evaluating NN smoke model..."
$exit3 = Run-PythonLogged -PythonExe $PythonExe -Arguments $EvalArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5_EVAL_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5_EVAL_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "EVAL"
Get-Content (Join-Path $OutputDir "D14_P5_EVAL_stdout.log") -ErrorAction SilentlyContinue

$VerifyArgs = @(
  $VerifyScript,
  "--output_dir", $OutputDir
) + $allowArgs

Write-Host "[D14-P5] Verifying outputs..."
$exit4 = Run-PythonLogged -PythonExe $PythonExe -Arguments $VerifyArgs `
  -StdoutPath (Join-Path $OutputDir "D14_P5_VERIFY_stdout.log") `
  -StderrPath (Join-Path $OutputDir "D14_P5_VERIFY_stderr.log") `
  -CombinedLogPath $CombinedLog `
  -Label "VERIFY"
Get-Content (Join-Path $OutputDir "D14_P5_VERIFY_stdout.log") -ErrorAction SilentlyContinue

if ($exit1 -ne 0) { throw "D14-P5 manifest step failed with exit code $exit1. See $CombinedLog" }
if ($exit2 -ne 0) { throw "D14-P5 training step failed with exit code $exit2. See $CombinedLog" }
if ($exit3 -ne 0) { throw "D14-P5 eval step failed with exit code $exit3. See $CombinedLog" }
if ($exit4 -ne 0) { throw "D14-P5 verify step failed with exit code $exit4. See $CombinedLog" }

Write-Host "[D14-P5] Done. OutputDir=$OutputDir"
