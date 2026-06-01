<#
QJW-2 / PINN-for-ASSB-V1 clean clone reproducibility check.
This script:
  1) clones the GitHub repo into a temporary clean directory,
  2) runs Python compileall for gv1/ and scripts/,
  3) checks current README/mainline markers,
  4) optionally checks whether external XJTU cache directories exist,
  5) writes a JSON/Markdown report.
It does NOT modify your main project and does NOT start training.
#>

param(
    [string]$RepoUrl = "https://github.com/Tiga789/PINN-for-ASSB-V1.git",
    [string]$WorkRoot = "$env:USERPROFILE\Desktop",
    [string]$CloneDirName = "PINN-for-ASSB-V1_cleancheck",
    [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$XJTUCacheRoot = "E:\XJTU battery dataset\_gv1_cache",
    [switch]$ForceRemoveExisting,
    [switch]$SkipExternalCacheCheck
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host "==== $Text ====" -ForegroundColor Cyan
}

function Invoke-LoggedCommand {
    param(
        [string]$CommandLine,
        [string]$LogFile,
        [string]$WorkingDirectory = $PWD.Path
    )
    Write-Host "> $CommandLine" -ForegroundColor DarkGray
    Push-Location $WorkingDirectory
    try {
        cmd.exe /c $CommandLine 2>&1 | Tee-Object -FilePath $LogFile
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
        }
    }
    finally {
        Pop-Location
    }
}

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CheckScript = Join-Path $PackageRoot "check_repo_after_clone.py"
if (-not (Test-Path $CheckScript)) {
    throw "Missing checker script: $CheckScript"
}

$TimeTag = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportRoot = Join-Path $WorkRoot "_qjw_clean_clone_reports"
$ReportDir = Join-Path $ReportRoot "clean_clone_$TimeTag"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$ClonePath = Join-Path $WorkRoot $CloneDirName

Write-Section "Environment"
Write-Host "RepoUrl     = $RepoUrl"
Write-Host "WorkRoot    = $WorkRoot"
Write-Host "ClonePath   = $ClonePath"
Write-Host "PythonExe   = $PythonExe"
Write-Host "ReportDir   = $ReportDir"

if (-not (Test-Path $PythonExe)) {
    throw "PythonExe not found: $PythonExe. Please pass -PythonExe with your actual python path."
}

$GitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($null -eq $GitCmd) {
    throw "git command not found. Please install Git or open a shell where git is available."
}

Write-Section "Clean clone"
if (Test-Path $ClonePath) {
    if ($ForceRemoveExisting) {
        Write-Host "Removing existing clone path: $ClonePath" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $ClonePath
    } else {
        throw "Clone path already exists: $ClonePath. Re-run with -ForceRemoveExisting or choose another -CloneDirName."
    }
}

Invoke-LoggedCommand -CommandLine "git clone --depth 1 $RepoUrl `"$ClonePath`"" -LogFile (Join-Path $ReportDir "01_git_clone.log") -WorkingDirectory $WorkRoot

Write-Section "Git identity"
Invoke-LoggedCommand -CommandLine "git rev-parse HEAD" -LogFile (Join-Path $ReportDir "02_git_head.log") -WorkingDirectory $ClonePath
Invoke-LoggedCommand -CommandLine "git status --short" -LogFile (Join-Path $ReportDir "03_git_status.log") -WorkingDirectory $ClonePath

Write-Section "Python version"
Invoke-LoggedCommand -CommandLine "`"$PythonExe`" --version" -LogFile (Join-Path $ReportDir "04_python_version.log") -WorkingDirectory $ClonePath

Write-Section "Static checker"
$CheckerArgs = @(
    "`"$CheckScript`"",
    "--repo", "`"$ClonePath`"",
    "--report-dir", "`"$ReportDir`"",
    "--python-exe", "`"$PythonExe`""
)
if (-not $SkipExternalCacheCheck) {
    $CheckerArgs += @("--cache-root", "`"$XJTUCacheRoot`"")
}
$CheckerCmd = "`"$PythonExe`" " + ($CheckerArgs -join " ")
Invoke-LoggedCommand -CommandLine $CheckerCmd -LogFile (Join-Path $ReportDir "05_static_checker.log") -WorkingDirectory $ClonePath

Write-Section "Summary"
$SummaryFile = Join-Path $ReportDir "clean_clone_check_report.md"
$JsonFile = Join-Path $ReportDir "clean_clone_check_report.json"
Write-Host "Report JSON: $JsonFile" -ForegroundColor Green
Write-Host "Report MD:   $SummaryFile" -ForegroundColor Green
Write-Host "Clean clone: $ClonePath" -ForegroundColor Green

if (Test-Path $SummaryFile) {
    Write-Host ""
    Get-Content $SummaryFile -TotalCount 80
}

