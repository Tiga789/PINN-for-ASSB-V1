param(
    [string]$PythonExe = "python",
    [string]$Config = ".\configs\d18_s0_s1_fix.json",
    [string]$OutputRoot = "",
    [switch]$NoPlots
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ProjectRoot\RUN_D18_S0_S1_FIX_V2.ps1" -PythonExe $PythonExe -Config $Config -OutputRoot $OutputRoot -NoPlots:$NoPlots
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
