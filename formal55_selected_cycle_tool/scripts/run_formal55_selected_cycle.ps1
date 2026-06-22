param(
    [string]$RequestJson = "",
    [string]$ProjectRoot = "",
    [string]$FormalRoot = "",
    [string]$CacheRoot = "",
    [string]$DeployModelRoot = "",
    [string]$OutputRoot = "",
    [switch]$NoShow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolRoot = Split-Path -Parent $ScriptRoot
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $ToolRoot
}
if ([string]::IsNullOrWhiteSpace($RequestJson)) {
    $RequestJson = Join-Path $ToolRoot "configs\selected_cycle_request.json"
}
if (-not (Test-Path -LiteralPath $RequestJson -PathType Leaf)) {
    throw "Request JSON not found: $RequestJson"
}

$PythonScript = Join-Path $ScriptRoot "formal55_selected_cycle_infer_plot.py"
$ArgsList = @(
    $PythonScript,
    "--request-json", $RequestJson,
    "--project-root", $ProjectRoot
)
if (-not [string]::IsNullOrWhiteSpace($FormalRoot)) { $ArgsList += @("--formal-root", $FormalRoot) }
if (-not [string]::IsNullOrWhiteSpace($CacheRoot)) { $ArgsList += @("--cache-root", $CacheRoot) }
if (-not [string]::IsNullOrWhiteSpace($DeployModelRoot)) { $ArgsList += @("--deploy-model-root", $DeployModelRoot) }
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) { $ArgsList += @("--output-root", $OutputRoot) }
if ($NoShow) { $ArgsList += "--no-show" }

Write-Host "[FORMAL55 selected-cycle] Preflight..."
Write-Host "[FORMAL55 selected-cycle] ProjectRoot: $ProjectRoot"
Write-Host "[FORMAL55 selected-cycle] RequestJson: $RequestJson"
python --version

& python @ArgsList
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    throw "FORMAL55 selected-cycle inference failed with exit code $ExitCode"
}
