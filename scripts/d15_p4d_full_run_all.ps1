param(
  [int]$Workers = 4,
  [ValidateSet('compressed','uncompressed')][string]$SaveMode = 'uncompressed',
  [int]$MaxProfiles = 0,
  [switch]$RegenerateCompleted,
  [switch]$SkipAudit,
  [switch]$SkipPack,
  [int]$MonitorIntervalSeconds = 10
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

function Quote-Arg([string]$s) {
  if ($null -eq $s) { return '""' }
  return '"' + ($s -replace '"','\"') + '"'
}

$ConfigPath = Join-Path $ProjectRoot 'configs\d15_p4d_full_remaining14_config.json'
$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$ManifestCsv = [string]$config.p4c_replay_manifest_csv
$PriorJson = [string]$config.prior_json
$SoftDir = [string]$config.output_softlabels_dir
$AuditDir = [string]$config.radial_audit_dir
$ScoreDir = [string]$config.scorecard_dir
$ReviewZip = [string]$config.review_zip
$RunDir = Join-Path ([string]$config.cache_root) 'xjtu_d15_p4d_batch56_remaining14_generation_run'
$StatusDir = Join-Path $RunDir 'cell_status'
$LogDir = Join-Path $RunDir 'logs'
$ResourceCsv = Join-Path $RunDir 'D15_P4D_FULL_RESOURCE_MONITOR.csv'
$GpuCsv = Join-Path $RunDir 'D15_P4D_FULL_GPU_MONITOR.csv'
$ProcessCsv = Join-Path $RunDir 'D15_P4D_FULL_PROCESS_RESULTS.csv'
$GenJson = Join-Path $SoftDir 'D15_P4D_BATCH56_REMAINING14_RG_GENERATION_REPORT.json'
$GenCsv = Join-Path $SoftDir 'D15_P4D_BATCH56_REMAINING14_RG_GENERATION_REPORT.csv'
$ScoreJson = Join-Path $ScoreDir 'D15_P4D_FINAL_SCORECARD.json'

New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force $StatusDir | Out-Null
New-Item -ItemType Directory -Force $LogDir | Out-Null
New-Item -ItemType Directory -Force $SoftDir | Out-Null
New-Item -ItemType Directory -Force $ScoreDir | Out-Null

Write-Host '[D15-P4D full] 0/6 selftest' -ForegroundColor Cyan
python scripts\d15_p4d_full_selftest.py
if ($LASTEXITCODE -ne 0) { throw 'D15-P4D full selftest failed' }

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

# Build target cell list from config and manifest readiness.
$targetCells = @($config.target_cells | ForEach-Object { [string]$_ })
if ($MaxProfiles -gt 0 -and $MaxProfiles -lt $targetCells.Count) {
  $targetCells = $targetCells[0..($MaxProfiles-1)]
}
if ($Workers -lt 1) { $Workers = 1 }
if ($Workers -gt $targetCells.Count) { $Workers = $targetCells.Count }

Write-Host "[D15-P4D full] 1/6 generate remaining14 P2Dlite-RG soft labels; cells=$($targetCells.Count); workers=$Workers; save_mode=$SaveMode; regenerate_completed=$RegenerateCompleted" -ForegroundColor Cyan
Write-Host "[D15-P4D full] output: $SoftDir" -ForegroundColor Cyan
Write-Host "[D15-P4D full] resource monitor: $ResourceCsv" -ForegroundColor Cyan

$queue = New-Object System.Collections.Queue
foreach ($c in $targetCells) { $queue.Enqueue([string]$c) }
$active = @()
$completed = @()
$monitorRows = New-Object System.Collections.Generic.List[object]
$gpuRows = New-Object System.Collections.Generic.List[object]
$startTime = Get-Date

function Start-CellProcess([string]$cell) {
  $logBase = Join-Path $LogDir $cell
  $stdout = $logBase + '.out.log'
  $stderr = $logBase + '.err.log'
  $argList = @(
    'scripts\d15_p4d_full_generate_one_rg_softlabel.py',
    '--config', 'configs\d15_p4d_full_remaining14_config.json',
    '--manifest-csv', $ManifestCsv,
    '--cell-id', $cell,
    '--prior-json', $PriorJson,
    '--output-root', $SoftDir,
    '--save-mode', $SaveMode,
    '--status-dir', $StatusDir,
    '--skip-if-complete'
  )
  if ($RegenerateCompleted) { $argList += '--overwrite-existing' }
  $quoted = $argList | ForEach-Object { Quote-Arg ([string]$_) }
  $argsLine = ($quoted -join ' ')
  $p = Start-Process -FilePath 'python' -ArgumentList $argsLine -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
  Write-Host "[D15-P4D FANOUT] START pid=$($p.Id) cell=$cell active=$($active.Count + 1) remaining=$($queue.Count)" -ForegroundColor Green
  return [PSCustomObject]@{ Cell = $cell; Process = $p; Stdout = $stdout; Stderr = $stderr; Start = Get-Date }
}

function Sample-Resource() {
  $now = Get-Date
  try { $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue } catch { $cpu = $null }
  try { $avail = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue } catch { $avail = $null }
  $pyCount = (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'd15_p4d_full_generate_one_rg_softlabel.py' } | Measure-Object).Count
  $monitorRows.Add([PSCustomObject]@{
    time = $now.ToString('s')
    elapsed_seconds = [math]::Round(($now - $startTime).TotalSeconds, 3)
    cpu_total_percent = if ($null -eq $cpu) { '' } else { [math]::Round($cpu, 3) }
    memory_available_mb = if ($null -eq $avail) { '' } else { [math]::Round($avail, 1) }
    active_cell_processes = $active.Count
    python_cell_processes = $pyCount
    completed_count = $completed.Count
  }) | Out-Null
  $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($null -ne $nvsmi) {
    try {
      $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
      if ($line) {
        $parts = $line -split ','
        $gpuRows.Add([PSCustomObject]@{
          time = $now.ToString('s')
          elapsed_seconds = [math]::Round(($now - $startTime).TotalSeconds, 3)
          gpu_util_percent = ($parts[0].Trim())
          gpu_memory_used_mb = ($parts[1].Trim())
        }) | Out-Null
      }
    } catch {}
  }
}

while ($queue.Count -gt 0 -or $active.Count -gt 0) {
  while ($queue.Count -gt 0 -and $active.Count -lt $Workers) {
    $cell = [string]$queue.Dequeue()
    $active += Start-CellProcess $cell
  }
  Start-Sleep -Seconds $MonitorIntervalSeconds
  Sample-Resource
  $still = @()
  foreach ($a in $active) {
    $p = $a.Process
    $p.Refresh()
    if ($p.HasExited) {
      $dur = ((Get-Date) - $a.Start).TotalSeconds
      $completed += [PSCustomObject]@{ Cell=$a.Cell; ExitCode=$p.ExitCode; Seconds=[math]::Round($dur,3); Stdout=$a.Stdout; Stderr=$a.Stderr }
      $color = if ($p.ExitCode -eq 0) { 'Cyan' } else { 'Red' }
      Write-Host "[D15-P4D FANOUT] DONE cell=$($a.Cell) exit=$($p.ExitCode) seconds=$([math]::Round($dur,1)) completed=$($completed.Count)/$($targetCells.Count)" -ForegroundColor $color
      # Echo last line from stdout for immediate visibility.
      if (Test-Path $a.Stdout) {
        $last = Get-Content $a.Stdout -Tail 1 -ErrorAction SilentlyContinue
        if ($last) { Write-Host "  stdout: $last" -ForegroundColor DarkGray }
      }
      if ($p.ExitCode -ne 0 -and (Test-Path $a.Stderr)) {
        Write-Host "  stderr tail:" -ForegroundColor Red
        Get-Content $a.Stderr -Tail 8 -ErrorAction SilentlyContinue | Write-Host -ForegroundColor Red
      }
    } else {
      $still += $a
    }
  }
  $active = $still
}

$monitorRows | Export-Csv $ResourceCsv -NoTypeInformation -Encoding UTF8
$gpuRows | Export-Csv $GpuCsv -NoTypeInformation -Encoding UTF8
$completed | Export-Csv $ProcessCsv -NoTypeInformation -Encoding UTF8

Write-Host '[D15-P4D full] 2/6 collect generation report' -ForegroundColor Cyan
python scripts\d15_p4d_full_collect_generation_report.py `
  --config configs\d15_p4d_full_remaining14_config.json `
  --status-dir $StatusDir `
  --output-softlabels-dir $SoftDir `
  --out-json $GenJson `
  --out-csv $GenCsv
if ($LASTEXITCODE -ne 0) { Write-Host '[D15-P4D full] generation report returned nonzero; continuing to pack diagnostics if possible' -ForegroundColor Yellow }

if (-not $SkipAudit) {
  Write-Host '[D15-P4D full] 3/6 radial audit' -ForegroundColor Cyan
  if (Test-Path $AuditDir) { Remove-Item -Recurse -Force $AuditDir }
  python scripts\d15_p0_radial_gradient_audit.py `
    --source-dir $SoftDir `
    --prior-json $PriorJson `
    --out-dir $AuditDir
  if ($LASTEXITCODE -ne 0) { Write-Host '[D15-P4D full] radial audit returned nonzero; scorecard may be REVIEW/FAIL' -ForegroundColor Yellow }
} else {
  Write-Host '[D15-P4D full] 3/6 radial audit skipped by user' -ForegroundColor Yellow
}

Write-Host '[D15-P4D full] 4/6 collect scorecard' -ForegroundColor Cyan
New-Item -ItemType Directory -Force $ScoreDir | Out-Null
python scripts\d15_p4d_full_collect_scorecard.py `
  --config configs\d15_p4d_full_remaining14_config.json `
  --generation-json $GenJson `
  --audit-dir $AuditDir `
  --resource-csv $ResourceCsv `
  --gpu-csv $GpuCsv `
  --process-csv $ProcessCsv `
  --out-json $ScoreJson
if ($LASTEXITCODE -ne 0) { Write-Host '[D15-P4D full] scorecard returned REVIEW/nonzero; packing review anyway' -ForegroundColor Yellow }

if (-not $SkipPack) {
  Write-Host '[D15-P4D full] 5/6 pack review zip' -ForegroundColor Cyan
  python scripts\d15_p4d_full_pack_review.py `
    --run-dir $RunDir `
    --softlabel-dir $SoftDir `
    --audit-dir $AuditDir `
    --scorecard-dir $ScoreDir `
    --out-zip $ReviewZip
}

Write-Host '[D15-P4D full] DONE' -ForegroundColor Green
Write-Host "Run dir: $RunDir"
Write-Host "Soft labels: $SoftDir"
Write-Host "Audit: $AuditDir"
Write-Host "Scorecard: $ScoreJson"
Write-Host "Review zip: $ReviewZip"
