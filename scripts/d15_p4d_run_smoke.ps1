param(
  [switch]$AllowOverwrite,
  [int]$Workers = 4,
  [int]$MaxTimePoints = 300000,
  [ValidateSet('compressed','uncompressed')][string]$SaveMode = 'uncompressed',
  [int]$MaxProfiles = 0,
  [switch]$SkipCudaSmoke
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

function Quote-Arg([string]$s) {
  if ($null -eq $s) { return '""' }
  return '"' + ($s -replace '"','\"') + '"'
}

$ConfigPath = Join-Path $ProjectRoot 'configs\d15_p4d_smoke_config.json'
$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$RunDir = [string]$config.output_dir
$ReviewZip = [string]$config.review_zip
$ManifestCsv = [string]$config.p4c_replay_manifest_csv
$PriorJson = [string]$config.prior_json
$interval = [int]$config.resource_monitor.interval_seconds
if ($interval -lt 2) { $interval = 5 }

if ((Test-Path $RunDir) -and ((Get-ChildItem $RunDir -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)) {
  if (-not $AllowOverwrite) {
    throw "Output directory exists and is not empty: $RunDir. Use -AllowOverwrite for a deliberate rerun."
  }
  Remove-Item -Recurse -Force $RunDir
}
New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $RunDir 'logs') | Out-Null

Write-Host '[D15-P4D-smoke] 0/5 selftest' -ForegroundColor Cyan
python scripts\d15_p4d_selftest.py
if ($LASTEXITCODE -ne 0) { throw 'D15-P4D selftest failed' }

# Limit BLAS oversubscription so process fanout is the main parallelism.
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

if (-not $SkipCudaSmoke) {
  Write-Host '[D15-P4D-smoke] 1/5 CUDA smoke' -ForegroundColor Cyan
  $cudaOut = Join-Path $RunDir 'D15_P4D_CUDA_SMOKE_REPORT.json'
  $cudaSecs = [double]$config.resource_monitor.cuda_smoke_seconds
  $matSize = [int]$config.resource_monitor.cuda_matrix_size
  python scripts\d15_p4d_cuda_smoke.py --out-json $cudaOut --seconds $cudaSecs --matrix-size $matSize
} else {
  Write-Host '[D15-P4D-smoke] 1/5 CUDA smoke skipped by user' -ForegroundColor Yellow
}

$cells = @($config.smoke_cells)
if ($MaxProfiles -gt 0 -and $MaxProfiles -lt $cells.Count) {
  $cells = $cells[0..($MaxProfiles-1)]
}
if ($Workers -lt 1) { $Workers = 1 }
if ($Workers -gt $cells.Count) { $Workers = $cells.Count }

Write-Host "[D15-P4D-smoke] 2/5 partial soft-label generation fanout; cells=$($cells.Count); workers=$Workers; max_time_points=$MaxTimePoints; save_mode=$SaveMode" -ForegroundColor Cyan

$queue = New-Object System.Collections.Queue
foreach ($c in $cells) { $queue.Enqueue([string]$c) }
$active = @()
$completed = @()
$monitorRows = New-Object System.Collections.Generic.List[object]
$gpuRows = New-Object System.Collections.Generic.List[object]
$startTime = Get-Date

function Start-CellProcess([string]$cell) {
  $logBase = Join-Path $RunDir ('logs\' + $cell)
  $stdout = $logBase + '.out.log'
  $stderr = $logBase + '.err.log'
  $argList = @(
    'scripts\d15_p4d_generate_one_smoke_profile.py',
    '--config', 'configs\d15_p4d_smoke_config.json',
    '--manifest-csv', $ManifestCsv,
    '--cell-id', $cell,
    '--prior-json', $PriorJson,
    '--output-root', $RunDir,
    '--max-time-points', [string]$MaxTimePoints,
    '--save-mode', $SaveMode
  ) | ForEach-Object { Quote-Arg ([string]$_) }
  $argsLine = ($argList -join ' ')
  $p = Start-Process -FilePath 'python' -ArgumentList $argsLine -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
  Write-Host "[D15-P4D FANOUT] START pid=$($p.Id) cell=$cell active=$($active.Count + 1) remaining=$($queue.Count)" -ForegroundColor Green
  return [PSCustomObject]@{ Cell = $cell; Process = $p; Stdout = $stdout; Stderr = $stderr; Start = Get-Date }
}

function Sample-Resource() {
  $now = Get-Date
  try { $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue } catch { $cpu = $null }
  try { $avail = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue } catch { $avail = $null }
  $pyCount = (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'd15_p4d_generate_one_smoke_profile.py' } | Measure-Object).Count
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
  Start-Sleep -Seconds $interval
  Sample-Resource
  $still = @()
  foreach ($a in $active) {
    $p = $a.Process
    $p.Refresh()
    if ($p.HasExited) {
      $dur = ((Get-Date) - $a.Start).TotalSeconds
      $completed += [PSCustomObject]@{ Cell=$a.Cell; ExitCode=$p.ExitCode; Seconds=[math]::Round($dur,3); Stdout=$a.Stdout; Stderr=$a.Stderr }
      $color = if ($p.ExitCode -eq 0) { 'Cyan' } else { 'Red' }
      Write-Host "[D15-P4D FANOUT] DONE cell=$($a.Cell) exit=$($p.ExitCode) seconds=$([math]::Round($dur,1)) completed=$($completed.Count)/$($cells.Count)" -ForegroundColor $color
    } else {
      $still += $a
    }
  }
  $active = $still
}

$monitorCsv = Join-Path $RunDir 'D15_P4D_SMOKE_RESOURCE_MONITOR.csv'
$gpuCsv = Join-Path $RunDir 'D15_P4D_SMOKE_GPU_MONITOR.csv'
$completedCsv = Join-Path $RunDir 'D15_P4D_SMOKE_PROCESS_RESULTS.csv'
$monitorRows | Export-Csv $monitorCsv -NoTypeInformation -Encoding UTF8
$gpuRows | Export-Csv $gpuCsv -NoTypeInformation -Encoding UTF8
$completed | Export-Csv $completedCsv -NoTypeInformation -Encoding UTF8

Write-Host '[D15-P4D-smoke] 3/5 collect scorecard' -ForegroundColor Cyan
$scoreJson = Join-Path $RunDir 'D15_P4D_SMOKE_FINAL_SCORECARD.json'
python scripts\d15_p4d_collect_smoke_scorecard.py --run-dir $RunDir --config configs\d15_p4d_smoke_config.json --out-json $scoreJson
if ($LASTEXITCODE -ne 0) { Write-Host '[D15-P4D-smoke] scorecard returned REVIEW/nonzero; packing review anyway' -ForegroundColor Yellow }

Write-Host '[D15-P4D-smoke] 4/5 pack review zip' -ForegroundColor Cyan
python scripts\d15_p4d_pack_review.py --run-dir $RunDir --out-zip $ReviewZip

Write-Host '[D15-P4D-smoke] DONE' -ForegroundColor Green
Write-Host "Run dir: $RunDir"
Write-Host "Review zip: $ReviewZip"
Write-Host "Scorecard: $scoreJson"
