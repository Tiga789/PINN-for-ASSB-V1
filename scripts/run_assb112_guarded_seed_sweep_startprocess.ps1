param(
  [int[]]$Seeds = @(7,42,2026,3407,7890),
  [int]$MaxParallel = 4,
  [string]$Root = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$Device = "cuda",
  [string]$DType = "float32",
  [int]$Epochs = 2500,
  [int]$EvalEvery = 10,
  [int]$PrintEvery = 100,
  [int]$MonitorEverySeconds = 15,
  [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $Root
$dataset = Join-Path $Root "Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv"
$manifest = Join-Path $Root "Data\assb111_seed42locked_repro_c00\split_manifest.json"
$logDir = Join-Path $Root "LogFin_112_v7_softscore_startprocess"
New-Item -ItemType Directory -Force $logDir | Out-Null

Write-Host "ASSB-112 v7 softscore sweep via Start-Process"
Write-Host "Root        = $Root"
Write-Host "Dataset     = $dataset"
Write-Host "Manifest    = $manifest"
Write-Host "Seeds       = $($Seeds -join ',')"
Write-Host "MaxParallel = $MaxParallel"
Write-Host "DType       = $DType"
Write-Host "Epochs      = $Epochs"
Write-Host "LogDir      = $logDir"
Write-Host "Selection   = visible soft-score; hard guards are audit-only"
Write-Host "No Start-Job / Receive-Job is used."

function New-ArgList([int]$Seed, [string]$OutDir) {
  return @(
    (Join-Path $Root "scripts\train_assb111_soh_head.py"),
    "--dataset_csv", $dataset,
    "--split_manifest_json", $manifest,
    "--output_model_dir", $OutDir,
    "--feature_mode", "g4_all_strict",
    "--soh_model_variant", "robust_saturating",
    "--seed", "$Seed",
    "--device", $Device,
    "--epochs", "$Epochs",
    "--lr", "2e-3",
    "--weight_decay", "1e-5",
    "--hidden_dim", "48",
    "--hidden_layers", "2",
    "--dropout", "0.05",
    "--feature_dropout", "0.05",
    "--soh_floor_prior", "0.72",
    "--soh_numeric_min", "0.60",
    "--min_train_r2_for_best", "0.990",
    "--max_train_mae_for_best", "0.0030",
    "--max_val_mae_for_best", "0.0030",
    "--min_val_r2_for_best", "0.80",
    "--min_val_corr_for_best", "0.95",
    "--max_val_bias_for_best", "0.0030",
    "--max_val_tail_bias_for_best", "0.0040",
    "--max_val_slope_mae_for_best", "0.0020",
    "--min_val_range_ratio_for_best", "0.40",
    "--max_val_range_ratio_for_best", "1.80",
    "--max_visible_monotonic_penalty_for_best", "5.0e-5",
    "--selection_strategy", "softscore",
    "--patience", "800",
    "--min_epochs_before_patience", "1200",
    "--eval_every", "$EvalEvery",
    "--print_every", "$PrintEvery",
    "--dtype", $DType,
    "--cuda_matmul_precision", "high",
    "--no_test_selection",
    "--progress_json", "progress.json",
    "--candidate_tag", ("v7_softscore_seed{0}" -f $Seed),
    "--protocol_tag", "ASSB112_v7_softscore_trainval_only"
  )
}

$queue = New-Object System.Collections.Generic.Queue[int]
foreach ($s in $Seeds) { $queue.Enqueue([int]$s) }
$running = @()
$results = @()
$lastProgress = @{}
$lastMonitor = Get-Date "2000-01-01"

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    $seed = $queue.Dequeue()
    $outDir = Join-Path $Root ("ModelFin_112_v7_softscore_seed{0}" -f $seed)
    if ($Clean -and (Test-Path $outDir)) { Remove-Item $outDir -Recurse -Force }
    $stdout = Join-Path $logDir ("seed_{0}.out.log" -f $seed)
    $stderr = Join-Path $logDir ("seed_{0}.err.log" -f $seed)
    Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue
    $args = New-ArgList -Seed $seed -OutDir $outDir
    Write-Host "[START] seed=$seed out=$outDir"
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $running += [pscustomobject]@{ Seed=$seed; Process=$proc; Stdout=$stdout; Stderr=$stderr; OutDir=$outDir }
  }

  Start-Sleep -Seconds 5
  $now = Get-Date
  if (($now - $lastMonitor).TotalSeconds -ge $MonitorEverySeconds) {
    foreach ($item in $running) {
      $progressJson = Join-Path $item.OutDir "progress.json"
      if (Test-Path $progressJson) {
        try {
          $p = Get-Content $progressJson -Raw | ConvertFrom-Json
          $line = "epoch=$($p.epoch) val_mae=$($p.val_mae) val_r2=$($p.val_r2) soft_best=$($p.best_soft_score)@$($p.best_soft_epoch) status=$($p.best_status)"
          $key = [string]$item.Seed
          if ((-not $lastProgress.ContainsKey($key)) -or $lastProgress[$key] -ne $line) {
            Write-Host ("[PROGRESS] seed={0} {1}" -f $item.Seed, $line)
            $lastProgress[$key] = $line
          }
        } catch {}
      } elseif (Test-Path $item.Stdout) {
        $latest = Get-Content $item.Stdout -Tail 20 | Where-Object { $_ -match "^(\[START\]|epoch=|Early stopping|\[OK\])" } | Select-Object -Last 1
        if ($latest) { Write-Host ("[PROGRESS] seed={0} {1}" -f $item.Seed, $latest) }
      }
    }
    $lastMonitor = $now
  }

  $still = @()
  foreach ($item in $running) {
    $proc = $item.Process
    if ($proc.HasExited) {
      $exit = $proc.ExitCode
      Write-Host "[DONE] seed=$($item.Seed) exit=$exit"
      Write-Host "--- stdout tail seed=$($item.Seed) ---"
      if (Test-Path $item.Stdout) { Get-Content $item.Stdout -Tail 25 }
      if ($exit -ne 0) {
        Write-Host "--- stderr tail seed=$($item.Seed) ---"
        if (Test-Path $item.Stderr) { Get-Content $item.Stderr -Tail 80 }
      }
      $results += [pscustomobject]@{ Seed=$item.Seed; ExitCode=$exit; OutDir=$item.OutDir; Stdout=$item.Stdout; Stderr=$item.Stderr }
    } else {
      $still += $item
    }
  }
  $running = $still
}

$summaryCsv = Join-Path $logDir "process_sweep_exit_summary.csv"
$results | Export-Csv $summaryCsv -NoTypeInformation -Encoding UTF8
Write-Host "Exit summary: $summaryCsv"
$bad = $results | Where-Object { $_.ExitCode -ne 0 }
if ($bad.Count -gt 0) {
  Write-Host "[FAILED] some seeds failed. See logs above and $logDir"
  exit 1
}

Write-Host "[OK] all seeds finished. Summarizing..."
& $Python (Join-Path $Root "scripts\summarize_assb112_guarded_seed_sweep.py") `
  --model_prefix (Join-Path $Root "ModelFin_112_v7_softscore_seed") `
  --seeds ($Seeds -join ",") `
  --output_dir (Join-Path $Root "EvalFin_112_v7_softscore_sweep")
