# D12 Battery-8 Data Diagnosis Package

This package diagnoses why `Batch-1_2C_battery-8` behaves as an outlier in the GV1/XJTU voltage inversion workflow.

It does **not** train a model and does **not** modify the D9.6/D9.5.1 mainline. It reads existing replay-profile NPZ files and compares battery-8 against other Batch-1 2C peer cells.

## What it checks

1. **Data acquisition / preprocessing clues**
   - non-monotonic time,
   - NaN in current/voltage/time,
   - large voltage jumps,
   - large current jumps,
   - large temperature jumps,
   - voltage outside expected range.

2. **Battery behavior clues**
   - discharge capacity / charge capacity outlier,
   - unusual duration,
   - unusual rest fraction,
   - unusual temperature rise,
   - unusual cycle count or energy.

3. **Model-boundary clues**
   - unusual low-voltage or high-voltage fraction,
   - unusual discharge/rest composition,
   - profile regime differs from Batch-1 2C peers.

## Files added

```text
scripts/gv1_d12_battery8_data_diagnosis.py
scripts/gv1_run_d12_battery8_data_diagnosis.ps1
README_D12_BATTERY8_DATA_DIAGNOSIS.md
install_manifest.json
```

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

$py = "D:\Anaconda\envs\torchgpu\python.exe"
$proj = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$cache = "E:\XJTU battery dataset\_gv1_cache"

.\scripts\gv1_run_d12_battery8_data_diagnosis.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -TargetProfile "Batch-1_2C_battery-8" `
  -MakePlots `
  -Clean
```

If matplotlib is not available, remove `-MakePlots`.

## Output

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_battery8_data_diagnosis
```

Important output files:

```text
D12_B8_diagnostic_summary.json
D12_B8_RECOMMENDATION.md
D12_B8_profile_peer_summary.csv
D12_B8_robust_peer_outlier_scores.csv
D12_B8_target_anomaly_events.csv
D12_B8_target_segment_summary.csv
D12_B8_target_cycle_summary.csv
D12_B8_soh_label_rows_if_available.csv
```

## How to interpret

- If `data_acquisition_flags` are strong, inspect the original `.mat` / standard parquet around the flagged indices.
- If data flags are weak but battery behavior/model-boundary flags are strong, keep battery-8 as a real outlier/special regime.
- If no flags are strong, compare model residuals and voltage trace plots before deciding whether the outlier is due to model expressiveness.
