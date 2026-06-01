# D11-S4 low-voltage tail / low-target correction smoke package

This package prepares and scores a **safe low-voltage tail correction smoke** for the GV1/XJTU line.

It is intentionally named **D11-S4** in this window. It does **not** replace the D9.6 / D9.5.1 mainline.

## Purpose

D13 showed that:

- D12-S3 completed 69/69 runs successfully.
- `metadata_on` increased MAE relative to `metadata_off`, so metadata_on must not be promoted to the mainline.
- The worst errors concentrate in `low_target` / `low_target_le_2p75` segments.

D11-S4 therefore tests whether a **delayed, trend-preserving low-tail objective** can improve low-voltage segments without breaking global voltage trends.

## What this package adds

```text
scripts/gv1_d11_s4_prepare_lowtail_smoke_commands.py
scripts/gv1_d11_s4_scorecard_from_predictions.py
scripts/run_gv1_d11_s4_preflight_check.ps1
scripts/run_gv1_d11_s4_prepare_commands.ps1
scripts/run_gv1_d11_s4_collect_scorecard.ps1
RUN_ORDER_D11_S4.txt
README_D11_S4_LOW_VOLTAGE_TAIL_CORRECTION.md
```

## Experiment design

Default smoke design:

```text
profiles: 6 focused profiles
modes: baseline_d951 / lowtail_mild / lowtail_strong_safe
runs: 6 × 3 = 18
window: 40 ks
epochs: 150
max_time_points: 1024
batch_size: 512
battery-8 policy: B1_2C battery-8 remains excluded
```

Selected focused profiles are chosen from known high-error / low-tail candidates when available:

```text
Batch-1_2C_battery-6
Batch-1_2C_battery-7
Batch-3_R2.5_battery-7
Batch-3_R2.5_battery-8
Batch-4_R3_battery-6
Batch-4_R3_battery-7
```

If a named profile is missing, the script falls back to a balanced non-battery-8 selection from the training-ready profile manifest.

## Run order

From the project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

D:\Anaconda\envs\torchgpu\python.exe -m compileall gv1 scripts

.\scripts\run_gv1_d11_s4_preflight_check.ps1
.\scripts\run_gv1_d11_s4_prepare_commands.ps1
.\scripts\run_gv1_d11_s4_preflight_check.ps1 -AfterPrepare
```

Then run the generated modes:

```powershell
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s4_lowtail_correction_smoke_commands\run_d11_s4_baseline_d951.generated.ps1"
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s4_lowtail_correction_smoke_commands\run_d11_s4_lowtail_mild.generated.ps1"
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s4_lowtail_correction_smoke_commands\run_d11_s4_lowtail_strong_safe.generated.ps1"
```

Collect the scorecard:

```powershell
.\scripts\run_gv1_d11_s4_collect_scorecard.ps1
```

## Outputs

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s4_lowtail_correction_smoke_scorecard
```

Important files:

```text
D11_S4_scorecard_summary.json
D11_S4_run_metrics.csv
D11_S4_segment_metrics.csv
D11_S4_mode_summary.csv
D11_S4_mode_segment_summary.csv
D11_S4_lowtail_comparison.csv
D11_S4_worst_segments.csv
D11_S4_RECOMMENDATION.md
```

## Decision rule

Promote a low-tail candidate only if it satisfies all of the following versus `baseline_d951`:

1. global MAE does not increase materially,
2. global corr does not drop materially,
3. `low_target` / `low_target_le_2p75` MAE improves,
4. no hard-clamp / high-voltage saturation failure appears,
5. charge/discharge segment metrics remain acceptable.

Otherwise, keep the D9.6 / D9.5.1 mainline and treat D11-S4 only as an ablation audit.
