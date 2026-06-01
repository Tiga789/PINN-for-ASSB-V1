# GV1 D13 segment/protocol diagnosis package

## Purpose

This package performs the next-step D13 diagnosis after D12-S3.
It is **analysis-only** and does **not** launch training.

It reads existing outputs from:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s3_metadata_ablation_scorecard
```

and writes:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d13_segment_protocol_diagnosis
```

## Files

```text
scripts/gv1_d13_segment_protocol_diagnosis.py
scripts/run_gv1_d13_preflight_check.ps1
scripts/run_gv1_d13_segment_protocol_diagnosis.ps1
README_D13_SEGMENT_PROTOCOL_DIAGNOSIS.md
RUN_ORDER_D13.txt
```

## Run order

From project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\run_gv1_d13_preflight_check.ps1
.\scripts\run_gv1_d13_segment_protocol_diagnosis.ps1
```

## Expected outputs

```text
D13_segment_protocol_summary.json
D13_run_metrics.csv
D13_segment_metrics.csv
D13_mode_summary.csv
D13_protocol_summary.csv
D13_mode_protocol_summary.csv
D13_mode_protocol_segment_summary.csv
D13_charge_discharge_summary.csv
D13_voltage_tail_summary.csv
D13_time_drift_summary.csv
D13_worst_runs_by_mae.csv
D13_worst_segments_by_mae.csv
D13_profile_summary.csv
D13_RECOMMENDATION.md
```

## Interpretation

Use `D13_RECOMMENDATION.md` first.  It should answer:

1. Whether `metadata_on` should stay stopped after D12-S3.
2. Which protocol has the largest average voltage error.
3. Whether charge, discharge, low-voltage tail, high-voltage tail, or late-time drift dominates.
4. Whether D14 should focus on protocol-specific adapters, low-tail correction, or higher-fidelity correction.

## Boundary

- Do not unflag `B1_2C battery-8`.
- Do not promote `metadata_on` to mainline.
- Do not modify D9.6/D9.5.1 mainline during this diagnosis.
- Do not launch 24-profile 200ks training from this package.
