# D14-P3B Batch-5/6 Controlled Replay-Profile Build Smoke

## Position

D14-P3B follows D14-P3 FAST. P3 FAST proved that Batch-5/6 files exist and can be
shallow-inspected. P3B performs a bounded, controlled smoke conversion for a very
small number of files.

## What this package does

- Selects a small number of Batch-5/6 raw files, default 1 per batch.
- Loads selected raw files only.
- Extracts measured current, measured voltage, optional temperature, and time.
- Reconstructs `cycle_id`, `step_id`, and `step_type` for smoke validation.
- Saves replay-profile NPZ files under the output directory.
- Generates a replay-profile smoke report and manifest.

## What this package does not do

- No model training.
- No replacement of D9.6/D9.5.1 or D12-S1K.
- No modification of `gv1/model.py`, `gv1/output_transform.py`, `gv1/losses.py`, `gv1/trainer.py`.
- No SOH generated inside the XJTU voltage soft-label generator.
- No P2D internal-state labels.
- No change to Batch-1_2C_battery-8 flagged/excluded policy.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p3b_batch56_replay_smoke.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -DataRoot "E:\XJTU battery dataset" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -P3FastDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit_fast" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3b_batch56_replay_smoke" `
  -FilesPerBatch 1 `
  -MaxSubrecordsPerFile 30 `
  -MaxTotalPointsPerProfile 120000 `
  -AllowWarn
```

## Expected runtime

This is a smoke test, not full profile generation. It should complete quickly for
one Batch-5 file and one Batch-6 file. If a file is unusually large, lower
`-MaxSubrecordsPerFile` to 5 or 10.
