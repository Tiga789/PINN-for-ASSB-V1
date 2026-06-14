# D16-P5A fixed v3: existing-model transfer evaluation with automatic model discovery

This package fixes the v2 error:

```text
ModelDir not found: E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark
```

The runner no longer hard-fails when that default directory does not exist.  It auto-discovers compatible D15 RG NN checkpoints under the GV1 cache root and the project root.

## Files

```text
scripts/gv1_d16_p5a_existing_transfer_eval_fixed.py
scripts/gv1_run_d16_p5a_fixed.ps1
scripts/gv1_find_d15_existing_model_dirs.ps1
README_D16_P5A_FIXED_V3.md
PACKAGE_MANIFEST.json
```

No empty docs directory, no __pycache__, no .pyc files.

## Step 1: install / overwrite

Copy this zip into the project root:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

It only overwrites the two D16-P5A fixed runner files under `scripts/` and adds a model-finder helper.

## Step 2: smoke run with auto-discovery

Use the same command as before.  You do not need to provide `-ModelDir` unless you already know it.

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -AllowOverwrite `
  -LimitCells 2 `
  -Device "cuda:0" `
  -BatchSize 65536
```

Expected behavior:

1. The script prints `ModelDir=auto`.
2. Python writes:
   ```text
   E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\D16_P5A_MODEL_DISCOVERY.json
   ```
3. It selects a compatible checkpoint containing D15 RG NN state keys.
4. It writes predictions to:
   ```text
   E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\predictions
   ```

If no compatible checkpoint exists anywhere in the searched locations, the script writes:

```text
D16_P5A_MODEL_DISCOVERY_FAILURE.json
```

and prints the candidate checkpoint paths it found.  In that case, run the finder below.

## Optional: manually list candidate model dirs

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_find_d15_existing_model_dirs.ps1 `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
```

Then rerun with an explicit candidate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -AllowOverwrite `
  -LimitCells 2 `
  -ModelDir "<candidate ModelDir printed by finder>" `
  -Device "cuda:0" `
  -BatchSize 65536
```

## Step 3: full ALL55 run

After the 2-cell smoke writes `.npz` files in `eval_full_profiles\predictions`, run all 55:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 `
  -AllowOverwrite `
  -Device "cuda:0" `
  -BatchSize 65536
```

If GPU memory is insufficient, use:

```powershell
-BatchSize 32768
```

or:

```powershell
-BatchSize 16384
```

## Important outputs

Main scorecard:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\D16_P5A_FIXED_SCORECARD.json
```

Model discovery:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\D16_P5A_MODEL_DISCOVERY.json
```

Primary predictions used by the internal D15-P2 style precision audit:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\predictions
```

Per-profile raw/projected metrics:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv
```

Routing table:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\D16_P5A_ROUTING_TABLE.csv
```

Batch/protocol summaries:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\D16_P5A_BATCH_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv
```

Internal precision audit summary:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55\precision_audit\D15_P2_PRECISION_AUDIT_SUMMARY.json
```

## How to interpret status

In `D16_P5A_FIXED_SCORECARD.json`:

```text
operational_status = PASS
profile_count_predicted = 55
```

means the D16-P5A execution chain actually generated ALL55 predictions.  This is the first gate.

```text
final_status = PASS
```

means the existing model transferred cleanly under the configured thresholds.

```text
final_status = REVIEW
```

means predictions were generated but some metrics/audit checks require diagnosis before D16-P5B.

```text
final_status = FAIL
```

means operational failure such as no compatible model checkpoint, no predictions, or unreadable data.
