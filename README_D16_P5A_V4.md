
# D16-P5A v4 — fixed existing-model transfer evaluation

This package fixes the previous D16-P5A failure where `eval_full_profiles` stayed empty because the runner tried to load incompatible D14/D12 `best.pt` checkpoints.

## What changed

- Accepts only D15-RG checkpoints named `best_with_state.pt`.
- Canonical model search order:
  1. `E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p2_rg_precision_benchmark/model/best_with_state.pt`
  2. `E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p1_rg_closedset_nn_smoke/model/best_with_state.pt`
  3. Recursive search for `best_with_state.pt` only, with D15/P2/P3/P1/RG paths ranked highest.
- Ignores D14/D12 `best.pt` files by design.
- Routes all 55 profiles into the existing model one-hot profile space, so an 8-cell D15-P2/P1 model can be evaluated on ALL55 without an out-of-range profile index.
- Writes actual prediction files to:
  `eval_full_profiles/predictions/*.npz`
- Runs D15-P2-style precision audit on those predictions.

## Smoke command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_v4.ps1 `
  -AllowOverwrite `
  -LimitCells 2 `
  -Device "cuda:0" `
  -BatchSize 65536
```

After smoke, verify this directory is not empty:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\predictions
```

Run the checker:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5a_v4_outputs.ps1
```

## Full ALL55 command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_v4.ps1 `
  -AllowOverwrite `
  -Device "cuda:0" `
  -BatchSize 65536
```

If GPU memory is insufficient, reduce batch size:

```powershell
-BatchSize 32768
```

or:

```powershell
-BatchSize 16384
```

## If D15-P2/P1 checkpoint is missing

The runner will search recursively for `best_with_state.pt` under the cache and project root. If no compatible D15-RG model exists, you can rebuild the D15-P2 existing benchmark from the D15-P0 8-cell labels:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_v4.ps1 `
  -AllowOverwrite `
  -LimitCells 2 `
  -RebuildExistingP2IfMissing `
  -Device "cuda:0" `
  -BatchSize 65536
```

This rebuild uses:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell
```

and writes:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark\model\best_with_state.pt
```

## Key outputs

Final scorecard:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\D16_P5A_FINAL_SCORECARD.json
```

Predictions used by audit:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\predictions\*.npz
```

Per-profile metrics:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv
```

Routing table:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_ROUTING_TABLE.csv
```

Batch summary:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_BATCH_METRICS.csv
```

Protocol summary:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv
```

Precision audit summary:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55\precision_audit\D15_P2_PRECISION_AUDIT_SUMMARY.json
```

## How to read result

In `D16_P5A_FINAL_SCORECARD.json`:

- `operational_status = PASS` means prediction generation actually ran and `eval_full_profiles/predictions` is not empty.
- `profile_count_evaluated = 55` and `prediction_file_count_primary = 55` means ALL55 ran.
- `final_status = PASS/REVIEW/FAIL` is the scientific audit status.
- `routing_table` tells you which existing model profile each ALL55 cell was routed to.

This is still an existing-model transfer evaluation, not a new ALL55 unified model training.
