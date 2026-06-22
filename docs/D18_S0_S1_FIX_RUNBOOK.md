# D18-S0/S1-FIX Runbook

## Prerequisites

1. The package is copied over the project root.
2. The previous D18-P0 run completed and this file exists:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle\d18_p0_freeze\p0_freeze_manifest.json
```

3. The D17-G21 candidate remains available under:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d17_g\g21_p4d_branch_repair
```

4. D17 split, G0 semantics, checkpoint, replay profiles, and D15 ALL55 soft labels remain at the paths recorded by D17.

## Verification

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_S0_S1_FIX.ps1
```

Verification performs manifest hashes, Python compilation, unit tests, and a synthetic P0→S0→casepack→S1 dry-run. The synthetic fixture is only a package integrity check; formal S1 uses real D17 arrays.

## Formal run

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX.ps1
```

Without plots:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX.ps1 -NoPlots
```

Explicit output directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX.ps1 `
  -OutputRoot "E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle_fix"
```

## Expected outputs

```text
xjtu_d18_fullcycle_fix\
  D18_S0_S1_FIX_OVERALL_SUMMARY.json
  D18_S0_S1_FIX_OVERALL_STATUS.md
  d18_s0_architecture_fix\
    d18_s0_validation.json
    d18_s0_architecture_contract.json
  d18_s1_dense_casepack\
    D18_S1_DENSE_CASEPACK_SUMMARY.json
    D18_S1_DENSE_CASEPACK_MANIFEST.csv
    D18_S1_DENSE_CASEPACK_FAILURES.csv
    cases\*.npz
  d18_s1_array_diagnostic\
    d18_s1_array_latent_summary.json
    d18_s1_coverage_audit.json
    d18_s1_recommendation.md
    d18_s1_state_metrics.csv
    d18_s1_error_components_by_cycle.csv
    d18_s1_cycle_boundary_audit.csv
    d18_s1_residual_rank.csv
    d18_s1_radial_components.csv
    d18_s1_cycle_features.csv
    plots\
```

## Valid diagnostic status

A valid fixed run should normally show:

```text
prior P0                  PASS
S0                        PASS
dense casepack            PASS
S1 coverage               PASS
S1 status                 PASS_VALID_DIAGNOSTIC_COVERAGE
training_launched         false
go_to_s2                  false
frozen_test_used          false
```

`PASS_VALID_DIAGNOSTIC_COVERAGE` means the failure diagnosis is now valid; it does not authorize S2 automatically.

## Failure handling

- `REVIEW_INVALID_DIAGNOSTIC_COVERAGE`: do not train; inspect missing paths/cases in the casepack failure CSV and coverage audit.
- `REVIEW_NO_STRUCTURAL_FAILURE_DETECTED`: do not train; the selected arrays failed to reproduce the known dense failure.
- Missing prior P0: rerun the original P0 or correct `paths.prior_p0_manifest`.
- Checkpoint/semantics/split path error: correct only `configs/d18_s0_s1_fix.json`; do not point S1 back at a broad D17 root scan.
