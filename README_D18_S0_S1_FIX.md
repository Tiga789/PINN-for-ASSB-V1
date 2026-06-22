# D18-S0/S1-FIX

This package repairs the two issues found in the first D18 P0/S0/S1 run. It is a diagnostic-only overlay for `PINN-for-ASSB-V1`; it never launches D18-S2 training.

## What is fixed

### S0 physical output transform

- The combined radial basis is projected to zero volume mean and normalized to unit pointwise peak.
- `delta_cs` is bounded by the admissible concentration inventory margin, not merely by a coefficient magnitude.
- `cs_a/cs_c` must remain inside concentration bounds implied by the configured theta window.
- `theta_a/theta_c` are derived from concentration and must have zero out-of-range fraction.

### S1 real dense-array diagnosis

- The old broad NPZ scan is replaced by an explicit 8-case manifest.
- Dense `pred/true` arrays are re-exported on demand from the frozen D17-G21 checkpoint using D17-G6F-compatible feature construction.
- The casepack covers train, G2 internal-heldout, validation, RG, P4D, 2C, 3C, R2.5, R3, random-walk, GEO, and early/middle/late cycles.
- Frozen-test, test, flagged-probe, unknown split, and the known Batch-1 battery-8 probe are blocked.
- S1 cannot report a valid pass unless cycle boundaries, current, cumulative Ah, phase groups, cycle positions, split coverage, protocol coverage, and branch coverage are all present.
- JSON is strict (`NaN/Infinity` become `null`), and empty CSV files retain schema headers.

## Run order

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_D18_S0_S1_FIX.ps1
powershell -ExecutionPolicy Bypass -File .\RUN_D18_S0_S1_FIX.ps1
```

Default output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d18_fullcycle_fix
```

Upload that complete output directory as a ZIP for review. Do not start D18-S2 from this package.
