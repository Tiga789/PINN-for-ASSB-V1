# PINN-for-ASSB-V1 · QJW-2

Research prototype for adapting a PINNSTRIPES-style physics-informed surrogate workflow to an NMC811 || Li-In all-solid-state battery (ASSB).

> Current status: **ASSB-D7 / ModelFin_112 deterministic wrapper**.  
> The latest stable five-target engineering package is `ModelFin_112_deterministic_wrapper`: frozen `ModelFin_107A` four-state predictions + deterministic ridge SOH prediction.

---

## 1. Project scope

This repository is being adapted for a specific all-solid-state NMC811 || Li-In cell. The current model family uses an effective SPM interpretation:

- positive electrode `c`: NMC811 composite positive electrode, represented by a spherical active particle;
- negative electrode `a`: Li-In/In foil, represented as an equivalent diffusion-length pseudo-particle;
- `ce` and `phie`: retained from the SPM notation, reinterpreted as effective quantities in the solid-state ionic network;
- charge/discharge: handled through the sign of the measured current `I(t)`; material identity and OCP functions are not switched by current sign.

The project currently emphasizes auditable modeling, strict split control, and reproducible evaluation over claiming generalization.

---

## 2. Latest stable result

### ModelFin_112 deterministic wrapper

`ModelFin_112_deterministic_wrapper` is a single auditable engineering wrapper directory. It combines:

1. **Frozen four-state source**: `ModelFin_107A` / 107A corrected state-evaluation NPZ.
2. **SOH source**: `ModelFin_112_deterministicSOH_ridge_g4`, a deterministic ridge SOH head trained under the strict30 protocol.

It outputs and evaluates five targets:

```text
cs_a, cs_c, phie, phis_c, SOH
```

Important boundary:

```text
This is one engineering wrapper / unified package.
It is NOT an end-to-end jointly trained single neural network.
It is NOT yet a cross-cell or cross-chemistry generalization model.
```

---

## 3. Latest five-target scorecard

From `EvalFin_112_deterministic_wrapper/five_target_compact_summary.csv`:

| target | source / split | n | MAE | RMSE | R2 | corr |
|---|---:|---:|---:|---:|---:|---:|
| cs_a | 107A state eval NPZ | 1,280,000 | 0.0234896 | 0.0333314 | 0.9952569 | 0.9976256 |
| cs_c | 107A state eval NPZ | 1,280,000 | 0.3922937 | 0.5413698 | 0.9892045 | 0.9962260 |
| phie | 107A state eval NPZ | 373,235 | 0.0061720 | 0.0072475 | 0.9985575 | 0.9994414 |
| phis_c | 107A state eval NPZ | 373,235 | 0.0094586 | 0.0111716 | 0.9984663 | 0.9996665 |
| SOH | deterministic ridge / test | 362 | 0.0029612 | 0.0034824 | 0.9903415 | 0.9995154 |

SOH audit fields:

```text
no_test_metrics_in_training_history = true
test_metrics_used_for_selection     = false
feature_mode                        = g4_all_strict
model_variant                       = deterministic_ridge_soh_head
```

---

## 4. Strict30 SOH protocol

The current SOH prediction split is inherited from ASSB-D6/D7:

```text
train: cycles 5-139
val:   cycles 140-159
test:  cycles 160-521
partial/incomplete: cycle 522 is not used as held-out test
```

The SOH target comes from real cycle-level discharge capacity, not from soft labels or 107A state predictions.

---

## 5. Why deterministic ridge replaced the neural SOH head

D6 ended with `ModelFin_111_seed42locked_repro_c00`, where `saturating_v2` seed42 reached approximately:

```text
SOH test R2  ≈ 0.97934
SOH test MAE ≈ 0.00414
```

However, it was not multi-seed robust. D7 tested several neural SOH-head strategies:

- `robust_saturating` 5-seed sweep;
- stricter visible guard rules;
- soft-score checkpoint selection;
- progress/parallel script variants;
- v7 neural softscore head.

The neural route remained unstable. Example v7 neural results:

| seed | test R2 | test MAE |
|---:|---:|---:|
| 7 | 0.9687 | 0.00479 |
| 42 | 0.8381 | 0.01290 |
| 2026 | 0.9030 | 0.00898 |
| 3407 | 0.8632 | 0.01178 |
| 7890 | 0.9428 | 0.00736 |

The deterministic ridge head on the G4 feature group is therefore the current stable SOH baseline.

---

## 6. Main artifacts

Current key directories:

```text
ModelFin_107A
EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only
Data/assb112_feature_audit_v1
EvalFin_112_feature_audit_v1
ModelFin_112_deterministicSOH_ridge_g4
ModelFin_112_deterministic_wrapper
EvalFin_112_deterministic_wrapper
```

Current key scripts:

```text
scripts/train_assb112_deterministic_soh_baseline.py
scripts/run_assb112_deterministic_ridge_baseline.ps1
scripts/summarize_assb112_deterministic_baseline.py
scripts/build_ModelFin112_deterministic_wrapper.py
scripts/build_ModelFin112_single_model.py
scripts/run_ModelFin112_deterministic_wrapper_eval.ps1
evaluate_ModelFin112_deterministic_5targets.py
evaluate_ModelFin112_unified_5targets.py
util/assb_soh_feature_schema.py
util/assb112_deterministic_wrapper.py
```

Deprecated or not recommended as formal entry points:

```text
scripts/run_assb112_guarded_seed_sweep_parallel.ps1
older Start-Job / Receive-Job / Start-Process neural seed-sweep wrappers
ModelFin_112_v7_softscore_seed* as final candidates
ModelFin_112_robustSOH_seed* as final candidates
```

---

## 7. Reproduction commands

### 7.1 Deterministic ridge SOH baseline

```powershell
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

.\scripts\run_assb112_deterministic_ridge_baseline.ps1 `
  -Clean `
  -GpuReserveGB 2.0 `
  -GpuWorkRepeats 4
```

Expected key line:

```text
[RIDGE TEST] R2≈0.99034 MAE≈0.00296 RMSE≈0.00348 BIAS≈-0.00281 corr≈0.99952
```

### 7.2 Build and evaluate the deterministic five-target wrapper

```powershell
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

.\scripts\run_ModelFin112_deterministic_wrapper_eval.ps1 -Clean
```

Expected outputs:

```text
ModelFin_112_deterministic_wrapper/unified_config.json
ModelFin_112_deterministic_wrapper/build_audit.json
EvalFin_112_deterministic_wrapper/five_state_scorecard.csv
EvalFin_112_deterministic_wrapper/five_target_compact_summary.csv
EvalFin_112_deterministic_wrapper/unified_eval_audit.json
```

### 7.3 Inspect the scorecard

```powershell
Import-Csv .\EvalFin_112_deterministic_wrapper\five_target_compact_summary.csv |
  Format-Table variable,split,n,MAE,RMSE,R2,corr -AutoSize

Get-Content .\ModelFin_112_deterministic_wrapper\build_audit.json
Get-Content .\EvalFin_112_deterministic_wrapper\unified_eval_audit.json
```

---

## 8. Development timeline

| stage | summary |
|---|---|
| D1 | ASSB priors, v3 soft labels, first physics-only training; ModelFin_52 failed, requiring cycle5-only debugging. |
| D2 | cycle5_v4 closed-loop success; ModelFin_101 with I(t)-cbar baseline, asymmetric radial deviation, and current-aware potential baseline. |
| D3 | continuous cycle5-522 soft labels; ModelFin_102/103 long-sequence migration exposed positive-electrode concentration failure. |
| D4 | v2 mass-closed candidate, common-mode gauge, and ModelFin_107A four-state full-cycle calibration benchmark. |
| D5 | real capacity/SOH labels and aging-head experiments; best five-target result was still hybrid: 107A states + StageB SOH. |
| D6 | strict30 SOH prediction route; ModelFin_111 seed42 recovery baseline, but neural SOH not multi-seed robust. |
| D7 | feature audit + deterministic ridge SOH + ModelFin_112 deterministic five-target wrapper. |

---

## 9. Current limitations

1. `ModelFin_112_deterministic_wrapper` is an engineering wrapper, not an end-to-end jointly trained single neural network.
2. The four electrochemical states still come from the `ModelFin_107A` full-cycle soft-label calibration benchmark.
3. The deterministic SOH ridge head is a strict30 held-out predictor for this cell, not a validated cross-cell aging mechanism.
4. Current results do not prove generalization across different ASSB/LIB cell formats or chemistries.
5. Old neural SOH seed-sweep scripts are not recommended for formal reproduction because PowerShell job/error handling caused repeated workflow failures.

---

## 10. Next work: generalization stage

The next research stage should focus on generalization, not further blind SOH neural-head tuning. Recommended direction:

- introduce `CellSpec` / `ChemistrySpec` conditional inputs;
- separate cell-level constants from cycle-level health features;
- create cross-cell / cross-format train-val-test splits;
- validate on held-out cells, not just held-out cycles from the same cell;
- preserve strict no-test selection and audit files;
- use the deterministic ridge SOH result as a baseline that any new neural/aging-aware model must beat.

---

## 11. Repository warning

Do not assume the remote GitHub copy equals the latest local experiment state. This project has many local overwrite packages and manually backed-up experiment directories. For any new code change, record:

```text
files changed
files added
run command
audit outputs
scorecard outputs
whether test metrics were used for selection
```
